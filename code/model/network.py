import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import rend_util
from model.embedder import *
from model.density import LaplaceDensity
from model.ray_sampler import ErrorBoundSampler
import matplotlib.pyplot as plt
import numpy as np

class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            sdf_bounding_sphere,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0,
            sphere_scale=1.0,
            inside_outside=False,
    ):
        super().__init__()

        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.sphere_scale = sphere_scale
        dims = [d_in] + dims + [d_out + feature_vector_size]

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            dims[0] = input_ch
        # print(multires, dims)
        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)

                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

    def forward(self, input_dict):
        if self.embed_fn is not None:
            input_dict = self.embed_fn(input_dict)

        x = input_dict

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input_dict], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient_sdf(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:,:1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients_sdf = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients_sdf

    def get_outputs(self, x):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf = output[:,:1]
        ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        feature_vectors = output[:, 1:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients_sdf = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        return sdf, feature_vectors, gradients_sdf

    def get_sdf_vals(self, x):
        sdf = self.forward(x)[:,:1]
        ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        return sdf


from hashencoder.hashgrid import _hash_encode, HashEncoder
class ImplicitNetworkGrid(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            sdf_bounding_sphere,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0,
            sphere_scale=1.0,
            inside_outside=False,
            base_size = 16,
            end_size = 2048,
            logmap = 19,
            num_levels=16,
            level_dim=2,
            divide_factor = 1.5, # used to normalize the points range for multi-res grid
            use_grid_feature = True, 
            debug=False, 
    ):
        super().__init__()

        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.sphere_scale = sphere_scale
        dims = [d_in] + dims + [d_out + feature_vector_size]
        self.embed_fn = None
        self.divide_factor = divide_factor
        self.grid_feature_dim = num_levels * level_dim
        self.use_grid_feature = use_grid_feature
        dims[0] += self.grid_feature_dim

        self.debug = debug
        
        print(f"using hash encoder with {num_levels} levels, each level with feature dim {level_dim}")
        print(f"resolution:{base_size} -> {end_size} with hash map size {logmap}")
        self.encoding = HashEncoder(input_dim=3, num_levels=num_levels, level_dim=level_dim, 
                    per_level_scale=2, base_resolution=base_size, 
                    log2_hashmap_size=logmap, desired_resolution=end_size)
        
        '''
        # can also use tcnn for multi-res grid as it now supports eikonal loss
        base_size = 16
        hash = True
        smoothstep = True
        self.encoding = tcnn.Encoding(3, {
                        "otype": "HashGrid" if hash else "DenseGrid",
                        "n_levels": 16,
                        "n_features_per_level": 2,
                        "log2_hashmap_size": 19,
                        "base_resolution": base_size,
                        "per_level_scale": 1.34,
                        "interpolation": "Smoothstep" if smoothstep else "Linear"
                    })
        '''
        
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            dims[0] += input_ch - 3
        print("network architecture")
        print(dims)
        
        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
            
            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)

                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)
        self.cache_sdf = None

    def forward(self, input_dict):
        if self.use_grid_feature:
            # normalize point range as encoding assume points are in [-1, 1]
            feature = self.encoding(input_dict / self.divide_factor)
        else:
            feature = torch.zeros_like(input_dict[:, :1].repeat(1, self.grid_feature_dim))
                    
        if self.embed_fn is not None:
            embed = self.embed_fn(input_dict)
            input_dict = torch.cat((embed, feature), dim=-1)
        else:
            input_dict = torch.cat((input_dict, feature), dim=-1)

        x = input_dict

        for l in range(0, self.num_layers - 1): # 0, 1, 2
            lin = getattr(self, "lin" + str(l))
            if l in self.skip_in:
                x = torch.cat([x, input_dict], 1) / np.sqrt(2)
            if self.debug: print(l, '-->', x.shape, lin)
            x = lin(x)
            if l < self.num_layers - 2: # 0, 1
                x = self.softplus(x)
            if self.debug: print(l, x.shape, '-->')
                
        return {
            'sdf': x[:, :1], 
            'feature': x[:, 1:]
            }

    def gradient_sdf(self, x):
        x.requires_grad_(True)
        y = self.forward(x)['sdf']
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients_sdf = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients_sdf

    def get_outputs(self, x):
        x.requires_grad_(True)
        output_dict = self.forward(x)
        sdf = output_dict['sdf']

        feature_vectors = output_dict['feature']
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients_sdf = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        return sdf, feature_vectors, gradients_sdf

    def get_sdf_vals(self, x):
        sdf = self.forward(x)['sdf']
        return sdf

    def mlp_parameters(self):
        parameters = []
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            parameters += list(lin.parameters())
        return parameters

    def grid_parameters(self):
        print("grid parameters", len(list(self.encoding.parameters())))
        for p in self.encoding.parameters():
            print(p.shape)
        return self.encoding.parameters()


class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            multires_view=0,
            per_image_code = False, 
            if_hdr=False, 
            spec = False, # separate spec from diffuse in pred. RGB, following rad-MLP
            debug = False, 

    ):
        super().__init__()

        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]

        self.debug = debug
        self.spec = spec

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.per_image_code = per_image_code
        if self.per_image_code:
            # nerf in the wild parameter
            # parameters
            # maximum 1024 images
            self.embeddings = nn.Parameter(torch.empty(1024, 32))
            std = 1e-4
            self.embeddings.data.uniform_(-std, std)
            dims[0] += 32

        print("rendering network architecture:")
        print(dims)

        self.num_layers = len(dims)
        self.if_hdr = if_hdr
        # assert self.if_hdr, 'for now'

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)
            if self.spec:
                if l == self.num_layers-3:
                    lin = nn.Linear(dims[l]-3, out_dim)
            if self.debug: print(l, lin)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, points, normals, view_dirs, feature_vectors, indices, if_pixel_input=False):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'nerf':
            rendering_input = torch.cat([view_dirs, feature_vectors], dim=-1)
        else:
            raise NotImplementedError

        if self.per_image_code:
            '''
            self.embeddings: (max_images, D)
            rendering_input.shape: (num_images*num_pixes, D')

            if_pixel_input = False: indices: (num_images)
            if_pixel_input = True: indices: (num_images*num_pixels)
            '''
            if not if_pixel_input:
                image_code = self.embeddings[indices].expand(rendering_input.shape[0], -1) # (num_images=1, D) -> (num_images=1*num_pixels*num_samples), D) -> 
            else:
                num_samples = rendering_input.shape[0] // indices.shape[0]
                image_code = self.embeddings[indices].unsqueeze(1).expand(-1, num_samples, -1).flatten(0, 1) # (num_pixels, D) -> (num_pixels*num_samples, D)
            rendering_input = torch.cat([rendering_input, image_code], dim=-1)
            
        x = rendering_input

        # for l in range(0, self.num_layers - 1):
        #     lin = getattr(self, "lin" + str(l))

        #     if self.debug: print(l, '-->', x.shape, lin)
        #     x = lin(x)

        #     if l < self.num_layers - 2:
        #         x = self.relu(x)
        #     if self.debug: print(l, x.shape, '-->')

        if self.spec:
            assert self.if_hdr
            for l in range(0, self.num_layers - 3): # 0, 1, 2
                lin = getattr(self, "lin" + str(l))
                if self.debug: print(l, '-->', x.shape, lin)
                x = lin(x)
                # if l < self.num_layers - 2: # 0, 1
                x = self.relu(x)
                if self.debug: print(l, x.shape, '-->')

            color_diff, x = x[:, :3], x[:, 3:]

            for l in range(self.num_layers - 3, self.num_layers - 1): # 3, 4
                lin = getattr(self, "lin" + str(l))
                if self.debug: print(l, '-->', x.shape, lin)
                x = lin(x)
                # if l < self.num_layers - 1: # 3
                x = self.relu(x)
                if self.debug: print(l, x.shape, '-->')

            color_spec = x
            rgb = color_diff + color_spec

            return {
                'rgb': rgb, 
                'rgb_diff': color_diff,
                'rgb_spec': color_spec,
            }

        else:
            for l in range(0, self.num_layers - 1): # 0, 1, 2
                lin = getattr(self, "lin" + str(l))
                if self.debug: print(l, '-->', x.shape, lin)
                x = lin(x)
                if l < self.num_layers - 2: # 0, 1
                    x = self.relu(x)
                if self.debug: print(l, x.shape, '-->')
                    
            if self.if_hdr:
                x = self.relu(x)
            else:
                x = self.sigmoid(x)

            return {'rgb': x}

class MonoSDFNetwork(nn.Module):

    def __init__(
        self, 
        conf, 
        if_hdr=False, 
        ):

        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.scene_bounding_sphere = conf.get_float('scene_bounding_sphere', default=1.0)
        self.white_bkgd = conf.get_bool('white_bkgd', default=False)
        self.bg_color = torch.tensor(conf.get_list("bg_color", default=[1.0, 1.0, 1.0])).float().cuda()
        self.if_hdr = if_hdr

        Grid_MLP = conf.get_bool('Grid_MLP', default=False)
        self.Grid_MLP = Grid_MLP
        if Grid_MLP:
            self.implicit_network = ImplicitNetworkGrid(self.feature_vector_size, 0.0 if self.white_bkgd else self.scene_bounding_sphere, **conf.get_config('implicit_network'))    
        else:
            self.implicit_network = ImplicitNetwork(self.feature_vector_size, 0.0 if self.white_bkgd else self.scene_bounding_sphere, **conf.get_config('implicit_network'))
        
        self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'), if_hdr=self.if_hdr)
        self.spec = conf.get_config('rendering_network').get_bool('spec', False)
        
        self.density = LaplaceDensity(**conf.get_config('density'))
        sampling_method = conf.get_string('sampling_method', default="errorbounded")
        self.ray_sampler = ErrorBoundSampler(self.scene_bounding_sphere, **conf.get_config('ray_sampler'))
        

    def forward(self, input_dict, indices, if_pixel_input=False):
        # Parse model input
        if not if_pixel_input:
            intrinsics = input_dict["intrinsics"]
            uv = input_dict["uv"]
            pose = input_dict["pose"]

            ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)
            # ray_dirs, cam_loc = ray_dirs.cuda(), cam_loc.cuda()
            
            # we should use unnormalized ray direction for depth
            ray_dirs_tmp, _ = rend_util.get_camera_params(uv, torch.eye(4).to(pose.device)[None], intrinsics)
            # ray_dirs_tmp = ray_dirs_tmp.cuda() # (batch_size, N_pixels_sample, 2)

            cam_loc = cam_loc.unsqueeze(1).repeat(1, ray_dirs.shape[1], 1).reshape(-1, 3)
        else:
            ray_dirs = input_dict['ray_dirs'].unsqueeze(0) # (1, N_pixels, 3)
            cam_loc = input_dict['ray_cam_loc'] # (N_pixels, 3)
            ray_dirs_tmp = input_dict['ray_dirs_tmp'].unsqueeze(0) # (1, N_pixels, 3)

        depth_scale = ray_dirs_tmp[0, :, 2:]
        
        batch_size, num_pixels, _ = ray_dirs.shape

        ray_dirs = ray_dirs.reshape(-1, 3)

        
        z_vals, z_samples_eik = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self)
        N_samples = z_vals.shape[1]

        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1) # (1024, 1, 3) + (1024, 98, 1) * (1024, 1, 3)
        points_flat = points.reshape(-1, 3) # (1024, N_samples, 3) -> (1024*N_samples, 3)

        dirs = ray_dirs.unsqueeze(1).repeat(1,N_samples,1)
        dirs_flat = dirs.reshape(-1, 3)

        sdf, feature_vectors, gradients_sdf = self.implicit_network.get_outputs(points_flat)
        
        '''
        if_pixel_input=False: indices.shape = (batch_size)
        if_pixel_input=True: indices.shape = (batch_size, num_pixels)
        '''

        # points_flat: (N_pixels*N_samples, 3)
        rendering_output_dict = self.rendering_network(points_flat, gradients_sdf, dirs_flat, feature_vectors, indices, if_pixel_input=if_pixel_input)
        rgb_flat = rendering_output_dict['rgb']
        rgb = rgb_flat.reshape(-1, N_samples, 3)

        weights = self.volume_rendering(z_vals, sdf)

        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)

        
        depth_values = torch.sum(weights * z_vals, 1, keepdims=True) / (weights.sum(dim=1, keepdims=True) +1e-8)
        # we should scale rendered distance to depth along z direction
        depth_values = depth_scale * depth_values
        
        # white background assumption
        if self.white_bkgd:
            acc_map = torch.sum(weights, -1)
            rgb_values = rgb_values + (1. - acc_map[..., None]) * self.bg_color.unsqueeze(0)

        output = {
            'rgb':rgb,
            'rgb_values': rgb_values,
            'depth_values': depth_values,
            'z_vals': z_vals,
            'depth_vals': z_vals * depth_scale,
            'sdf': sdf.reshape(z_vals.shape),
            'weights': weights,
        }

        if self.spec:
            rgb_spec_flat = rendering_output_dict['rgb_spec']
            rgb_spec = rgb_spec_flat.reshape(-1, N_samples, 3)
            rgb_spec_values = torch.sum(weights.unsqueeze(-1) * rgb_spec, 1)
            output.update({
                'rgb_spec': rgb_spec, 
                'rgb_spec_values': rgb_spec_values, 
            })
            
        if self.training:
            # Sample points for the eikonal loss
            n_eik_points = batch_size * num_pixels
            
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-self.scene_bounding_sphere, self.scene_bounding_sphere).cuda()

            # add some of the near surface points
            eik_near_points = (cam_loc.unsqueeze(1) + z_samples_eik.unsqueeze(2) * ray_dirs.unsqueeze(1)).reshape(-1, 3)
            eikonal_points = torch.cat([eikonal_points, eik_near_points], 0)
            # add some neighbour points as unisurf
            neighbour_points = eikonal_points + (torch.rand_like(eikonal_points) - 0.5) * 0.01   
            eikonal_points = torch.cat([eikonal_points, neighbour_points], 0)
                   
            grad_theta = self.implicit_network.gradient_sdf(eikonal_points)
            
            # split gradient to eikonal points and heighbour ponits
            output['grad_theta'] = grad_theta[:grad_theta.shape[0]//2]
            output['grad_theta_nei'] = grad_theta[grad_theta.shape[0]//2:]
        
        # compute normal map
        normals = gradients_sdf / (gradients_sdf.norm(2, -1, keepdim=True) + 1e-6)
        normals = normals.reshape(-1, N_samples, 3)
        normal_map = torch.sum(weights.unsqueeze(-1) * normals, 1)
        
        # transform to local coordinate system
        if if_pixel_input:
            rot = input_dict['ray_pose'][:, :3, :3].transpose(1, 2)
            normal_map = (rot @ normal_map.unsqueeze(-1)).squeeze(-1) # (N, 3)
        else:
            # normal_map_ = normal_map.clone()

            rot = pose[0, :3, :3].permute(1, 0).contiguous() # (3, 3)
            normal_map = rot @ normal_map.permute(1, 0) # ((3, 3) @ (3, N)).T = (N, 3) @ rot (3,3)
            normal_map = normal_map.permute(1, 0).contiguous()

            # rot = pose[0:1, :3, :3].expand(normal_map_.shape[0], -1, -1).transpose(1, 2) # (N, 3, 3)
            # normal_map_ = (rot @ normal_map_.unsqueeze(-1)).squeeze(-1) # (N, 3)

        
        output['normal_map'] = normal_map

        return output

    def volume_rendering(self, z_vals, sdf):
        density_flat = self.density(sdf)
        density = density_flat.reshape(-1, z_vals.shape[1])  # (batch_size * num_pixels) x N_samples

        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat([dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(dists.shape[0], 1)], -1)

        # LOG SPACE
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy[:, :-1]], dim=-1)  # shift one step
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        weights = alpha * transmittance # probability of the ray hits something here

        return weights
