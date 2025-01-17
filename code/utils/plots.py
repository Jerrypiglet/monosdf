import numpy as np
import torch
from skimage import measure
import torchvision
import trimesh
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

from utils import rend_util
import utils.general as utils
from utils.general import trans_topil
from utils.utils_misc import vis_disp_colormap

# def gamma2_th(x):
#     return x ** (1./2.2)

def gamma2_th(x):
    mask = x <= 0.0031308
    ret = torch.empty_like(x)
    ret[mask] = 12.92*x[mask]
    mask = ~mask
    ret[mask] = 1.055*x[mask].pow(1/2.4) - 0.055
    return ret

def plot(
    implicit_network, indices, plot_data, path, epoch, img_res, plot_nimgs, resolution, grid_boundary, if_gt_plotted=None, 
    if_hdr=False, level=0, PREFIX='', 
    if_tensorboard=False, writer=None, tid=0, batch_id=0, 
    ):

    if plot_data is not None:
        cam_loc, cam_dir = rend_util.get_camera_for_plot(plot_data['pose'])

        # plot images
        print('-[def plot()] plot images...')
        if if_hdr:
            plot_data['rgb_eval'] = torch.clip(gamma2_th(plot_data['rgb_eval']), 0., 1.)
            plot_data['rgb_gt'] = torch.clip(gamma2_th(plot_data['rgb_gt']), 0., 1.)
        plot_images(plot_data['rgb_eval'], plot_data['rgb_gt'], path, epoch, plot_nimgs, img_res, indices, PREFIX=PREFIX)

        # plot normal maps
        print('-[def plot()] plot normal maps...')
        plot_normal_maps(plot_data['normal_map'], plot_data['normal_gt'], path, epoch, plot_nimgs, img_res, indices, PREFIX=PREFIX)

        # plot depth maps
        print('-[def plot()] plot depth maps...')
        plot_depth_maps(plot_data['depth_map'], plot_data['depth_gt'], path, epoch, plot_nimgs, img_res, indices, PREFIX=PREFIX)

        # concat output images to single large image
        print('-[def plot()] plot merge...')
        images = []
        for name in ["rendering", "depth", "normal"]:
            images.append(cv2.imread('{0}/{1}{2}_{3}_{4}.png'.format(path, PREFIX, name, epoch, indices[0])))        

        images = np.concatenate(images, axis=1)
        merge_path = '{0}/{1}merge_{2}_{3}.png'.format(path, PREFIX, epoch, indices[0])
        cv2.imwrite(merge_path, images)

        H, W = img_res[0], img_res[1]
        rgb_gt = plot_data['rgb_gt'].detach().cpu().squeeze().numpy().reshape(H, W, 3)
        rgb_gt = cv2.resize(rgb_gt, (W//2, H//2), interpolation=cv2.INTER_AREA)
        
        if if_tensorboard:
            # resized = cv2.resize(images[:, :, [2, 1, 0]], (images.shape[1]//2, images.shape[0]//2), interpolation=cv2.INTER_AREA)
            # writer.add_image('%s_merge/%d'%(PREFIX, batch_id), resized, tid, dataformats='HWC')
            mask = plot_data['mask'].detach().cpu().squeeze().numpy().reshape(H, W)

            rgb_est = plot_data['rgb_eval'].detach().cpu().squeeze().numpy().reshape(H, W, 3)
            rgb_est = cv2.resize(rgb_est, (W//2, H//2), interpolation=cv2.INTER_AREA)
            writer.add_image('%s_rgb_est/%d'%(PREFIX, batch_id), rgb_est, tid, dataformats='HWC')
            
            rgb_DIFF = np.abs(rgb_est - rgb_gt)
            writer.add_image('%s_rgb_DIFF/%d'%(PREFIX, batch_id), rgb_DIFF, tid, dataformats='HWC')

            depth_est = plot_data['depth_map'].detach().cpu().squeeze().numpy().reshape(H, W)
            depth_est_vis = vis_disp_colormap(depth_est, normalize=True, valid_mask=mask==1., cmap_name='viridis')[0]
            depth_est_vis = cv2.resize(depth_est_vis, (W//2, H//2), interpolation=cv2.INTER_AREA)
            writer.add_image('%s_depth_est/%d'%(PREFIX, batch_id), depth_est_vis, tid, dataformats='HWC')

            normal_est = plot_data['normal_map'].detach().cpu().squeeze().numpy().reshape(H, W, 3)
            normal_est_vis = cv2.resize(normal_est, (W//2, H//2), interpolation=cv2.INTER_AREA)
            writer.add_image('%s_normal_est/%d'%(PREFIX, batch_id), normal_est_vis, tid, dataformats='HWC')
            
            normal_est_vis_OVERLAY = normal_est_vis * 0.5 + rgb_gt * 0.5
            writer.add_image('%s_normal_est_OVERLAY/%d'%(PREFIX, batch_id), normal_est_vis_OVERLAY, tid, dataformats='HWC')

            if if_gt_plotted is not None and not if_gt_plotted[PREFIX]:
                mask_vis = cv2.resize(mask, (W//2, H//2), interpolation=cv2.INTER_NEAREST)
                writer.add_image('%s_mask/%d'%(PREFIX, batch_id), mask_vis, tid, dataformats='HW')

                writer.add_image('%s_rgb_gt/%d'%(PREFIX, batch_id), rgb_gt, tid, dataformats='HWC')

                depth_gt = plot_data['depth_gt'].detach().cpu().squeeze().numpy().reshape(H, W)
                depth_gt_vis = vis_disp_colormap(depth_gt, normalize=True, valid_mask=mask==1., cmap_name='viridis')[0]
                depth_gt_vis = cv2.resize(depth_gt_vis, (W//2, H//2), interpolation=cv2.INTER_AREA)
                writer.add_image('%s_depth_gt/%d'%(PREFIX, batch_id), depth_gt_vis, tid, dataformats='HWC')

                normal_gt = plot_data['normal_gt'].detach().cpu().squeeze().numpy().reshape(H, W, 3)
                normal_gt_vis = cv2.resize(normal_gt, (W//2, H//2), interpolation=cv2.INTER_AREA)
                writer.add_image('%s_normal_gt/%d'%(PREFIX, batch_id), normal_gt_vis, tid, dataformats='HWC')


avg_pool_3d = torch.nn.AvgPool3d(2, stride=2)
upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')

@torch.no_grad()
def get_surface_sliding(path, epoch, sdf, resolution=100, grid_boundary=[-2.0, 2.0], return_mesh=False, level=0, center=None, scale=None):
    # cropN = 512
    cropN = 128 if resolution < 512 else 512
    assert resolution % cropN == 0, 'resolution: %d, cropN: %d'%(resolution, cropN)
    resN = resolution
    # level = 0
    N = resN // cropN

    grid_min = [grid_boundary[0], grid_boundary[0], grid_boundary[0]]
    grid_max = [grid_boundary[1], grid_boundary[1], grid_boundary[1]]
    xs = np.linspace(grid_min[0], grid_max[0], N+1)
    ys = np.linspace(grid_min[1], grid_max[1], N+1)
    zs = np.linspace(grid_min[2], grid_max[2], N+1)

    print('xs', xs)
    print('ys', ys)
    print('zs', zs)
    meshes = []
    for i in range(N):
        for j in range(N):
            for k in range(N):
                print(i, j, k)
                x_min, x_max = xs[i], xs[i+1]
                y_min, y_max = ys[j], ys[j+1]
                z_min, z_max = zs[k], zs[k+1]

                x = np.linspace(x_min, x_max, cropN)
                y = np.linspace(y_min, y_max, cropN)
                z = np.linspace(z_min, z_max, cropN)
                # print('[def get_surface_sliding()] np.meshgrid() with res %d... this could be slow...'%x.shape[0])
                # xx, yy, zz = np.meshgrid(x, y, z, indexing='ij') # this is really slow...
                # points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda()
                # print('[def get_surface_sliding()] np.meshgrid() DONE.')
                xx, yy, zz = torch.meshgrid(torch.tensor(x), torch.tensor(y), torch.tensor(z)) # this is really slow...
                points = torch.vstack([xx.flatten(), yy.flatten(), zz.flatten()]).T.float().cuda()
                
                def evaluate(points):
                    z = []
                    for _, pnts in enumerate(torch.split(points, 100000, dim=0)):
                        _ = sdf(pnts).detach().cpu().clone()
                        z.append(_)
                    z = torch.cat(z, axis=0)
                    return z

                # construct point pyramids
                points = points.reshape(cropN, cropN, cropN, 3).permute(3, 0, 1, 2)
                points_pyramid = [points]
                for _ in range(3):            
                    points = avg_pool_3d(points[None])[0]
                    points_pyramid.append(points)
                points_pyramid = points_pyramid[::-1]
                
                # evalute pyramid with mask
                mask = None
                threshold = 2 * (x_max - x_min)/cropN * 8
                print('[def get_surface_sliding()] evalute %d pyramids with mask...'%len(points_pyramid))
                for pid, pts in tqdm(enumerate(points_pyramid)):
                    coarse_N = pts.shape[-1]
                    pts = pts.reshape(3, -1).permute(1, 0).contiguous()
                    
                    if mask is None:    
                        pts_sdf = evaluate(pts)
                    else:                    
                        mask = mask.reshape(-1)
                        pts_to_eval = pts[mask]
                        #import pdb; pdb.set_trace()
                        if pts_to_eval.shape[0] > 0:
                            pts_sdf_eval = evaluate(pts_to_eval.contiguous())
                            pts_sdf[mask] = pts_sdf_eval
                        print("ratio", pts_to_eval.shape[0] / pts.shape[0])

                    if pid < 3:
                        # update mask
                        mask = torch.abs(pts_sdf) < threshold
                        mask = mask.reshape(coarse_N, coarse_N, coarse_N)[None, None]
                        mask = upsample(mask.float()).bool()

                        pts_sdf = pts_sdf.reshape(coarse_N, coarse_N, coarse_N)[None, None]
                        pts_sdf = upsample(pts_sdf)
                        pts_sdf = pts_sdf.reshape(-1)

                    threshold /= 2.

                print('[def get_surface_sliding()] evalute %d pyramids with mask... DONE.'%len(points_pyramid))

                z = pts_sdf.detach().cpu().numpy()

                print('[def get_surface_sliding()] measure.marching_cubes...')
                if (not (np.min(z) > level or np.max(z) < level)):
                    z = z.astype(np.float32)
                    verts, faces, normals, values = measure.marching_cubes(
                    volume=z.reshape(cropN, cropN, cropN), #.transpose([1, 0, 2]),
                    level=level,
                    spacing=(
                            (x_max - x_min)/(cropN-1),
                            (y_max - y_min)/(cropN-1),
                            (z_max - z_min)/(cropN-1) ))
                    print(np.array([x_min, y_min, z_min]))
                    print(verts.min(), verts.max())
                    verts = verts + np.array([x_min, y_min, z_min])
                    print(verts.min(), verts.max())
                    
                    if scale is not None and center is not None:
                        verts = verts / scale + center.reshape((1, 3))
                    
                    meshcrop = trimesh.Trimesh(verts, faces, normals)
                    #meshcrop.export(f"{i}_{j}_{k}.ply")
                    meshes.append(meshcrop)
                print('[def get_surface_sliding()] measure.marching_cubes... DONE.')

    print('[def get_surface_sliding()] trimesh.util.concatenate...')
    combined = trimesh.util.concatenate(meshes)
    print('[def get_surface_sliding()] trimesh.util.concatenate... DONE.')

    if return_mesh:
        return combined
    else:
        combined.export('{0}/surface_{1}.ply'.format(path, epoch), 'ply')    
        
def get_3D_scatter_trace(points, name='', size=3, caption=None):
    assert points.shape[1] == 3, "3d scatter plot input points are not correctely shaped "
    assert len(points.shape) == 2, "3d scatter plot input points are not correctely shaped "

    trace = go.Scatter3d(
        x=points[:, 0].cpu(),
        y=points[:, 1].cpu(),
        z=points[:, 2].cpu(),
        mode='markers',
        name=name,
        marker=dict(
            size=size,
            line=dict(
                width=2,
            ),
            opacity=1.0,
        ), text=caption)

    return trace


def get_3D_quiver_trace(points, directions, color='#bd1540', name=''):
    assert points.shape[1] == 3, "3d cone plot input points are not correctely shaped "
    assert len(points.shape) == 2, "3d cone plot input points are not correctely shaped "
    assert directions.shape[1] == 3, "3d cone plot input directions are not correctely shaped "
    assert len(directions.shape) == 2, "3d cone plot input directions are not correctely shaped "

    trace = go.Cone(
        name=name,
        x=points[:, 0].cpu(),
        y=points[:, 1].cpu(),
        z=points[:, 2].cpu(),
        u=directions[:, 0].cpu(),
        v=directions[:, 1].cpu(),
        w=directions[:, 2].cpu(),
        sizemode='absolute',
        sizeref=0.125,
        showscale=False,
        colorscale=[[0, color], [1, color]],
        anchor="tail"
    )

    return trace


def get_surface_trace(path, epoch, sdf, resolution=100, grid_boundary=[-2.0, 2.0], return_mesh=False, level=0):
    grid = get_grid_uniform(resolution, grid_boundary)
    points = grid['grid_points']

    z = []
    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts.cuda()).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    if (not (np.min(z) > level or np.max(z) < level)):

        z = z.astype(np.float32)

        verts, faces, normals, values = measure.marching_cubes(
            volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                             grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=level,
            spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1]))

        verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])
        '''
        I, J, K = faces.transpose()
        traces = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                            i=I, j=J, k=K, name='implicit_surface',
                            color='#ffffff', opacity=1.0, flatshading=False,
                            lighting=dict(diffuse=1, ambient=0, specular=0),
                            lightposition=dict(x=0, y=0, z=-1), showlegend=True)]
        '''
        meshexport = trimesh.Trimesh(verts, faces, normals)
        meshexport.export('{0}/surface_{1}.ply'.format(path, epoch), 'ply')

        if return_mesh:
            return meshexport
        #return traces
    return None

def get_surface_high_res_mesh(sdf, resolution=100, grid_boundary=[-2.0, 2.0], level=0, take_components=True):
    # get low res mesh to sample point cloud
    grid = get_grid_uniform(100, grid_boundary)
    z = []
    points = grid['grid_points']

    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    z = z.astype(np.float32)

    verts, faces, normals, values = measure.marching_cubes(
        volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                         grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
        level=level,
        spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                 grid['xyz'][0][2] - grid['xyz'][0][1],
                 grid['xyz'][0][2] - grid['xyz'][0][1]))

    verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

    mesh_low_res = trimesh.Trimesh(verts, faces, normals)
    if take_components:
        components = mesh_low_res.split(only_watertight=False)
        areas = np.array([c.area for c in components], dtype=np.float)
        mesh_low_res = components[areas.argmax()]

    recon_pc = trimesh.sample.sample_surface(mesh_low_res, 10000)[0]
    recon_pc = torch.from_numpy(recon_pc).float().cuda()

    # Center and align the recon pc
    s_mean = recon_pc.mean(dim=0)
    s_cov = recon_pc - s_mean
    s_cov = torch.mm(s_cov.transpose(0, 1), s_cov)
    vecs = torch.view_as_real(torch.linalg.eig(s_cov)[1].transpose(0, 1))[:, :, 0]
    if torch.det(vecs) < 0:
        vecs = torch.mm(torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]]).cuda().float(), vecs)
    helper = torch.bmm(vecs.unsqueeze(0).repeat(recon_pc.shape[0], 1, 1),
                       (recon_pc - s_mean).unsqueeze(-1)).squeeze()

    grid_aligned = get_grid(helper.cpu(), resolution)

    grid_points = grid_aligned['grid_points']

    g = []
    for i, pnts in enumerate(torch.split(grid_points, 100000, dim=0)):
        g.append(torch.bmm(vecs.unsqueeze(0).repeat(pnts.shape[0], 1, 1).transpose(1, 2),
                           pnts.unsqueeze(-1)).squeeze() + s_mean)
    grid_points = torch.cat(g, dim=0)

    # MC to new grid
    points = grid_points
    z = []
    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    meshexport = None
    if (not (np.min(z) > level or np.max(z) < level)):

        z = z.astype(np.float32)

        verts, faces, normals, values = measure.marching_cubes(
            volume=z.reshape(grid_aligned['xyz'][1].shape[0], grid_aligned['xyz'][0].shape[0],
                             grid_aligned['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=level,
            spacing=(grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                     grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                     grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1]))

        verts = torch.from_numpy(verts).cuda().float()
        verts = torch.bmm(vecs.unsqueeze(0).repeat(verts.shape[0], 1, 1).transpose(1, 2),
                   verts.unsqueeze(-1)).squeeze()
        verts = (verts + grid_points[0]).cpu().numpy()

        meshexport = trimesh.Trimesh(verts, faces, normals)

    return meshexport


def get_surface_by_grid(grid_params, sdf, resolution=100, level=0, higher_res=False):
    grid_params = grid_params * [[1.5], [1.0]]

    # params = PLOT_DICT[scan_id]
    input_min = torch.tensor(grid_params[0]).float()
    input_max = torch.tensor(grid_params[1]).float()

    if higher_res:
        # get low res mesh to sample point cloud
        grid = get_grid(None, 100, input_min=input_min, input_max=input_max, eps=0.0)
        z = []
        points = grid['grid_points']

        for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
            z.append(sdf(pnts).detach().cpu().numpy())
        z = np.concatenate(z, axis=0)

        z = z.astype(np.float32)

        verts, faces, normals, values = measure.marching_cubes(
            volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                             grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=level,
            spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1]))

        verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

        mesh_low_res = trimesh.Trimesh(verts, faces, normals)
        components = mesh_low_res.split(only_watertight=False)
        areas = np.array([c.area for c in components], dtype=np.float)
        mesh_low_res = components[areas.argmax()]

        recon_pc = trimesh.sample.sample_surface(mesh_low_res, 10000)[0]
        recon_pc = torch.from_numpy(recon_pc).float().cuda()

        # Center and align the recon pc
        s_mean = recon_pc.mean(dim=0)
        s_cov = recon_pc - s_mean
        s_cov = torch.mm(s_cov.transpose(0, 1), s_cov)
        vecs = torch.view_as_real(torch.linalg.eig(s_cov)[1].transpose(0, 1))[:, :, 0]
        if torch.det(vecs) < 0:
            vecs = torch.mm(torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]]).cuda().float(), vecs)
        helper = torch.bmm(vecs.unsqueeze(0).repeat(recon_pc.shape[0], 1, 1),
                           (recon_pc - s_mean).unsqueeze(-1)).squeeze()

        grid_aligned = get_grid(helper.cpu(), resolution, eps=0.01)
    else:
        grid_aligned = get_grid(None, resolution, input_min=input_min, input_max=input_max, eps=0.0)

    grid_points = grid_aligned['grid_points']

    if higher_res:
        g = []
        for i, pnts in enumerate(torch.split(grid_points, 100000, dim=0)):
            g.append(torch.bmm(vecs.unsqueeze(0).repeat(pnts.shape[0], 1, 1).transpose(1, 2),
                               pnts.unsqueeze(-1)).squeeze() + s_mean)
        grid_points = torch.cat(g, dim=0)

    # MC to new grid
    points = grid_points
    z = []
    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    meshexport = None
    if (not (np.min(z) > level or np.max(z) < level)):

        z = z.astype(np.float32)

        verts, faces, normals, values = measure.marching_cubes(
            volume=z.reshape(grid_aligned['xyz'][1].shape[0], grid_aligned['xyz'][0].shape[0],
                             grid_aligned['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=level,
            spacing=(grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                     grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                     grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1]))

        if higher_res:
            verts = torch.from_numpy(verts).cuda().float()
            verts = torch.bmm(vecs.unsqueeze(0).repeat(verts.shape[0], 1, 1).transpose(1, 2),
                       verts.unsqueeze(-1)).squeeze()
            verts = (verts + grid_points[0]).cpu().numpy()
        else:
            verts = verts + np.array([grid_aligned['xyz'][0][0], grid_aligned['xyz'][1][0], grid_aligned['xyz'][2][0]])

        meshexport = trimesh.Trimesh(verts, faces, normals)

        # CUTTING MESH ACCORDING TO THE BOUNDING BOX
        if higher_res:
            bb = grid_params
            transformation = np.eye(4)
            transformation[:3, 3] = (bb[1,:] + bb[0,:])/2.
            bounding_box = trimesh.creation.box(extents=bb[1,:] - bb[0,:], transform=transformation)

            meshexport = meshexport.slice_plane(bounding_box.facets_origin, -bounding_box.facets_normal)

    return meshexport

def get_grid_uniform(resolution, grid_boundary=[-2.0, 2.0]):
    x = np.linspace(grid_boundary[0], grid_boundary[1], resolution)
    y = x
    z = x

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)

    return {"grid_points": grid_points,
            "shortest_axis_length": 2.0,
            "xyz": [x, y, z],
            "shortest_axis_index": 0}

def get_grid(points, resolution, input_min=None, input_max=None, eps=0.1):
    if input_min is None or input_max is None:
        input_min = torch.min(points, dim=0)[0].squeeze().numpy()
        input_max = torch.max(points, dim=0)[0].squeeze().numpy()

    bounding_box = input_max - input_min
    shortest_axis = np.argmin(bounding_box)
    if (shortest_axis == 0):
        x = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(x) - np.min(x)
        y = np.arange(input_min[1] - eps, input_max[1] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
    elif (shortest_axis == 1):
        y = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(y) - np.min(y)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
    elif (shortest_axis == 2):
        z = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(z) - np.min(z)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))
        y = np.arange(input_min[1] - eps, input_max[1] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda()
    return {"grid_points": grid_points,
            "shortest_axis_length": length,
            "xyz": [x, y, z],
            "shortest_axis_index": shortest_axis}


def plot_normal_maps(normal_maps, ground_true, path, epoch, plot_nrow, img_res, indices, PREFIX=''):
    ground_true = ground_true.cuda()
    normal_maps = torch.cat((normal_maps, ground_true), dim=0)
    normal_maps_plot = lin2img(normal_maps, img_res)

    tensor = torchvision.utils.make_grid(normal_maps_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    img.save('{0}/{1}normal_{2}_{3}.png'.format(path, PREFIX, epoch, indices[0]))

    #import pdb; pdb.set_trace()
    #trans_topil(normal_maps_plot[0, :, :, 260:260+680]).save('{0}/2normal_{1}.png'.format(path, epoch))


def plot_images(rgb_points, ground_true, path, epoch, plot_nrow, img_res, indices, exposure=False, PREFIX=''):
    ground_true = ground_true.cuda()

    output_vs_gt = torch.cat((rgb_points, ground_true), dim=0)
    output_vs_gt_plot = lin2img(output_vs_gt, img_res)

    tensor = torchvision.utils.make_grid(output_vs_gt_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()

    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    if exposure:
        img.save('{0}/{1}exposure_{2}_{3}.png'.format(path, PREFIX, epoch, indices[0]))
    else:
        img.save('{0}/{1}rendering_{2}_{3}.png'.format(path, PREFIX, epoch, indices[0]))


def plot_depth_maps(depth_maps, ground_true, path, epoch, plot_nrow, img_res, indices, PREFIX=''):
    ground_true = ground_true.cuda()
    depth_maps = torch.cat((depth_maps[..., None], ground_true), dim=0)
    depth_maps_plot = lin2img(depth_maps, img_res)
    depth_maps_plot = depth_maps_plot.expand(-1, 3, -1, -1)

    tensor = torchvision.utils.make_grid(depth_maps_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    
    save_path = '{0}/{1}depth_{2}_{3}.png'.format(path, PREFIX, epoch, indices[0])
    
    plt.imsave(save_path, tensor[:, :, 0], cmap='viridis')
    

def lin2img(tensor, img_res):
    batch_size, num_samples, channels = tensor.shape
    return tensor.permute(0, 2, 1).view(batch_size, channels, img_res[0], img_res[1])