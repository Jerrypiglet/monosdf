from utils.utils_misc import blue_text
import torch
import numpy as np

total_pixels = 4136960
total_pixels_im = 320*640
sampling_size = 1024
_count = np.zeros(total_pixels, dtype=np.int32)
total_samples = 0

for i in range(5000):
    total_sampling_size = int(float(sampling_size) / float(total_pixels_im) * total_pixels)
    sampling_idx = torch.randperm(total_pixels)[:total_sampling_size]
    _count[np.array(sampling_idx)] = _count[np.array(sampling_idx)] + 1
    total_samples += total_sampling_size

    print(i, blue_text('sampling idx counts: shape; max, min; mean, median; exp'), _count.shape[0], np.amax(_count), np.amin(_count), np.mean(_count), np.median(_count), float(total_samples)/float(total_pixels))
