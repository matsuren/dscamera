import numpy as np
import PIL
import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage, ToTensor

from dscamera import DSCamera

if __name__ == "__main__":
    # Load camera and image
    json_file = "./calibration.json"
    cam = DSCamera(json_file)
    img = np.array(PIL.Image.open("./sample.jpg"))

    img_size = (512, 512)
    f = 0.25
    # Generate 3D points
    h, w = img_size
    z = f * min(img_size)
    x = np.arange(w) - w / 2
    y = np.arange(h) - h / 2
    x_grid, y_grid = np.meshgrid(x, y, indexing="xy")
    point3D = np.stack([x_grid, y_grid, np.full_like(x_grid, z)], axis=-1)

    # Project on image plane
    img_pts, valid_mask = cam.world2cam(point3D)
    img_pts[~valid_mask] = -1.0

    # To torch tensor
    src = ToTensor()(img)
    mapxy = torch.from_numpy(img_pts).float()
    mapxy[..., 0] = 2 * mapxy[..., 0] / cam.w - 1.0
    mapxy[..., 1] = 2 * mapxy[..., 1] / cam.h - 1.0
    grid = mapxy.view(1, h, w, 2)

    # Remap and show
    dst = F.grid_sample(src.unsqueeze(0), grid, align_corners=False)[0]
    ToPILImage()(src).show(title="Fisheye")
    ToPILImage()(dst).show(title="Perspective")
