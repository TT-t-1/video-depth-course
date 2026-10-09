from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from external.gmflow.utils.flow_viz import flow_to_image

sequence = "abandonedfactory/Hard/abandonedfactory/abandonedfactory/Hard/P000"
rgb_dir = Path("outputs/flow_smoke/TartanAir") / sequence / "image_left"
flow_dir = Path("outputs/flow_smoke/TartanAir_flow") / sequence

flow = np.load(flow_dir / "f_flow/000000_left.npy").astype(np.float32)
mask = np.load(flow_dir / "f_mask/000000_left.npy")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].imshow(Image.open(rgb_dir / "000000_left.png"))
axes[0, 0].set_title("RGB frame 0")

axes[0, 1].imshow(Image.open(rgb_dir / "000001_left.png"))
axes[0, 1].set_title("RGB frame 1")

axes[1, 0].imshow(flow_to_image(flow))
axes[1, 0].set_title("Forward flow: frame 0 -> frame 1")

axes[1, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
axes[1, 1].set_title("Unreliable pixels: white")

for ax in axes.flat:
    ax.axis("off")

fig.tight_layout()
fig.savefig("outputs/sample_flow.png", dpi=150)
plt.close(fig)
print("已保存到 outputs/sample_flow.png")

import torch
import torch.nn.functional as F

rgb1 = np.array(Image.open(rgb_dir / "000001_left.png"))
height, width = flow.shape[:2]

x, y = np.meshgrid(np.arange(width), np.arange(height))
sample_x = x + flow[:, :, 0]
sample_y = y + flow[:, :, 1]

grid = np.stack([
    2 * sample_x / (width - 1) - 1,
    2 * sample_y / (height - 1) - 1,
], axis=-1)
grid = torch.from_numpy(grid).float().unsqueeze(0)

image1 = torch.from_numpy(rgb1).permute(2, 0, 1).float() / 255
image1 = image1.unsqueeze(0)

warped = F.grid_sample(
    image1, grid,
    mode="bilinear",
    padding_mode="zeros",
    align_corners=True,
)
warped = warped[0].permute(1, 2, 0).numpy()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].imshow(Image.open(rgb_dir / "000000_left.png"))
axes[0].set_title("Frame 0")
axes[1].imshow(rgb1)
axes[1].set_title("Frame 1")
axes[2].imshow(warped)
axes[2].set_title("Frame 1 warped to frame 0")

for ax in axes:
    ax.axis("off")

fig.tight_layout()
fig.savefig("outputs/sample_warp.png", dpi=150)
plt.close(fig)
print("已保存到 outputs/sample_warp.png")
