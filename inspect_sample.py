from pathlib import Path
import numpy as np
from PIL import Image

sequence = Path(
    "datasets/TartanAir/abandonedfactory/Hard/"
    "abandonedfactory/abandonedfactory/Hard/P000"
)

rgb_path = sorted((sequence / "image_left").glob("*.png"))[0]
frame_id = rgb_path.name.split("_")[0]
depth_path = sequence / "depth_left" / f"{frame_id}_left_depth.npy"

rgb = np.array(Image.open(rgb_path).convert("RGB"))
depth = np.load(depth_path)

print("图片文件:", rgb_path.name)
print("深度文件:", depth_path.name)
print("图片形状:", rgb.shape)
print("图片数据类型:", rgb.dtype)
print("深度形状:", depth.shape)
print("深度数据类型:", depth.dtype)
print("深度最小值:", depth.min())
print("深度最大值:", depth.max())

import matplotlib.pyplot as plt

Path("outputs").mkdir(exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].imshow(rgb)
axes[0].set_title("RGB frame 000000")
axes[0].axis("off")

image = axes[1].imshow(1.0 / depth, cmap="magma")
axes[1].set_title("Ground-truth inverse depth")
axes[1].axis("off")
fig.colorbar(image, ax=axes[1])

fig.tight_layout()
fig.savefig("outputs/sample_rgb_depth.png", dpi=150)
plt.close(fig)

print("图片已保存到 outputs/sample_rgb_depth.png")
