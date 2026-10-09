"""验证第二课的数据层，并保存预处理过程图。"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import DepthVideoDataset, RandomScale, RandomHorizontalFlip, DistributedSamplerNoEvenlyDivisible
from external.gmflow.utils.flow_viz import flow_to_image


def audit_files(data_root):
    flow_root = data_root.with_name(data_root.name + "_flow")
    sequences = []
    for image_dir in sorted(data_root.rglob("image_left")):
        images = sorted(image_dir.glob("*.png"))
        ids = [int(p.stem.split("_")[0]) for p in images]
        assert all(b == a + 1 for a, b in zip(ids, ids[1:])), image_dir
        sequence_flow = flow_root / image_dir.parent.relative_to(data_root)
        expected = {image.with_suffix(".npy").name for image in images[:-1]}
        counts = {}
        for kind in ("f_flow", "b_flow", "f_mask", "b_mask"):
            actual = {p.name for p in (sequence_flow / kind).glob("*.npy")}
            assert actual == expected, (sequence_flow, kind, len(actual), len(expected))
            counts[kind] = len(actual)
            # 每条轨迹检查首尾输出的形状、数值和掩码；完整目录检查文件名覆盖率。
            for name in (images[0].with_suffix(".npy").name, images[-2].with_suffix(".npy").name):
                value = np.load(sequence_flow / kind / name)
                shape = (480, 640, 2) if "flow" in kind else (480, 640)
                assert value.shape == shape and value.dtype == np.float16
                assert np.isfinite(value).all()
                if "mask" in kind:
                    assert np.isin(value, (0, 1)).all()
        matched = sum((image_dir.parent / "depth_left" /
                       f"{p.stem.split('_')[0]}_left_depth.npy").is_file() for p in images)
        sequences.append({
            "sequence": str(image_dir.parent.relative_to(data_root)),
            "rgb_frames": len(images), "matched_depth_frames": matched,
            "unlabeled_rgb_frames": len(images) - matched, "flow_files": counts,
        })
    return {
        "sequences": sequences,
        "sequence_count": len(sequences),
        "rgb_frames": sum(s["rgb_frames"] for s in sequences),
        "matched_depth_frames": sum(s["matched_depth_frames"] for s in sequences),
        "unlabeled_rgb_frames": sum(s["unlabeled_rgb_frames"] for s in sequences),
        "adjacent_pairs": sum(s["flow_files"]["f_flow"] for s in sequences),
        "generated_files": sum(sum(s["flow_files"].values()) for s in sequences),
    }


def check_geometry_and_sampler():
    # T=2 的通道布局：RGB 6、逆深度 2、掩码 2、光流 4。
    packed = np.zeros((40, 60, 14), dtype=np.float32)
    packed[..., 6:8] = 2.0
    y, x = np.indices((40, 60))
    packed[..., 8:10] = ((x + y) % 2)[..., None]
    packed[..., 10:] = (10, -5, -10, 5)
    scaled = RandomScale((0.83, 0.83), last_ch=4)(packed)
    assert scaled.shape == (33, 50, 14)
    assert np.allclose(scaled[..., 6:8], 2.0)
    assert np.isin(scaled[..., 8:10], (0, 1)).all()
    assert np.allclose(scaled[..., 10:], (10 * 50 / 60, -5 * 33 / 40,
                                        -10 * 50 / 60, 5 * 33 / 40))
    np.random.seed(0)  # 第一个随机值大于 0.5，确定触发翻转。
    flipped = RandomHorizontalFlip(last_ch=4)(scaled)
    expected = np.flip(scaled, axis=1).copy()
    expected[..., 10::2] *= -1
    assert np.allclose(flipped, expected)

    samplers = [DistributedSamplerNoEvenlyDivisible(range(10), 3, rank, shuffle=True)
                for rank in range(3)]
    shards = [list(s) for s in samplers]
    assert sorted(sum(shards, [])) == list(range(10))
    assert [len(s) for s in samplers] == [4, 3, 3]
    samplers[0].set_epoch(1)
    assert list(samplers[0]) != shards[0]
    return {"scale": "PASS", "flip": "PASS", "sampler": "PASS"}


def save_preview(dataset, output_dir):
    sample = dataset[0]
    image0, image1 = ((sample["image"][i] + 1) / 2 for i in range(2))
    flow = sample["f_flows"][0]
    mask = sample["f_masks"][0, 0]
    height, width = image0.shape[-2:]
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    grid = torch.stack((2 * (x + flow[0]) / (width - 1) - 1,
                        2 * (y + flow[1]) / (height - 1) - 1), dim=-1)
    warped = F.grid_sample(image1[None], grid[None], align_corners=True)[0]
    valid = mask.bool()
    before = (image0 - image1).abs().mean(dim=0)[valid].mean().item()
    after = (image0 - warped).abs().mean(dim=0)[valid].mean().item()
    frame0 = dataset.data_paths[0][1][0][0].stem.split("_")[0]
    frame1 = dataset.data_paths[0][1][1][0].stem.split("_")[0]
    panels = [
        (image0.permute(1, 2, 0).numpy(), f"RGB frame {frame0}", None),
        (image1.permute(1, 2, 0).numpy(), f"RGB frame {frame1}", None),
        (sample["depth"][0, 0].numpy(), "Ground-truth inverse depth", "magma"),
        (flow_to_image(flow.permute(1, 2, 0).numpy().copy()), "Forward optical flow", None),
        (mask.numpy(), "Usable pixels: white", "gray"),
        (warped.permute(1, 2, 0).numpy(), "Frame 1 aligned to frame 0", None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (value, title, cmap) in zip(axes.flat, panels):
        if cmap == "gray":
            ax.imshow(value, cmap=cmap, vmin=0, vmax=1)
        else:
            ax.imshow(value, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "lesson2_preview.png", dpi=140)
    plt.close(fig)
    return {"rgb_mae_before_alignment": before, "rgb_mae_after_alignment": after,
            "usable_pixel_ratio": valid.float().mean().item(),
            "note": "RGB alignment diagnostic only; this is not a depth prediction score."}


def save_sequence_preview(sample, output_dir):
    images = (sample["image"] + 1) / 2
    depths = sample["depth"][:, 0].numpy()
    maximum = np.percentile(depths, 99)
    height, width = images.shape[-2:]
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    frames = []
    font = ImageFont.load_default(size=18)
    for t in range(len(images) - 1):
        flow = sample["f_flows"][t]
        grid = torch.stack((2 * (x + flow[0]) / (width - 1) - 1,
                            2 * (y + flow[1]) / (height - 1) - 1), dim=-1)
        warped = F.grid_sample(images[t + 1:t + 2], grid[None], align_corners=True)[0]
        depth_color = plt.get_cmap("magma")(np.clip(depths[t] / maximum, 0, 1))[..., :3]
        panels = [images[t].permute(1, 2, 0).numpy(), depth_color,
                  warped.permute(1, 2, 0).numpy()]
        frame = Image.new("RGB", (3 * width, height + 36), "white")
        draw = ImageDraw.Draw(frame)
        for column, (panel, title) in enumerate(zip(panels, (
                f"RGB clip frame {t}", "Ground-truth inverse depth", "Next RGB aligned to this frame"))):
            draw.text((column * width + 8, 8), title, fill="black", font=font)
            frame.paste(Image.fromarray((np.clip(panel, 0, 1) * 255).astype(np.uint8)),
                        (column * width, 36))
        frames.append(frame)
    frames[0].save(output_dir / "lesson2_sequence.gif", save_all=True,
                   append_images=frames[1:], duration=500, loop=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="datasets/TartanAir")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    np.random.seed(0)
    torch.manual_seed(0)
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    report = {"geometry_checks": check_geometry_and_sampler()}

    train = DepthVideoDataset("train", args.data_dir, seq_len=12)
    val = DepthVideoDataset("val", args.data_dir, seq_len=12)
    train_frames = {str(paths[0]) for _, clip in train.data_paths for paths in clip}
    val_frames = {str(paths[0]) for _, clip in val.data_paths for paths in clip}
    assert train_frames.isdisjoint(val_frames)
    batch = next(iter(DataLoader(train, batch_size=2, shuffle=False, num_workers=2)))
    shapes = {"image": (2, 12, 3, 384, 384), "depth": (2, 12, 1, 384, 384),
              "f_flows": (2, 11, 2, 384, 384), "b_flows": (2, 11, 2, 384, 384),
              "f_masks": (2, 11, 1, 384, 384), "b_masks": (2, 11, 1, 384, 384)}
    for key, tensor in batch.items():
        assert tuple(tensor.shape) == shapes[key]
        assert tensor.dtype == torch.float32 and torch.isfinite(tensor).all()
        if "masks" in key:
            assert ((tensor == 0) | (tensor == 1)).all()
    assert batch["image"].min() >= -1 and batch["image"].max() <= 1
    assert batch["depth"].min() >= 0

    # 验证单帧深度确实读成 1/D，并核对后向光流使用前一帧命名的文件。
    first_val = val[0]
    first_paths = val.data_paths[0][1]
    original_depth = np.load(first_paths[0][1])
    top, left = (original_depth.shape[0] - 384) // 2, (original_depth.shape[1] - 384) // 2
    np.testing.assert_allclose(first_val["depth"][0, 0].numpy(),
                               1 / original_depth[top:top + 384, left:left + 384])
    backward = np.load(first_paths[1][3]).astype(np.float32)
    np.testing.assert_allclose(first_val["b_flows"][0].permute(1, 2, 0).numpy(),
                               backward[top:top + 384, left:left + 384])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_batch = {key: value.to(device) for key, value in batch.items()}
    if device.type == "cuda":
        torch.cuda.synchronize()
    report.update({
        "pytorch": torch.__version__, "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "train_clips": len(train), "val_clips": len(val),
        "sequence_length": 12, "train_val_frames_disjoint": True,
        "batch_shapes": {key: list(value.shape) for key, value in gpu_batch.items()},
        "image_range": [batch["image"].min().item(), batch["image"].max().item()],
        "inverse_depth_range": [batch["depth"].min().item(), batch["depth"].max().item()],
        "preview": save_preview(val, output_dir),
    })
    save_sequence_preview(first_val, output_dir)
    if args.audit:
        report["file_audit"] = audit_files(Path(args.data_dir))
    (output_dir / "lesson2_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "file_audit"}, indent=2))
    if args.audit:
        print("AUDIT", {key: value for key, value in report["file_audit"].items() if key != "sequences"})
    print("PASS: outputs/lesson2_checks.json, lesson2_preview.png and lesson2_sequence.gif")


if __name__ == "__main__":
    main()
