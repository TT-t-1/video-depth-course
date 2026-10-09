"""第二课：读取 TartanAir 连续帧及 GMFlow 输出，完成同步数据增强。"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
import torch.distributed as dist


class RandomScale:
    def __init__(self, scale_limit, last_ch):
        self.scale_limit = scale_limit
        self.last_ch = last_ch

    def __call__(self, x):
        height, width = x.shape[:2]
        scale = np.random.uniform(*self.scale_limit)
        new_height, new_width = round(height * scale), round(width * scale)
        resized = cv2.resize(x, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # 布局：RGB、逆深度、两组掩码、两组光流；掩码只用最近邻插值。
        mask_start = x.shape[-1] - self.last_ch - self.last_ch // 2
        mask_end = x.shape[-1] - self.last_ch
        resized[..., mask_start:mask_end] = cv2.resize(
            x[..., mask_start:mask_end], (new_width, new_height),
            interpolation=cv2.INTER_NEAREST,
        )
        flows = resized[..., -self.last_ch:]
        flows[..., 0::2] *= new_width / width
        flows[..., 1::2] *= new_height / height
        # 这里只改变图像分辨率，相机到物体的距离没有变化，逆深度不乘 scale。
        return resized


class RandomHorizontalFlip:
    def __init__(self, last_ch):
        self.last_ch = last_ch

    def __call__(self, x):
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=1).copy()
            x[..., -self.last_ch:][..., 0::2] *= -1
        return x


class DepthVideoDataset(Dataset):
    def __init__(self, mode, data_dirs="datasets/TartanAir", crop_size=384, seq_len=12):
        if mode not in ("train", "val"):
            raise ValueError("mode must be 'train' or 'val'")
        if seq_len < 2:
            raise ValueError("seq_len must be at least 2")
        if isinstance(data_dirs, (str, Path)):
            data_dirs = [data_dirs]
        self.mode = mode
        self.crop_size = crop_size
        self.seq_len = seq_len
        self.data_paths = []
        self.sequence_stats = []

        for data_dir in data_dirs:
            data_root = Path(data_dir)
            flow_root = data_root.with_name(data_root.name + "_flow")
            for image_dir in sorted(data_root.rglob("image_left")):
                sequence_dir = image_dir.parent
                image_paths = sorted(image_dir.glob("*.png"))
                matched = []
                for image_path in image_paths:
                    frame_id = image_path.stem.split("_")[0]
                    depth_path = sequence_dir / "depth_left" / f"{frame_id}_left_depth.npy"
                    if depth_path.is_file():
                        matched.append((int(frame_id), image_path, depth_path))

                # 按帧编号配对；缺少标签时断开序列，不跨越缺帧生成训练样本。
                runs = []
                for frame in matched:
                    if not runs or frame[0] != runs[-1][-1][0] + 1:
                        runs.append([])
                    runs[-1].append(frame)
                clips = []
                for run in runs:
                    for start in range(0, len(run) - seq_len + 1, seq_len):
                        clips.append(run[start:start + seq_len])

                # 每条轨迹最后一个完整片段留作验证，其余片段用于训练。
                selected = clips[:-1] if mode == "train" else clips[-1:]
                sequence_flow = flow_root / sequence_dir.relative_to(data_root)
                for clip in selected:
                    paths = []
                    for j, (_, image_path, depth_path) in enumerate(clip):
                        forward_name = image_path.with_suffix(".npy").name
                        backward_name = clip[j - 1][1].with_suffix(".npy").name
                        paths.append([
                            image_path,
                            depth_path,
                            sequence_flow / "f_flow" / forward_name if j < seq_len - 1 else None,
                            sequence_flow / "b_flow" / backward_name if j > 0 else None,
                            sequence_flow / "f_mask" / forward_name if j < seq_len - 1 else None,
                            sequence_flow / "b_mask" / backward_name if j > 0 else None,
                        ])
                    self.data_paths.append(("TartanAir", paths))
                self.sequence_stats.append({
                    "sequence": str(sequence_dir.relative_to(data_root)),
                    "rgb_frames": len(image_paths),
                    "matched_frames": len(matched),
                    "missing_depth": len(image_paths) - len(matched),
                    "clips": len(selected),
                })

        self.scale = RandomScale((0.8, 0.85), last_ch=4 * (seq_len - 1))
        self.flip = RandomHorizontalFlip(last_ch=4 * (seq_len - 1))

    def __getitem__(self, item):
        _, paths = self.data_paths[item]
        images, depths = [], []
        f_flows, b_flows, f_masks, b_masks = [], [], [], []
        for image_path, depth_path, f_flow_path, b_flow_path, f_mask_path, b_mask_path in paths:
            image = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32)
            image = image / 127.5 - 1.0
            metric_depth = np.load(depth_path).astype(np.float32)
            valid_depth = np.isfinite(metric_depth) & (metric_depth > 0)
            inverse_depth = np.divide(
                1.0, metric_depth, out=np.zeros_like(metric_depth), where=valid_depth,
            )[..., None]
            images.append(image)
            depths.append(inverse_depth)

            for flow_path, mask_path, flows, masks in [
                (f_flow_path, f_mask_path, f_flows, f_masks),
                (b_flow_path, b_mask_path, b_flows, b_masks),
            ]:
                if flow_path is None:
                    continue
                flow = np.load(flow_path).astype(np.float32)
                mask = np.load(mask_path).astype(np.float32)
                height, width = image.shape[:2]
                flow_height, flow_width = flow.shape[:2]
                if (flow_height, flow_width) != (height, width):
                    flow = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
                    flow[..., 0] *= width / flow_width
                    flow[..., 1] *= height / flow_height
                    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                flows.append(flow)
                # GMFlow 的 1 表示不可靠；数据层的 1 表示可以参与计算。
                masks.append((1.0 - mask)[..., None])

        arrays = [np.concatenate(values, axis=-1) for values in
                  (images, depths, f_masks, b_masks, f_flows, b_flows)]
        channels = [array.shape[-1] for array in arrays]
        packed = np.concatenate(arrays, axis=-1)
        if self.mode == "train":
            packed = self.scale(packed)
            packed = self.flip(packed)

        height, width = packed.shape[:2]
        if self.crop_size is not None:
            size = self.crop_size
            if size > min(height, width):
                raise ValueError(f"crop_size={size} exceeds image size {(height, width)}")
            if self.mode == "train":
                top = np.random.randint(height - size + 1)
                left = np.random.randint(width - size + 1)
            else:
                top, left = (height - size) // 2, (width - size) // 2
            packed = packed[top:top + size, left:left + size].copy()

        # 按拼接时的通道数切回六组数据，再将通道展开为时间维度。
        arrays = np.split(packed, np.cumsum(channels)[:-1], axis=-1)
        keys = ("image", "depth", "f_masks", "b_masks", "f_flows", "b_flows")
        sample = {}
        for key, array in zip(keys, arrays):
            length = self.seq_len if key in ("image", "depth") else self.seq_len - 1
            frames = np.stack(np.split(array, length, axis=-1))
            sample[key] = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
        height, width = sample["image"].shape[-2:]
        y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        for direction in ("f", "b"):
            flow = sample[f"{direction}_flows"]
            target_x, target_y = x + flow[:, 0], y + flow[:, 1]
            inside = ((target_x >= 0) & (target_x <= width - 1)
                      & (target_y >= 0) & (target_y <= height - 1))
            # 裁剪后，对应位置落在裁剪区域外的像素也不能用于运动对齐损失。
            sample[f"{direction}_masks"] *= inside.unsqueeze(1)
        return sample

    def __len__(self):
        return len(self.data_paths)


class DistributedSamplerNoEvenlyDivisible(Sampler):
    """把索引分给各个进程，不为凑整重复添加样本。"""

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        self.dataset = dataset
        self.num_replicas = dist.get_world_size() if num_replicas is None else num_replicas
        self.rank = dist.get_rank() if rank is None else rank
        self.shuffle = shuffle
        self.epoch = 0
        self.num_samples = len(range(self.rank, len(dataset), self.num_replicas))
        self.total_size = len(dataset)

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        else:
            indices = list(range(len(self.dataset)))
        return iter(indices[self.rank::self.num_replicas])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
