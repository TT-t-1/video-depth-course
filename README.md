# video-depth-course

第二课任务：在 TartanAir 上使用 GMFlow 生成逐帧光流，完成 RGB-D 视频、双向光流和掩码的数据读取与预处理。依据课件第 88、90 页，本阶段的交付是数据层；ZipDepth 训练属于后续工作。

## 这次实验在做什么

输入是已经拍好的视频帧。我们为每两张相邻图片计算光流，找出同一个位置在下一张图片中的对应位置；之后把图片、真实深度和这些对应关系一起交给深度模型训练。

光流不是深度，也不是未来画面。光流的两个数 `u, v` 分别表示横向、纵向的像素位移。例如 `(320, 240)` 加上 `(-12.25, -16.21875)`，对应下一帧的 `(307.75, 223.78125)`。

Warping 用第 0 帧的位置作为输出位置，在第 1 帧的对应位置取颜色：

```text
aligned_frame1[y, x] = frame1[y + v(y,x), x + u(y,x)]
```

非整数坐标用双线性插值。这样对齐的是同一个物体，才能进一步比较它在两帧中的预测。真实深度会随相机或物体运动变化，不能直接要求两帧深度完全相等。

## 文件与运行环境

服务器项目目录：`/home/dancer/tu/video-depth-course`，现有 conda 环境：`vita`。已验证 PyTorch 1.13.0、RTX 3090、480×640 输入。具体依赖版本记录在 `environment-lesson2.json`。

```text
dataset.py                  完成的数据层和课程要求的增强、采样器
check_data.py               最小端到端检查和过程图生成
inspect_sample.py          单帧 RGB 与真实逆深度查看
visualize_flow.py           双帧光流与 warping 查看
external/gmflow/            课件提供的 GMFlow 源码
docs/lesson2_walkthrough.md 中文过程说明与验证结果
docs/lesson2_preview.png    实际数据的过程展示图
datasets/TartanAir/         数据集，Git 忽略
datasets/TartanAir_flow/    完整光流与掩码，Git 忽略
outputs/                   日志和检查输出，Git 忽略
```

GMFlow 使用原有权重 `external/gmflow/pretrained/gmflow_things-e9887eda.pth`。权重、压缩包和完整数据集均不进入仓库。第三方代码及其许可证保留在 `external/gmflow`。

## 1. 生成所有相邻帧的光流

在服务器运行：

```bash
cd /home/dancer/tu/video-depth-course
conda activate vita
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python external/gmflow/generate_flow.py \
  --inference_dir datasets/TartanAir \
  --output_path datasets/TartanAir_flow \
  --resume external/gmflow/pretrained/gmflow_things-e9887eda.pth \
  --strict_resume \
  --pred_bidir_flow \
  --fwd_bwd_consistency_check
```

- `cd`：进入项目目录，让后面的相对路径从这里开始。
- `conda activate vita`：使用已经验证的 Python 和 PyTorch 环境。
- `CUDA_VISIBLE_DEVICES=1`：这次推理只使用服务器编号为 1 的 GPU；程序内部会把它称为 `cuda:0`。
- `OMP_NUM_THREADS=4`、`MKL_NUM_THREADS=4`：限制 CPU 数学计算线程数，避免启动过多线程。
- `--inference_dir`：原始 RGB 所在的数据集目录。
- `--output_path`：填写输出目录。课件版本的 TartanAir 分支实际通过将输入路径中的 `TartanAir` 替换为 `TartanAir_flow` 决定输出位置，因此本命令保持二者一致。
- `--resume`：载入已有权重，无需从头训练 GMFlow。
- `--strict_resume`：严格检查权重与模型结构一致。
- `--pred_bidir_flow`：同时预测第 t 帧到 t+1 帧、t+1 帧到 t 帧的位移。
- `--fwd_bwd_consistency_check`：用双向位移的一致性找出不可靠位置。
- 行尾 `\`：下一行仍属于同一条命令。

如果一条视频有 N 帧，就有 N−1 对相邻帧。每对生成四个 `.npy` 文件：

| 子目录 | 数组形状 | 内容 |
|---|---|---|
| `f_flow` | `[480, 640, 2]` | t → t+1 的前向位移 |
| `b_flow` | `[480, 640, 2]` | t+1 → t 的后向位移 |
| `f_mask` | `[480, 640]` | 前向不可靠位置：1 不可靠，0 通过检查 |
| `b_mask` | `[480, 640]` | 后向不可靠位置：1 不可靠，0 通过检查 |

四个文件都用这一对中的第 t 帧命名。例如 `b_flow/000000_left.npy` 是第 1 帧回到第 0 帧的光流，不是第 0 帧回到不存在的第 −1 帧。

## 2. 数据层的处理顺序

1. 按帧编号匹配 `000000_left.png` 与 `000000_left_depth.npy`。只使用有对应深度的连续帧，遇到缺帧就断开片段。
2. 每 12 帧组成一个不重叠片段。每条轨迹的最后一个完整片段作为验证集，其余作为训练集，末尾不足 12 帧的部分不使用。
3. RGB 读成 float32，并由 `[0,255]` 转为 `[-1,1]`。
4. 深度读成 `1 / D`。这是逆深度；没有乘相机焦距和双目基线，不能把它直接当作具有像素单位的双目视差。非正值或非有限标签设为 0。
5. 读取 11 对前向和后向光流。RGB 与光流尺寸不同时，分别按宽、高比例调整 `u, v`。
6. 把原始不可靠掩码转换成可用掩码：`usable = 1 - unreliable`。
7. 训练时同步缩放到原尺寸的 0.8～0.85 倍、随机水平翻转、随机裁剪到 384×384；验证时只中心裁剪到 384×384。
8. RGB、逆深度和光流使用双线性缩放，掩码使用最近邻缩放。光流位移随分辨率缩放，逆深度数值不乘缩放比例。
9. 水平翻转时 `u` 变号，`v` 保持符号。裁剪后对应位置落在裁剪区域外的像素从可用掩码中排除。
10. 将数据转成 PyTorch 张量，并把时间、通道、高、宽明确分开。

原始数据共 24 条轨迹、5607 张 RGB，其中 abandonedfactory/P005 有 1007 张 RGB，但只有前 200 张有深度。其余 807 张仍计算光流，但不进入带深度标签的训练样本。共有 4800 对 RGB-D；12 帧方案产生 360 个训练片段、24 个验证片段，另有 192 个末尾 RGB-D 帧不足一段而未使用。

## 3. 读取一个批次

```python
from torch.utils.data import DataLoader
from dataset import DepthVideoDataset

dataset = DepthVideoDataset("train", data_dirs="datasets/TartanAir", seq_len=12)
loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)
batch = next(iter(loader))

for name, tensor in batch.items():
    print(name, tensor.shape)
```

`Dataset` 负责读取一个片段；`DataLoader` 负责把片段组合成批次。`next(iter(loader))` 取出第一个批次。批次中 `depth` 这一名称沿用参考代码，但内容是逆深度。

| 字段 | 一个样本 | 2 个样本的批次 |
|---|---|---|
| `image` | `[12, 3, 384, 384]` | `[2, 12, 3, 384, 384]` |
| `depth` | `[12, 1, 384, 384]` | `[2, 12, 1, 384, 384]` |
| `f_flows`, `b_flows` | `[11, 2, 384, 384]` | `[2, 11, 2, 384, 384]` |
| `f_masks`, `b_masks` | `[11, 1, 384, 384]` | `[2, 11, 1, 384, 384]` |

这里 12 是帧数，11 是相邻帧对数，3 是颜色通道，2 是光流的 `u,v`，1 是深度或掩码的单通道。所有输出都是 float32。

`DistributedSamplerNoEvenlyDivisible` 补齐了参考文件的采样器任务，可在多个进程间分配样本而不补齐重复样本。它不参与当前单 GPU 的光流生成或数据层验证。

## 4. 验证与展示

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python check_data.py --audit
```

`--audit` 要在完整光流生成结束后使用。它检查全部轨迹输出的文件名覆盖率，并检查每条轨迹首尾四种输出的形状、类型、有限数值和二值掩码；同时检查数据增强的几何变换、后向文件索引、逆深度数值、训练与验证帧不重叠，以及一个批次的实际 GPU 传输。

输出：`outputs/lesson2_checks.json`、`outputs/lesson2_preview.png`。过程图的上排是相邻 RGB 和真实逆深度，下排是前向光流、可用掩码和 warping 结果。

![实际样本的数据处理过程](docs/lesson2_preview.png)

当前展示样例的有效区域 RGB 平均绝对差，运动对齐前为 0.03685，对齐后为 0.01197。这只是检查 RGB 对齐效果的数值，不代表深度模型精度，也没有使用深度预测模型。

## 来源

- 本课程第 2 课 PDF 第 88、90 页和老师提供的 `dataset.py`。
- GMFlow：<https://github.com/haofeixu/gmflow>，许可证见 `external/gmflow/LICENSE`。
- TartanAir：<https://theairlab.org/tartanair-dataset/>。

本次补全数据层、检查脚本和说明使用了 AI 辅助；课程数据和 GMFlow 权重由课件下载资源提供。老师参考文件的原始副本保留在 `docs/dataset_template.txt`，便于对照学习。
