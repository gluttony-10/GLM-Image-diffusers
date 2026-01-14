# GLM-Image-diffusers

基于 GLM-Image 模型的图像生成工具，支持文本生成图像（T2I）和图像到图像转换（I2I）。

本工具使用 [mmgp](https://github.com/deepbeepmeep/mmgp) 进行显存管理，可以在消费级 GPU 上运行大型模型。

## 安装说明

### 1. 安装依赖

首先需要从源码安装 diffusers 和 transformers：

```bash
pip install git+https://github.com/huggingface/diffusers.git
pip install git+https://github.com/huggingface/transformers.git
```

然后安装其他依赖：

```bash
pip install -r requirements.txt
```

或者使用一键安装：

```bash
pip install git+https://github.com/huggingface/diffusers.git git+https://github.com/huggingface/transformers.git torch torchvision torchaudio gradio numpy accelerate Pillow huggingface_hub
```

### 2. 下载模型

模型需要从 ModelScope 下载。有两种方式：

#### 方式一：使用下载脚本（推荐）

```bash
python download_model.py
```

#### 方式二：使用 Python 代码

```python
from modelscope import snapshot_download

snapshot_download(
    "ZhipuAI/GLM-Image",
    cache_dir="models/GLM-Image-diffusers",
    local_dir="models/GLM-Image-diffusers",
)
```

模型将下载到 `models/GLM-Image-diffusers` 目录。

### 4. 运行程序

程序使用预量化模型以加快加载速度和降低显存占用。请确保以下文件存在：

- `models/GLM-Image-diffusers/vision_language_encoder-mmgp.safetensors` - 视觉语言编码器的量化模型
- `models/GLM-Image-diffusers/transformer-mmgp.safetensors` - Transformer 的量化模型

如果这些文件不存在，程序会自动加载并量化模型（需要较长时间）。

### 5. 运行程序

直接运行 `01运行程序.bat` 或在命令行中运行：

```bash
python glut.py
```

### 5. 命令行参数

- `--server_name`: IP地址，默认 127.0.0.1，局域网访问改为 0.0.0.0
- `--server_port`: 使用端口，默认 7892
- `--share`: 是否启用 gradio 共享链接
- `--compile`: 是否启用 compile 加速（需要 Triton，Linux/WSL 支持，Windows 需要额外安装）

示例：

```bash
python glut.py --server_name 0.0.0.0 --server_port 7892 --share --compile
```

## 功能说明

### 文本生成图像 (T2I)

输入文本提示词，生成对应的图像。

- **提示词**: 描述想要生成的图像内容
- **负面提示词**: 描述不希望在图像中出现的内容
- **宽度/高度**: 生成图像的尺寸（必须是32的倍数）
- **推理步数**: 生成过程的迭代次数（1-100）
- **引导强度**: 控制提示词的影响程度（0.1-10.0）
- **种子**: 随机种子，-1 表示随机
- **生成数量**: 一次生成多张图像

### 图像到图像 (I2I)

基于输入图像和提示词，生成修改后的图像。

- **输入图像**: 上传要修改的图像
- **提示词**: 描述想要修改的内容
- 其他参数与 T2I 相同

### 便捷功能

- **交换宽高**: 快速交换宽度和高度值
- **1.5分辨率**: 将当前分辨率放大 1.5 倍

## 配置说明

程序会在启动时读取 `config.json` 文件（如果存在），可以配置默认参数：

```json
{
    "NUM_INFERENCE_STEPS": "50",
    "GUIDANCE_SCALE": "1.5",
    "WIDTH": "1024",
    "HEIGHT": "1024",
    "RES_VRAM": "1000"
}
```

### 配置参数说明

- `NUM_INFERENCE_STEPS`: 默认推理步数
- `GUIDANCE_SCALE`: 默认引导强度
- `WIDTH/HEIGHT`: 默认生成的图像尺寸
- `RES_VRAM`: 保留的显存（MB），用于处理图像数据，默认 1000MB

### mmgp 显存管理

程序使用 mmgp 进行智能显存管理：

- **量化模型**: vision_language_encoder 和 transformer 会被量化以减少显存占用
- **自动换页**: 模型组件会在 CPU 和 GPU 之间自动切换
- **内存钉住**: 高内存系统可以钉住更多组件到内存
- **目标显存**: 根据 RES_VRAM 配置自动计算可用显存预算

启动时会显示：`限制目标显存：XXXXMB`

## 输出说明

生成的图像会保存在 `outputs/` 目录下，文件名包含时间戳和种子数。图像中会嵌入元数据信息，包括提示词、参数等。

推理完成后会显示：
- 生成的图片数量
- 总耗时（秒）

## 注意事项

1. **模型下载**: 首次使用前需要手动下载模型（见"下载模型"部分），模型较大，需要较长时间和足够的磁盘空间
2. 建议使用支持 CUDA 的 GPU，显存建议 6GB 以上（使用 mmgp 后）
3. 宽度和高度必须是 32 的倍数，程序会自动调整
4. 生成过程中可以点击"停止"按钮中断生成
5. 使用 mmgp 后，模型会自动在 CPU 和 GPU 之间切换，降低显存占用
6. 如果遇到显存不足，可以：
   - 降低 `RES_VRAM` 值
   - 减少生成图像的尺寸或批量数量

## 模型信息

- **ModelScope 模型**: [ZhipuAI/GLM-Image](https://modelscope.cn/models/ZhipuAI/GLM-Image)
- **模型路径**: `models/GLM-Image-diffusers`（本地目录）
- **量化模型**: `vision_language_encoder-mmgp.safetensors` 和 `transformer-mmgp.safetensors`
- **支持的数据类型**: bfloat16 (需要 CUDA Compute Capability >= 8.0)
- **显存管理**: 使用 [mmgp](https://github.com/deepbeepmeep/mmgp) 进行智能显存管理
