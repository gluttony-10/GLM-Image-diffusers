# GLM-Image-diffusers
基于 GLM-Image 模型的图像生成工具，支持文本生成图像（T2I）和图像到图像转换（I2I）。

目前支持的功能有：
- **文生图 (T2I)**：输入描述图像的提示词生成高质量图像。
- **图生图 (I2I)**：基于底图和提示词进行图像变换。
- **智能缓存**：自动缓存提示词编码和图像令牌，相同提示词和分辨率的后续生成提速显著。
- **显存优化**：使用 mmgp 进行智能显存管理，支持在消费级显卡上运行。

一键包详见 [bilibili@十字鱼](https://space.bilibili.com/893892)

## 使用需求
1. 建议使用支持 CUDA 的显卡，显存建议 6GB 以上（支持 4GB 显存配合 mmgp 使用）。
2. 显卡最好支持 BF16（Compute Capability >= 8.0）。如果不支持将自动降级为 FP32 精度。
3. 图像宽度和高度必须是 32 的倍数（程序会自动调整）。

## 安装依赖
```bash
git clone https://github.com/gluttony-10/GLM-Image-diffusers
cd GLM-Image-diffusers
conda create -n glut python=3.12
conda activate glut
pip install git+https://github.com/huggingface/diffusers.git
pip install git+https://github.com/huggingface/transformers.git
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

## 下载模型
使用 ModelScope CLI 下载包含预量化文件的模型：
```bash
modelscope download --model Gluttony10/GLM-Image-diffusers --local_dir ./models/GLM-Image-diffusers
```

## 开始运行
```bash
python glut.py
```

## 命令行参数
- `--server_name`: IP地址，默认 127.0.0.1 (局域网访问设为 0.0.0.0)
- `--server_port`: 监听端口，默认 7891
- `--share`: 开启 gradio 共享链接
- `--compile`: 启用 compile 加速（需安装相应环境）
- `--res_vram`: 保留显存（MB），默认 1000

## 参考项目
https://github.com/zai-org/GLM-Image

https://github.com/deepbeepmeep/mmgp

https://github.com/lrzjason/Comfyui-DiffusersUtils
