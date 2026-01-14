import warnings
warnings.filterwarnings("ignore")
import gc
import os
from PIL import Image, PngImagePlugin
import json
import time
import torch
import numpy as np
import random
import gradio as gr
import socket
import argparse
import datetime
import psutil
from diffusers.pipelines.glm_image import GlmImagePipeline
from diffusers.models import GlmImageTransformer2DModel
from diffusers.utils import load_image
from transformers import GlmImageForConditionalGeneration

parser = argparse.ArgumentParser() 
parser.add_argument("--server_name", type=str, default="127.0.0.1", help="IP地址，局域网访问改为0.0.0.0")
parser.add_argument("--server_port", type=int, default=7891, help="使用端口")
parser.add_argument("--share", action="store_true", help="是否启用gradio共享")
parser.add_argument("--compile", action="store_true", help="是否启用compile加速")
args = parser.parse_args()

print(" 启动中，请耐心等待 bilibili@十字鱼 https://space.bilibili.com/893892")
print(f'\033[32mPytorch版本：{torch.__version__}\033[0m')
if torch.cuda.is_available():
    device = "cuda" 
    print(f'\033[32m显卡型号：{torch.cuda.get_device_name()}\033[0m')
    total_vram_in_gb = torch.cuda.get_device_properties(0).total_memory / 1073741824
    print(f'\033[32m显存大小：{total_vram_in_gb:.2f}GB\033[0m')
    mem = psutil.virtual_memory()
    print(f'\033[32m内存大小：{mem.total/1073741824:.2f}GB\033[0m')
    if torch.cuda.get_device_capability()[0] >= 8:
        print(f'\033[32m支持BF16\033[0m')
        dtype = torch.bfloat16
    else:
        print(f'\033[32m不支持BF16，使用FP32\033[0m')
        dtype = torch.float32
else:
    print(f'\033[32mCUDA不可用，请检查\033[0m')
    device = "cpu"
    dtype = torch.float32
    mem = psutil.virtual_memory()
    print(f'\033[32m内存大小：{mem.total/1073741824:.2f}GB\033[0m')

# 初始化
pipe = None
mmgp = None
stop_generation = False
model_id = "models/GLM-Image-diffusers"

# 确保输出文件夹存在
os.makedirs("outputs", exist_ok=True)

# 读取设置
CONFIG_FILE = "config.json"
config = {}
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = json.load(f)

# 默认设置
num_inference_steps_default = int(config.get("NUM_INFERENCE_STEPS", "50"))
guidance_scale_default = float(config.get("GUIDANCE_SCALE", "1.5"))
width_default = int(config.get("WIDTH", "1024"))
height_default = int(config.get("HEIGHT", "1024"))
res_vram = float(config.get("RES_VRAM", "1000"))


def load_pipeline():
    """加载 GLM-Image 管道"""
    global pipe, mmgp
    if pipe is None:
        print("正在加载 GLM-Image 模型...")
        try:
            # 量化模型路径
            vision_lang_encoder_path = "models/GLM-Image-diffusers/vision_language_encoder-mmgp.safetensors"
            transformer_path = "models/GLM-Image-diffusers/transformer-mmgp.safetensors"
            
            # 先加载基础模型到 CPU
            pipe = GlmImagePipeline.from_pretrained(
                model_id, 
                vision_language_encoder = None,
                transformer = None,
                torch_dtype=dtype,
            ).to("cpu")
            
            # 加载完成后再导入 mmgp
            from mmgp import offload
            # 使用 mmgp 快速加载量化模型
            if hasattr(pipe, 'vision_language_encoder'):
                pipe.vision_language_encoder = offload.fast_load_transformers_model(
                    vision_lang_encoder_path,
                    modelClass=GlmImageForConditionalGeneration,
                    forcedConfigPath=f"{model_id}/vision_language_encoder/config.json",
                )
            pipe.transformer = offload.fast_load_transformers_model(
                transformer_path,
                modelClass=GlmImageTransformer2DModel,
                forcedConfigPath=f"{model_id}/transformer/config.json",
            )
            
            # 计算显存预算
            if device == "cuda":
                free_memory, _ = torch.cuda.mem_get_info(0)
                budgets = int(free_memory / 1048576 - res_vram)  # 转换为 MB
            else:
                budgets = 0
            
            # 配置 mmgp（不重新量化）
            mmgp = offload.all(
                pipe, 
                pinnedMemory=["vision_language_encoder", "text_encoder", "transformer"] if mem.total/1073741824 > 30 else ["transformer"],
                budgets={'*': budgets}, 
                extraModelsToQuantize=["vision_language_encoder"], 
                compile=True if args.compile else False,
            )
            
            print("✅ 量化模型加载完成！mmgp 配置完成，限制目标显存：" + str(budgets) + "MB")
                
                
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    return pipe


# 解决冲突端口
def find_port(port: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        if s.connect_ex(("localhost", port)) == 0:
            print(f"端口 {port} 已被占用，正在寻找可用端口...")
            return find_port(port=port + 1)
        else:
            return port


def stop_generate():
    """停止生成"""
    global stop_generation
    stop_generation = True
    return "✅ 生成已停止"


def generate_t2i(prompt, negative_prompt, width, height, num_inference_steps, 
                 guidance_scale, seed_param, batch_images):
    """文本生成图像"""
    global stop_generation, pipe
    
    if not prompt or not prompt.strip():
        yield None, "❌ 请输入提示词"
        return
    
    if pipe is None:
        pipe = load_pipeline()
        if pipe is None:
            yield None, "❌ 模型加载失败，请检查"
            return
    
    stop_generation = False
    results = []
    start_time = time.time()  # 记录开始时间
    
    # 处理种子
    if seed_param < 0:
        seed = random.randint(0, np.iinfo(np.int32).max)
    else:
        seed = seed_param
    
    # 确保宽高是32的倍数
    width = (width // 32) * 32
    height = (height // 32) * 32
    
    try:
        for i in range(batch_images):
            if stop_generation:
                stop_generation = False
                yield results if results else None, f"✅ 生成已中止，最后种子数{seed+i-1}"
                break
            
            current_seed = seed + i
            generator = torch.Generator(device=device).manual_seed(current_seed)
            
            # 生成图像
            output = pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            
            image = output.images[0]
            
            # 保存图像
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"outputs/t2i_{timestamp}_{current_seed}.png"
            
            # 添加元数据
            pnginfo = PngImagePlugin.PngInfo()
            pnginfo.add_text("mode", "t2i\n")
            pnginfo.add_text("prompt", f"{prompt}\n")
            pnginfo.add_text("negative_prompt", f"{negative_prompt}\n")
            pnginfo.add_text("width", f"{width}\n")
            pnginfo.add_text("height", f"{height}\n")
            pnginfo.add_text("num_inference_steps", f"{num_inference_steps}\n")
            pnginfo.add_text("guidance_scale", f"{guidance_scale}\n")
            pnginfo.add_text("seed", f"{current_seed}\n")
            
            image.save(filename, pnginfo=pnginfo)
            results.append(image)
            
            yield results, f"✅ 种子数{current_seed}，保存地址: {filename}"
            
            # mmgp 会自动管理显存，这里只需要清理 Python 对象
            gc.collect()
        
        # 计算总时间
        end_time = time.time()
        total_time = end_time - start_time
        yield results, f"✅ 推理完成，共生成{len(results)}张图片，总耗时{total_time:.2f}秒"
    
    except Exception as e:
        yield results if results else None, f"❌ 生成失败: {str(e)}"


def generate_i2i(image, prompt, negative_prompt, width, height, num_inference_steps,
                 guidance_scale, seed_param, batch_images):
    """图像到图像生成"""
    global stop_generation, pipe
    
    if image is None:
        yield None, "❌ 请上传输入图像"
        return
    
    if not prompt or not prompt.strip():
        yield None, "❌ 请输入提示词"
        return
    
    if pipe is None:
        pipe = load_pipeline()
        if pipe is None:
            yield None, "❌ 模型加载失败，请检查"
            return
    
    stop_generation = False
    results = []
    start_time = time.time()  # 记录开始时间
    
    # 处理输入图像
    if isinstance(image, dict):
        image = image.get("background", image)
    image = load_image(image).convert("RGB")
    
    # 处理种子
    if seed_param < 0:
        seed = random.randint(0, np.iinfo(np.int32).max)
    else:
        seed = seed_param
    
    # 确保宽高是32的倍数
    width = (width // 32) * 32
    height = (height // 32) * 32
    
    try:
        for i in range(batch_images):
            if stop_generation:
                stop_generation = False
                yield results if results else None, f"✅ 生成已中止，最后种子数{seed+i-1}"
                break
            
            current_seed = seed + i
            generator = torch.Generator(device=device).manual_seed(current_seed)
            
            # 生成图像
            output = pipe(
                prompt=prompt,
                image=[image],  # 可以输入多个图像，如 [image, image1]
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            
            generated_image = output.images[0]
            
            # 保存图像
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"outputs/i2i_{timestamp}_{current_seed}.png"
            
            # 添加元数据
            pnginfo = PngImagePlugin.PngInfo()
            pnginfo.add_text("mode", "i2i\n")
            pnginfo.add_text("prompt", f"{prompt}\n")
            pnginfo.add_text("negative_prompt", f"{negative_prompt}\n")
            pnginfo.add_text("width", f"{width}\n")
            pnginfo.add_text("height", f"{height}\n")
            pnginfo.add_text("num_inference_steps", f"{num_inference_steps}\n")
            pnginfo.add_text("guidance_scale", f"{guidance_scale}\n")
            pnginfo.add_text("seed", f"{current_seed}\n")
            
            generated_image.save(filename, pnginfo=pnginfo)
            results.append(generated_image)
            
            yield results, f"✅ 种子数{current_seed}，保存地址: {filename}"
            
            # mmgp 会自动管理显存，这里只需要清理 Python 对象
            gc.collect()
        
        # 计算总时间
        end_time = time.time()
        total_time = end_time - start_time
        yield results, f"✅ 推理完成，共生成{len(results)}张图片，总耗时{total_time:.2f}秒"
    
    except Exception as e:
        yield results if results else None, f"❌ 生成失败: {str(e)}"


def exchange_width_height(width, height):
    """交换宽高"""
    return height, width, "✅ 宽高交换完毕"


def scale_resolution_1_5(width, height):
    """
    将宽度和高度都放大1.5倍，并按照32的倍数向下取整
    """
    new_width = int(width * 1.5) // 32 * 32
    new_height = int(height * 1.5) // 32 * 32
    return new_width, new_height, "✅ 分辨率已调整为1.5倍"


# 创建 Gradio 界面
css = """
.gradio-container {
    font-family: 'IBM Plex Sans', sans-serif;
}
"""

with gr.Blocks() as demo:
    gr.Markdown("""
            <div>
                <h2 style="font-size: 30px;text-align: center;">GLM-Image</h2>
            </div>
            <div style="text-align: center;">
                十字鱼
                <a href="https://space.bilibili.com/893892">🌐bilibili</a> 
                |GLM-Image-diffusers
                <a href="https://github.com/gluttony-10/GLM-Image-diffusers">🌐github</a> 
            </div>
            <div style="text-align: center; font-weight: bold; color: red;">
                ⚠️ 该演示仅供学术研究和体验使用。
            </div>
            """)
    
    with gr.Tabs():
        # Text to Image 标签页
        with gr.Tab("文生图 "):
            with gr.Row():
                with gr.Column(scale=1):
                    prompt_t2i = gr.Textbox(
                        label="提示词",
                        placeholder="输入描述图像的文本提示词...",
                        lines=1,
                        value="A beautifully designed modern food magazine style dessert recipe illustration, themed around a raspberry mousse cake."
                    )
                    negative_prompt_t2i = gr.Textbox(
                        label="负面提示词",
                        placeholder="输入不希望在图像中出现的内容...",
                        lines=1
                    )
                    
                    with gr.Row():
                        width_t2i = gr.Slider(
                            label="宽度",
                            minimum=32,
                            maximum=2048,
                            value=width_default,
                            step=32
                        )
                        height_t2i = gr.Slider(
                            label="高度",
                            minimum=32,
                            maximum=2048,
                            value=height_default,
                            step=32
                        )
                    
                    with gr.Row():
                        exchange_button_t2i = gr.Button("🔄 交换宽高", scale=1)
                    
                    num_inference_steps_t2i = gr.Slider(
                        label="推理步数",
                        minimum=1,
                        maximum=100,
                        value=num_inference_steps_default,
                        step=1
                    )
                    guidance_scale_t2i = gr.Slider(
                        label="引导强度",
                        minimum=0.1,
                        maximum=10.0,
                        value=guidance_scale_default,
                        step=0.1
                    )
                    
                    with gr.Row():
                        seed_t2i = gr.Number(
                            label="种子 (-1为随机)",
                            value=-1,
                            precision=0
                        )
                        batch_images_t2i = gr.Slider(
                            label="生成数量",
                            minimum=1,
                            maximum=10,
                            value=1,
                            step=1
                        )
                    
                    with gr.Row():
                        run_btn_t2i = gr.Button("🎨 生成图像", variant="primary", scale=2)
                        stop_button_t2i = gr.Button("⏹️ 停止", scale=1)
                
                with gr.Column(scale=1):
                    result_t2i = gr.Gallery(
                        label="生成结果",
                        show_label=True,
                        elem_id="gallery_t2i",
                        columns=2,
                        rows=2,
                        height="auto"
                    )
                    info_t2i = gr.Textbox(
                        label="信息",
                        lines=3,
                        interactive=False
                    )
        
        # Image to Image 标签页
        with gr.Tab("图生图"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_i2i = gr.Image(
                        label="输入图像",
                        type="pil",
                        sources=["upload", "clipboard"]
                    )
                    
                    prompt_i2i = gr.Textbox(
                        label="提示词",
                        placeholder="输入描述想要修改的内容...",
                        lines=1,
                        value="Replace the background of the snow forest with an underground station featuring an automatic escalator."
                    )
                    negative_prompt_i2i = gr.Textbox(
                        label="负面提示词",
                        placeholder="输入不希望在图像中出现的内容...",
                        lines=1
                    )
                    
                    with gr.Row():
                        width_i2i = gr.Slider(
                            label="宽度",
                            minimum=32,
                            maximum=2048,
                            value=width_default,
                            step=32
                        )
                        height_i2i = gr.Slider(
                            label="高度",
                            minimum=32,
                            maximum=2048,
                            value=height_default,
                            step=32
                        )
                    
                    with gr.Row():
                        exchange_button_i2i = gr.Button("🔄 交换宽高", scale=1)
                        scale_1_5_button_i2i = gr.Button("📊 1.5分辨率", scale=1)
                    
                    num_inference_steps_i2i = gr.Slider(
                        label="推理步数",
                        minimum=1,
                        maximum=100,
                        value=num_inference_steps_default,
                        step=1
                    )
                    guidance_scale_i2i = gr.Slider(
                        label="引导强度",
                        minimum=0.1,
                        maximum=10.0,
                        value=guidance_scale_default,
                        step=0.1
                    )
                    
                    with gr.Row():
                        seed_i2i = gr.Number(
                            label="种子 (-1为随机)",
                            value=-1,
                            precision=0
                        )
                        batch_images_i2i = gr.Slider(
                            label="生成数量",
                            minimum=1,
                            maximum=10,
                            value=1,
                            step=1
                        )
                    
                    with gr.Row():
                        run_btn_i2i = gr.Button("🎨 生成图像", variant="primary", scale=2)
                        stop_button_i2i = gr.Button("⏹️ 停止", scale=1)
                
                with gr.Column(scale=1):
                    result_i2i = gr.Gallery(
                        label="生成结果",
                        show_label=True,
                        elem_id="gallery_i2i",
                        columns=2,
                        rows=2,
                        height="auto"
                    )
                    info_i2i = gr.Textbox(
                        label="信息",
                        lines=3,
                        interactive=False
                    )
    
    # 绑定事件
    # T2I 事件
    gr.on(
        triggers=[run_btn_t2i.click, prompt_t2i.submit, negative_prompt_t2i.submit],
        fn=generate_t2i,
        inputs=[prompt_t2i, negative_prompt_t2i, width_t2i, height_t2i, 
                num_inference_steps_t2i, guidance_scale_t2i, seed_t2i, batch_images_t2i],
        outputs=[result_t2i, info_t2i]
    )
    
    stop_button_t2i.click(
        fn=stop_generate,
        inputs=[],
        outputs=[info_t2i]
    )
    
    exchange_button_t2i.click(
        fn=exchange_width_height,
        inputs=[width_t2i, height_t2i],
        outputs=[width_t2i, height_t2i, info_t2i]
    )
    
    # I2I 事件
    gr.on(
        triggers=[run_btn_i2i.click, prompt_i2i.submit, negative_prompt_i2i.submit],
        fn=generate_i2i,
        inputs=[image_i2i, prompt_i2i, negative_prompt_i2i, width_i2i, height_i2i,
                num_inference_steps_i2i, guidance_scale_i2i, seed_i2i, batch_images_i2i],
        outputs=[result_i2i, info_i2i]
    )
    
    stop_button_i2i.click(
        fn=stop_generate,
        inputs=[],
        outputs=[info_i2i]
    )
    
    exchange_button_i2i.click(
        fn=exchange_width_height,
        inputs=[width_i2i, height_i2i],
        outputs=[width_i2i, height_i2i, info_i2i]
    )
    
    scale_1_5_button_i2i.click(
        fn=scale_resolution_1_5,
        inputs=[width_i2i, height_i2i],
        outputs=[width_i2i, height_i2i, info_i2i]
    )
    
    adjust_button_i2i.click(
        fn=adjust_width_height,
        inputs=[image_i2i],
        outputs=[width_i2i, height_i2i, info_i2i]
    )
    
    # 上传图片时自动调整宽高
    image_i2i.upload(
        fn=adjust_width_height, 
        inputs=[image_i2i], 
        outputs=[width_i2i, height_i2i, info_i2i]
    )


if __name__ == "__main__":
    # 预加载模型
    print("正在预加载模型...")
    load_pipeline()
    
    demo.launch(
        server_name=args.server_name,
        server_port=find_port(args.server_port),
        share=args.share,
        inbrowser=True,
        css=css, 
        theme=gr.themes.Soft(font=[gr.themes.GoogleFont("IBM Plex Sans")]),
    )
