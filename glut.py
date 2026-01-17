import warnings
warnings.filterwarnings("ignore")
import os
from PIL import Image, PngImagePlugin
import time
import torch
import numpy as np
import random
import gradio as gr
import socket
import argparse
import datetime
import psutil
from glm_image import GlmImagePipeline
from diffusers.models import GlmImageTransformer2DModel
from diffusers.utils import load_image
from transformers import GlmImageForConditionalGeneration, ByT5Tokenizer

parser = argparse.ArgumentParser() 
parser.add_argument("--server_name", type=str, default="127.0.0.1", help="IP地址，局域网访问改为0.0.0.0")
parser.add_argument("--server_port", type=int, default=7891, help="使用端口")
parser.add_argument("--share", action="store_true", help="是否启用gradio共享")
parser.add_argument("--compile", action="store_true", help="是否启用compile加速")
parser.add_argument("--res_vram", type=int, default=1000, help="保留显存(MB)，默认1000")
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

# vision_language_encoder 缓存（缓存 prior_tokens 和 prompt_embeds）
prior_cache = {
    "key": None,  # (prompt, height, width, image_hash)
    "prompt": None,  # 用于 prompt_embeds 缓存
    "prior_token_ids": None,
    "prior_image_token_ids": None,
    "prompt_embeds": None,
}

# 启用 CUDA 加速优化
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # 自动寻找最优卷积算法
    torch.backends.cuda.matmul.allow_tf32 = True  # 允许 TF32 矩阵乘法
    torch.backends.cudnn.allow_tf32 = True  # 允许 TF32 加速


def get_image_hash(img):
    """获取图像的简单哈希值用于缓存"""
    if img is None:
        return None
    # 使用图像尺寸和部分像素数据生成简单哈希
    return hash((img.size, img.mode, img.tobytes()[:1000]))


def get_cached_prompt_embeds(prompt):
    """获取缓存的 prompt_embeds"""
    global prior_cache
    
    # 检查缓存（只基于 prompt）
    if prior_cache["prompt"] == prompt and prior_cache["prompt_embeds"] is not None:
        print("📦 使用缓存的 prompt_embeds")
        return prior_cache["prompt_embeds"]
    
    # 编码新的提示词
    print("🔄 编码提示词...")
    with torch.inference_mode():
        prompt_embeds, _ = pipe.encode_prompt(
            prompt=prompt,
            do_classifier_free_guidance=False,
        )
    
    # 更新缓存
    prior_cache["prompt"] = prompt
    prior_cache["prompt_embeds"] = prompt_embeds
    
    return prompt_embeds


def get_cached_prior_tokens(prompt, height, width, image=None):
    """获取缓存的 prior tokens，如果没有则生成"""
    global prior_cache
    
    # 生成缓存键
    image_hash = get_image_hash(image)
    cache_key = (prompt, height, width, image_hash)
    
    # 检查缓存
    if prior_cache["key"] == cache_key and prior_cache["prior_token_ids"] is not None:
        print("📦 使用缓存的 vision_language_encoder 结果")
        return (
            prior_cache["prior_token_ids"],
            prior_cache["prior_image_token_ids"],
            prior_cache["prompt_embeds"],
        )
    
    # 生成新的 prior tokens
    # print("🔄 编码提示词和生成 prior tokens（无进度条，时间较长，请耐心等待）...")
    prior_token_ids = None
    prior_image_token_ids = None
    prompt_embeds = None
    
    try:
        with torch.inference_mode():
            # 生成 prior tokens（这是耗时的 vision_language_encoder 操作）
            # 总是返回 (prior_token_ids, prior_image_token_ids) 元组
            prior_token_ids, prior_image_token_ids = pipe.generate_prior_tokens(
                prompt=prompt,
                height=height,
                width=width,
                image=[image] if image is not None else None,
            )
            
            # 编码提示词
            prompt_embeds, _ = pipe.encode_prompt(
                prompt=prompt,
                do_classifier_free_guidance=False,
            )
    except Exception as e:
        print(f"❌ 生成 prior tokens 失败: {e}")
        raise
    
    # 更新缓存
    prior_cache["key"] = cache_key
    prior_cache["prior_token_ids"] = prior_token_ids
    prior_cache["prior_image_token_ids"] = prior_image_token_ids
    prior_cache["prompt_embeds"] = prompt_embeds
    
    return prior_token_ids, prior_image_token_ids, prompt_embeds



# 确保输出文件夹存在
os.makedirs("outputs", exist_ok=True)

# 默认设置
num_inference_steps_default = 50
guidance_scale_default = 1.5
width_default = 1024
height_default = 1024
res_vram = args.res_vram


def load_pipeline():
    """加载 GLM-Image 管道"""
    global pipe, mmgp
    if pipe is None:
        print("正在加载 GLM-Image 模型...")
        try:
            # 量化模型路径
            vision_lang_encoder_path = "models/GLM-Image-diffusers/vision_language_encoder-mmgp.safetensors"
            transformer_path = "models/GLM-Image-diffusers/transformer-mmgp.safetensors"
            
            # 加载 tokenizer
            tokenizer = ByT5Tokenizer.from_pretrained(
                model_id,
                subfolder="tokenizer",
                use_fast=False,
            )
            
            # 先加载基础模型到 CPU
            pipe = GlmImagePipeline.from_pretrained(
                model_id, 
                vision_language_encoder = None,
                transformer = None,
                tokenizer = tokenizer,
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
            
            # 启用 Channels Last 内存格式加速
            if device == "cuda" and hasattr(pipe, 'transformer'):
                try:
                    pipe.transformer = pipe.transformer.to(memory_format=torch.channels_last)
                    print("✅ Channels Last 内存格式已启用")
                except Exception as e:
                    print(f"⚠️ Channels Last 启用失败: {e}")
            
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
    inference_times = []
    start_time = time.time()
    
    # 处理种子
    if seed_param < 0:
        seed = random.randint(0, np.iinfo(np.int32).max)
    else:
        seed = seed_param
    
    # 确保宽高是32的倍数
    width = (width // 32) * 32
    height = (height // 32) * 32
    
    start_msg = f"🚀 开始生成，共{batch_images}张，分辨率{width}x{height}，步数{num_inference_steps}..."
    print(start_msg)
    yield None, start_msg
    
    # 检查缓存状态，如果未命中则提示
    cache_key = (prompt, height, width, get_image_hash(None))
    if prior_cache["key"] != cache_key or prior_cache["prior_token_ids"] is None:
        msg = "🔄 编码提示词和生成 prior tokens（无进度条，时间较长，请耐心等待）..."
        print(msg)
        yield None, msg

    # 获取缓存的 prior tokens（文生图时 prior_image_token_ids 为 None）
    prior_token_ids, prior_image_token_ids, prompt_embeds = get_cached_prior_tokens(
        prompt=prompt, height=height, width=width, image=None
    )
    
    try:
        for i in range(batch_images):
            if stop_generation:
                stop_generation = False
                yield results if results else None, f"✅ 生成已中止，最后种子数{seed+i-1}"
                break
            
            current_seed = seed + i
            generator = torch.Generator(device=device).manual_seed(current_seed)
            
            # 记录单张图推理开始时间
            img_start_time = time.time()
            
            # T2I 使用缓存的 prior_token_ids 和 prompt_embeds
            with torch.inference_mode():
                # 使用 yield_progress=True 获取进度
                generator_obj = pipe(
                    prompt_embeds=prompt_embeds,
                    prior_token_ids=prior_token_ids,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    yield_progress=True
                )
                
                output = None
                for res, step, total in generator_obj:
                    if res is None:
                        # 进度更新
                        progress_msg = f"🚀 生成中 {step}/{total}..."
                        yield results if results else None, progress_msg
                    else:
                        # 完成
                        output = res
            
            # 记录单张图推理时间
            img_time = time.time() - img_start_time
            inference_times.append(img_time)
            
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
            
            img_msg = f"✅ 第{i+1}张完成，种子{current_seed}，耗时{img_time:.2f}秒"
            print(img_msg)
            yield results, img_msg
        
        # 计算总时间和平均时间
        total_time = time.time() - start_time
        avg_time = total_time / len(results) if results else 0
        done_msg = f"✅ 推理完成，共{len(results)}张，总耗时{total_time:.2f}秒，平均{avg_time:.2f}秒/张"
        print(done_msg)
        yield results, done_msg
    
    except Exception as e:
        import traceback
        error_msg = f"❌ 生成失败: {str(e)}"
        print(error_msg)
        print("=" * 80)
        print("完整错误堆栈:")
        traceback.print_exc()
        print("=" * 80)
        yield results if results else None, error_msg


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
    inference_times = []
    start_time = time.time()
    
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
    
    start_msg = f"🚀 开始生成，共{batch_images}张，分辨率{width}x{height}，步数{num_inference_steps}..."
    print(start_msg)
    yield None, start_msg
    
    # 检查缓存状态
    cache_key = (prompt, height, width, get_image_hash(image))
    if prior_cache["key"] != cache_key or prior_cache["prior_token_ids"] is None:
        msg = "🔄 编码提示词和生成 prior tokens（无进度条，时间较长，请耐心等待）..."
        print(msg)
        yield None, msg

    # 获取缓存的 prior tokens（包含 prior_token_ids、prior_image_token_ids 和 prompt_embeds）
    prior_token_ids, prior_image_token_ids, prompt_embeds = get_cached_prior_tokens(
        prompt=prompt, height=height, width=width, image=image
    )
    
    try:
        for i in range(batch_images):
            if stop_generation:
                stop_generation = False
                yield results if results else None, f"✅ 生成已中止，最后种子数{seed+i-1}"
                break
            
            current_seed = seed + i
            generator = torch.Generator(device=device).manual_seed(current_seed)
            
            # 记录单张图推理开始时间
            img_start_time = time.time()
            
            # 使用缓存的 prior_token_ids、prior_image_token_ids 和 prompt_embeds
            with torch.inference_mode():
                # 使用 yield_progress=True 获取进度
                generator_obj = pipe(
                    prompt_embeds=prompt_embeds,
                    prior_token_ids=prior_token_ids,
                    prior_image_token_ids=prior_image_token_ids,
                    image=[image],
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    yield_progress=True
                )
                
                output = None
                for res, step, total in generator_obj:
                    if res is None:
                        # 进度更新
                        progress_msg = f"🚀 生成中 {step}/{total}..."
                        yield results if results else None, progress_msg
                    else:
                        # 完成
                        output = res
            
            # 记录单张图推理时间
            img_time = time.time() - img_start_time
            inference_times.append(img_time)
            
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
            
            img_msg = f"✅ 第{i+1}张完成，种子{current_seed}，耗时{img_time:.2f}秒"
            print(img_msg)
            yield results, img_msg
        
        # 计算总时间和平均时间
        total_time = time.time() - start_time
        avg_time = total_time / len(results) if results else 0
        done_msg = f"✅ 推理完成，共{len(results)}张，总耗时{total_time:.2f}秒，平均{avg_time:.2f}秒/张"
        print(done_msg)
        yield results, done_msg
    
    except Exception as e:
        import traceback
        error_msg = f"❌ 生成失败: {str(e)}"
        print(error_msg)
        print("=" * 80)
        print("完整错误堆栈:")
        traceback.print_exc()
        print("=" * 80)
        yield results if results else None, error_msg


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


def calculate_dimensions(target_area, ratio):
    """
    根据目标像素面积和宽高比计算宽高
    """
    import math
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return int(width), int(height)


def auto_adjust_resolution(image):
    """
    根据上传的图像自动调整宽高（等效 1024x1024 像素面积）
    """
    if image is None:
        return width_default, height_default, ""
    
    # 处理不同类型的输入
    if isinstance(image, dict):
        image = image.get("background", image)
    
    # 获取图像尺寸
    if hasattr(image, 'size'):  # PIL Image
        img_width, img_height = image.size
    elif hasattr(image, 'shape'):  # numpy array
        img_height, img_width = image.shape[:2]
    else:
        return width_default, height_default, ""
    
    # 计算等效 1024x1024 的尺寸（保持宽高比）
    target_area = 1024 * 1024  # 目标像素面积
    ratio = img_width / img_height
    new_width, new_height = calculate_dimensions(target_area, ratio)
    
    # 确保在有效范围内
    new_width = max(32, min(2048, new_width))
    new_height = max(32, min(2048, new_height))
    
    return new_width, new_height, f"✅ 已调整为 {new_width}x{new_height}（等效1024²）"


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
                        run_btn_t2i = gr.Button("🎨 生成图像", variant="primary", scale=2)
                        stop_button_t2i = gr.Button("⏹️ 停止", scale=1)
                    
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
                        sources=["upload", "clipboard"],
                        height=500
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
                        run_btn_i2i = gr.Button("🎨 生成图像", variant="primary", scale=2)
                        stop_button_i2i = gr.Button("⏹️ 停止", scale=1)
                    
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
                    



                
                with gr.Column(scale=1):
                    info_i2i = gr.Textbox(
                        label="信息",
                        lines=3,
                        interactive=False
                    )
                    result_i2i = gr.Gallery(
                        label="生成结果",
                        show_label=True,
                        elem_id="gallery_i2i",
                        columns=2,
                        rows=2,
                        height="auto"
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
    
    # 上传图像后自动调整宽高
    image_i2i.change(
        fn=auto_adjust_resolution,
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
