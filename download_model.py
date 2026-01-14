"""
从 ModelScope 下载 GLM-Image 模型
"""
import os
import sys

# 设置输出编码为 UTF-8（Windows 兼容）
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def download_model():
    """从 ModelScope 下载 GLM-Image 模型"""
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("错误: 请先安装 modelscope: pip install modelscope")
        sys.exit(1)
    
    model_id = "ZhipuAI/GLM-Image"
    local_dir = "models/GLM-Image-diffusers"
    
    print(f"正在从 ModelScope 下载模型: {model_id}")
    print(f"保存路径: {local_dir}")
    print("这可能需要一些时间，请耐心等待...")
    
    try:
        # 确保目录存在
        os.makedirs("models", exist_ok=True)
        
        # 下载模型
        snapshot_download(
            model_id,
            cache_dir=local_dir,
            local_dir=local_dir
        )
        
        print(f"成功: 模型下载完成！")
        print(f"模型已保存到: {os.path.abspath(local_dir)}")
        
    except Exception as e:
        print(f"错误: 下载失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    download_model()
