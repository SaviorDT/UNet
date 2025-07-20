import torch

def check_gpu():
    if torch.cuda.is_available():
        print(f"GPU 可用！")
        print(f"GPU 名稱: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 記憶體總量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("GPU 不可用，請檢查 CUDA 驅動或硬體配置。")

if __name__ == "__main__":
    check_gpu()