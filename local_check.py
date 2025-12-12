import sys
import platform
try:
    import torch
    print(f"Torch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("Torch not installed")

print(f"OS: {platform.system()} {platform.release()}")
print(f"Python: {sys.version}")
