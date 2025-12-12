import modal
import torch

app = modal.App("debug-gpu", image=modal.Image.from_registry("nvcr.io/nvidia/pytorch:25.09-py3"))

@app.function(gpu="H100")
def check():
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")

@app.local_entrypoint()
def main():
    check.remote()
