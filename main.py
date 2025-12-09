# from dbm import gnu
import sys
import modal

# -----------------------------
# Build Image
# -----------------------------

evo2_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.12"
    )
    .apt_install(
        [
            "build-essential",
            "cmake",
            "ninja-build",
            "libcudnn8",
            "libcudnn8-dev",
            "git",
            "gcc",
            "g++",
            "wget",
            "curl",
        ]
    )
    .env({"CC": "/usr/bin/gcc", "CXX": "/usr/bin/g++"})
    .run_commands(
        [
            "git clone https://github.com/arcinstitute/evo2",
            "cd evo2 && pip install -e ."
        ]
    )
    .run_commands(
        [
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
        ]
    )
    .pip_install_from_requirements("requirments.txt")
)

# -----------------------------
# Modal App
# -----------------------------

app = modal.App("variant-analysis-evo2", image=evo2_image)


@app.function(gpu="H100")
def test():
    print("Testing")


@app.local_entrypoint()
def main():
    test.remote()
