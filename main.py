import sys 
import modal 
 

evo2_image=(modal.image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04",add_python="3.12.10")
            .apt_install(["build-essential","cmake","ninja-build","libcudnn8","libcudnn8-dev","git","gcc","g++"])
            .env({
                "CC":"/usr/bin/gcc",
                "CXX":"/usr/bin/g++"
            })
            .run_commands("git clone https://github.com/arcinstitute/evo2 && cd evo2 && pip install -e .")
            .run_commands("pip uninstall -y transformer-engine transformer_engine")
            .run_command("pip install 'transformer_engine[pytorch]=1.13'--no-build-isolation"))



app=modal.App("example-hello-world")

@app.function()
def f(i):
    if i % 2 == 0:
        print("hello", i)
    else:
        print("world", i, file=sys.stderr)

    return i * i

@app.local_entrypoint()
def main():
    # run the function locally
    print(f.local(1000))

    # run the function remotely on Modal
    print(f.remote(1000))

    # run the function in parallel and remotely on Modal
    total = 0
    for ret in f.map(range(200)):
        total += ret

    print(total)

