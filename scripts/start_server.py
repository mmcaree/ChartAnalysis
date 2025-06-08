import os
import subprocess
import sys

# Configuration
model_path = "C:\\Users\\mmcar\\Desktop\\Dev\\ChartAnalysis\\llama.cpp\\models\\Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
host = "0.0.0.0"
port = 8000
n_ctx = 4096  # Context window size
n_gpu_layers = -1  # Use all GPU layers
seed = 42
verbose = True

if __name__ == "__main__":
    print(f"Starting server with model: {model_path}")
    print(f"Context window: {n_ctx}, GPU Layers: {n_gpu_layers}")
    
    # Build the command
    cmd = [
        sys.executable, "-m", "llama_cpp.server",
        "--model", model_path,
        "--host", host,
        "--port", str(port),
        "--n_ctx", str(n_ctx),
        "--n_gpu_layers", str(n_gpu_layers),
        "--seed", str(seed),
        "--verbose", str(verbose).lower()  # Convert to lowercase 'true'
    ]
    
    print("Running command:", " ".join(cmd))
    
    # Run the command
    subprocess.run(cmd)