import torch
import subprocess
import sys

def main():
    # Check if CUDA is available
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # Check NVIDIA driver version
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        print("\nNVIDIA-SMI Output:")
        print(result.stdout)
    except Exception as e:
        print(f"Error running nvidia-smi: {e}")
    
    # Check installed packages
    print("\nRelevant installed packages:")
    packages = ["llama-cpp-python"]
    for package in packages:
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "show", package], 
                                   capture_output=True, text=True)
            print(result.stdout)
        except Exception as e:
            print(f"Error checking {package}: {e}")

if __name__ == "__main__":
    main()