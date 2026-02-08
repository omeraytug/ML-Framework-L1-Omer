import sys
import torch
import pandas as pd
import sklearn

def main():
    print("Python version:", sys.version)
    print("PyTorch version:", torch.__version__)
    print("Pandas version:", pd.__version__)
    print("Scikit-learn version:", sklearn.__version__)

    # GPU / accelerator check
    if torch.backends.mps.is_available():
        device = "mps"
        print("Accelerator: Apple MPS (GPU)")
    elif hasattr(torch.version, 'hip') and torch.version.hip is not None:
        # Check for AMD ROCm (HIP) - must check before CUDA since ROCm uses CUDA API
        device = "cuda"  # ROCm uses CUDA API compatibility
        print("Accelerator: AMD ROCm (GPU)")
        if torch.cuda.is_available():
            print(f"  ROCm version: {torch.version.hip}")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Accelerator: CUDA GPU (NVIDIA)")
    else:
        device = "cpu"
        print("Accelerator: CPU only")

    # Tensor computation
    x = torch.tensor([1.0, 2.0, 3.0], device=device)

    print("Tensor computation result:", x)
    print("Device used:", x.device)

if __name__ == "__main__":
    main()