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
    elif torch.cuda.is_available():
        device = "cuda"
        print("Accelerator: CUDA GPU")
    else:
        device = "cpu"
        print("Accelerator: CPU only")

    # Tensor computation
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    y = torch.tensor([4.0, 5.0, 6.0], device=device)
    z = x + y

    print("Tensor computation result:", z)
    print("Device used:", z.device)

if __name__ == "__main__":
    main()