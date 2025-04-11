import torch

def check_memory(device: str = "mps") -> None:
    """
    Print the memory usage on Apple MPS or CUDA GPU.

    Parameters
    ----------
    device : str - 'mps' for Apple Silicon or 'cuda' for standard NVIDIA GPU usage
    """
    if device == "mps":
        allocated = torch.mps.current_allocated_memory() / 1024 / 1024 / 1024
        total = torch.mps.recommended_max_memory() / 1024 / 1024 / 1024
        print(f"Allocated: {allocated:.4f} GB")
        print(f"Total: {total:.4f} GB")
    elif device == "cuda":
        allocated = torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024
        reserved = torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024
        total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
        print(f"Allocated: {allocated:.4f} GB")
        print(f"Reserved : {reserved:.4f} GB")
        print(f"Total    : {total:.4f} GB")
    else:
        print("No GPU device to check or device not recognized.")

