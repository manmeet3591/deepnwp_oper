import torch

# Check if CUDA (GPU support) is available
gpu_available = torch.cuda.is_available()

print("GPU Available:", gpu_available)

# If a GPU is available, print the GPU name
if gpu_available:
    print("GPU Name:", torch.cuda.get_device_name(0))

