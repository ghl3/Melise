"""
Piece 1: Hello, GPU.

The single most important idea in GPU programming: your Python code runs on
the CPU, but tensors can live in two different memories — host (CPU RAM) and
device (GPU RAM). Moving a tensor between them is an explicit, observable act.
"""

import torch

# 1. Pick a device. On Apple Silicon the GPU shows up as "mps" (Metal
#    Performance Shaders). On a CUDA box it would be "cuda". On any machine,
#    "cpu" always exists. We assert here so that if MPS isn't available we
#    fail loudly instead of silently running on CPU.
assert torch.backends.mps.is_available(), "MPS not available on this machine"
device = torch.device("mps")
print(f"using device: {device}")

# 2. Make a tensor on the CPU. At this moment the numbers `[1, 2, 3, 4]` live
#    in regular system RAM, reachable by Python like any other object.
x_cpu = torch.tensor([1.0, 2.0, 3.0, 4.0])
print(f"x_cpu       device={x_cpu.device}  data_ptr={hex(x_cpu.data_ptr())}")

# 3. Copy it to the GPU. `.to(device)` allocates GPU memory, copies the bytes
#    across the PCIe-equivalent bus (on Apple Silicon it's unified memory, so
#    physically nothing moves — but the *logical* device changes), and returns
#    a new tensor handle. The original x_cpu is untouched.
x_gpu = x_cpu.to(device)
print(f"x_gpu       device={x_gpu.device}  data_ptr={hex(x_gpu.data_ptr())}")

# 4. Do some math. This `*` is *not* executed by Python. PyTorch builds a
#    Metal command, hands it to the GPU driver, and the GPU does the multiply.
#    The result tensor `y` lives on the GPU.
y = x_gpu * 2 + 1
print(f"y           device={y.device}")

# 5. To *see* the numbers from Python, we have to copy them back to the CPU.
#    `.cpu()` is the inverse of `.to(device)`. Forgetting this step is one of
#    the most common bugs when starting out.
y_cpu = y.cpu()
print(f"y_cpu       device={y_cpu.device}  values={y_cpu.tolist()}")
