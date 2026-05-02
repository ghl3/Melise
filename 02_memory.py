"""
Piece 2: Watching the GPU caching allocator.

PyTorch never asks the driver for one tensor's worth of memory at a time. It
asks for big chunks, hands out slices to your tensors, and when those tensors
die it keeps the chunks for the *next* allocation. That layer is the "caching
allocator," and once you can see it, GPU memory stops being mysterious.

We'll use two MPS-specific introspection calls:

  torch.mps.current_allocated_memory()
      bytes currently held by *live tensors* (the user-visible view)

  torch.mps.driver_allocated_memory()
      bytes the allocator has obtained from the Metal driver — i.e. how much
      GPU memory PyTorch is *holding* on behalf of this process, including
      cached-but-unused pages.

The gap between them is the cache.
"""

import torch

assert torch.backends.mps.is_available()
device = torch.device("mps")

MB = 1024 * 1024


def show(label: str) -> None:
    used = torch.mps.current_allocated_memory() / MB
    held = torch.mps.driver_allocated_memory() / MB
    print(f"{label:<35} live={used:7.1f} MB   held by allocator={held:7.1f} MB")


show("baseline")

# Allocate ~128 MB of float32. 4096 * 8192 * 4 bytes = 128 MiB.
# Both numbers should jump: nothing was cached, so the allocator had to ask
# the driver for fresh pages.
a = torch.empty(4096, 8192, device=device)
show("after allocating 128 MB")

# Drop the Python reference. The destructor runs; the buffer goes back to the
# allocator's free pool but is *not* returned to the driver. So `live` falls
# but `held` does not.
del a
show("after del a")

# Allocate a tensor of the same size. The allocator finds a perfectly-shaped
# free block in its cache and reuses it. No driver call. `held` should stay
# put while `live` climbs back up.
b = torch.empty(4096, 8192, device=device)
show("after re-allocating same size")

# Allocate something bigger that won't fit any cached block. The allocator
# has to ask the driver for more. `held` grows.
c = torch.empty(8192, 8192, device=device)  # 256 MB
show("after allocating 256 MB more")

del b, c
show("after del b, c")

# Force the allocator to actually release cached pages back to the driver.
# This is the only way `held` comes down without ending the process.
torch.mps.empty_cache()
show("after empty_cache()")
