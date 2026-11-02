"""Container sanity check: accelerator visibility + a small matmul."""
import time

import torch

print("torch", torch.__version__)
if hasattr(torch, "xpu"):
    print("xpu available:", torch.xpu.is_available())
    if torch.xpu.is_available():
        for i in range(torch.xpu.device_count()):
            print(f"  xpu:{i}", torch.xpu.get_device_name(i), torch.xpu.get_device_properties(i))
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  cuda:{i}", torch.cuda.get_device_name(i))

dev = "cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu"
for dtype in (torch.float32, torch.bfloat16):
    a = torch.randn(2048, 2048, device=dev, dtype=dtype)
    b = torch.randn(2048, 2048, device=dev, dtype=dtype)
    c = a @ b
    torch.accelerator.synchronize() if dev != "cpu" else None
    t = time.perf_counter()
    for _ in range(10):
        c = a @ b
    torch.accelerator.synchronize() if dev != "cpu" else None
    dt = (time.perf_counter() - t) / 10
    ref = (a.float().cpu() @ b.float().cpu())
    err = (c.float().cpu() - ref).abs().max().item() / ref.abs().max().item()
    print(f"{dev} {dtype} matmul 2048^2: {2*2048**3/dt/1e12:.2f} TFLOPS, rel err {err:.2e}")
