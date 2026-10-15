r"""Params, GFLOPs at 640 and batch-1 fp16 latency (PyTorch, this GPU) of RGB models; the Jetson number must be measured on the ORIN.
    python3 src/eval/model_cost.py NAME=weights.pt ... """
import sys, time
import torch
from ultralytics import YOLO
rows = []
for spec in sys.argv[1:]:
    name, pt = spec.split("=", 1)
    m = YOLO(pt); net = m.model.fuse().eval().cuda().half()
    p = sum(x.numel() for x in net.parameters())
    from ultralytics.utils.torch_utils import get_flops
    g = get_flops(net, 640)
    x = torch.zeros(1, 3, 640, 640, device="cuda", dtype=torch.half)
    with torch.no_grad():
        for _ in range(20): net(x)
        torch.cuda.synchronize(); t = time.time()
        for _ in range(200): net(x)
        torch.cuda.synchronize(); ms = (time.time() - t) / 200 * 1000
    rows.append((name, p / 1e6, g, ms))
print("| model | params (M) | GFLOPs @640 | latency ms (L4, fp16, batch 1, PyTorch) |\n|---|---|---|---|")
for n, p, g, ms in rows: print(f"| {n} | {p:.2f} | {g:.1f} | {ms:.2f} |")
