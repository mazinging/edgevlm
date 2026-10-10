"""验证核心假设：INT8 的收益取决于 GEMM 的 M 维度.

预测：
  * M=1（decode）      -> INT8 应无加速甚至更慢
  * M 大（prefill/batch）-> INT8 应有显著加速

若预测成立，则「INT8 对端侧单流解码无益、对 prefill/批处理有益」的结论成立。
这是 M2 发现 A 的直接可证伪检验。
"""
import os, sys, time
WS = r"E:\VLM project"
sys.path.insert(0, os.path.join(WS, "src"))
import torch

dev = "cuda"
torch.manual_seed(0)

def ramp(sec=3.0):
    a = torch.randn(2048, 2048, device=dev)
    t = time.perf_counter() + sec
    while time.perf_counter() < t:
        for _ in range(10): _ = a @ a
        torch.cuda.synchronize()
    del a; torch.cuda.empty_cache()

def bench(fn, iters=50, warm=10):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

ramp()

# 模拟 LLM 解码的 GEMV/GEMM： (M, K) @ (K, N)
K, N = 576, 576          # SmolVLM text_model 的 hidden 维
print("K=%d N=%d (SmolVLM text hidden)" % (K, N))
print("%-8s %12s %12s %10s" % ("M", "fp32(ms)", "int8(ms)", "speedup"))
print("-" * 46)

results = []
for M in [1, 2, 4, 8, 16, 17, 32, 64, 128, 256, 512]:
    x32 = torch.randn(M, K, device=dev)
    w32 = torch.randn(K, N, device=dev)

    xi = torch.randint(-127, 127, (M, K), dtype=torch.int8, device=dev)
    wi = torch.randint(-127, 127, (K, N), dtype=torch.int8, device=dev)

    t32 = bench(lambda: x32 @ w32, iters=100)

    # int8 走 _int_mm（需 M>16），否则回退
    if M > 16:
        ti = bench(lambda: torch._int_mm(xi, wi), iters=100)
    else:
        ti = None

    if ti is not None:
        sp = t32 / ti
        print("%-8d %12.4f %12.4f %9.2fx" % (M, t32*1000, ti*1000, sp))
        results.append((M, t32, ti, sp))
    else:
        print("%-8d %12.4f %12s %10s" % (M, t32*1000, "N/A(TC需M>16)", "-"))

print()
print("=== 更贴近实际的 hidden=1536（MLP 中间维）===")
K2, N2 = 576, 1536
print("%-8s %12s %12s %10s" % ("M", "fp32(ms)", "int8(ms)", "speedup"))
print("-" * 46)
for M in [1, 8, 16, 32, 128, 512]:
    x32 = torch.randn(M, K2, device=dev); w32 = torch.randn(K2, N2, device=dev)
    xi = torch.randint(-127,127,(M,K2),dtype=torch.int8,device=dev)
    wi = torch.randint(-127,127,(K2,N2),dtype=torch.int8,device=dev)
    t32 = bench(lambda: x32 @ w32, iters=100)
    if M > 16:
        ti = bench(lambda: torch._int_mm(xi, wi), iters=100)
        print("%-8d %12.4f %12.4f %9.2fx" % (M, t32*1000, ti*1000, t32/ti))
    else:
        print("%-8d %12.4f %12s %10s" % (M, t32*1000, "N/A", "-"))

print()
print("=== 大 N（lm_head, vocab=49280）===")
K3, N3 = 576, 49280
for M in [1, 16, 64, 256]:
    x32 = torch.randn(M, K3, device=dev); w32 = torch.randn(K3, N3, device=dev)
    xi = torch.randint(-127,127,(M,K3),dtype=torch.int8,device=dev)
    wi = torch.randint(-127,127,(K3,N3),dtype=torch.int8,device=dev)
    t32 = bench(lambda: x32 @ w32, iters=50, warm=5)
    if M > 16:
        ti = bench(lambda: torch._int_mm(xi, wi), iters=50, warm=5)
        print("M=%-6d fp32=%8.3f ms  int8=%8.3f ms  %6.2fx" % (M, t32*1000, ti*1000, t32/ti))
    else:
        print("M=%-6d fp32=%8.3f ms  int8=N/A (TC需M>16)" % (M, t32*1000))
    del x32, w32, xi, wi
    torch.cuda.empty_cache()

print("\n结论检验：")
print("  若 M=1 无法使用 TC 而 M>=17 有加速 -> 「INT8 收益依赖 GEMM 形状」成立")
