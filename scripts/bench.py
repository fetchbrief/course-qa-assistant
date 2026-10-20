"""
压测脚本 —— 量「连续请求」有多快。

跑法：
    cd C:/Users/admin/Desktop/ai/lanqiao
    .venv/Scripts/python.exe scripts/bench.py

输出：
    p50 / p95 / max 延迟，以及每秒能处理多少个请求（QPS）

为什么用 p95 而不是平均值：
    平均值会把少数"特别慢"的请求平均掉。
    p95 = 100 个请求里最慢那 5 个的门槛 —— 用户感受到的卡顿，就在这。
"""

import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

os.environ["NO_PROXY"] = ",".join(
    x for x in os.environ.get("NO_PROXY", "127.0.0.1,localhost").replace("[::1]", "").split(",") if x
)

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app as app_module

client = TestClient(app_module.app)

QUESTIONS = [
    "直播断线了怎么办？",
    "作业迟交会扣分吗？",
    "账号可以借给同学吗？",
    "退费多久到账？",
    "结业证书怎么下载？",
]


def 一次请求(i: int) -> float:
    q = QUESTIONS[i % len(QUESTIONS)]
    t0 = time.perf_counter()
    r = client.post("/answer", json={"question": q, "top_k": 3})
    dt = (time.perf_counter() - t0) * 1000
    assert r.status_code == 200, r.text
    return dt


def 统计(name: str, 耗时列表):
    耗时列表 = sorted(耗时列表)
    n = len(耗时列表)
    p50 = 耗时列表[int(n * 0.50) - 1]
    p95 = 耗时列表[min(int(n * 0.95), n - 1)]
    print(f"\n{name}")
    print(f"  请求数   : {n}")
    print(f"  p50      : {p50:8.1f} ms")
    print(f"  p95      : {p95:8.1f} ms")
    print(f"  max      : {耗时列表[-1]:8.1f} ms")
    print(f"  QPS      : {n / (sum(耗时列表) / 1000):8.1f}")
    return p50, p95


if __name__ == "__main__":
    print("预热（第一次请求要加载模型）...")
    一次请求(0)

    # 1) 串行 20 次
    seq = [一次请求(i) for i in range(20)]
    统计("【串行 20 次】", seq)

    # 2) 并发 40 次（8 个线程同时打）
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        conc = list(pool.map(一次请求, range(40)))
    总耗时 = time.perf_counter() - t0
    p50, p95 = 统计("【并发 40 次 / 8 线程】", conc)
    print(f"  总耗时   : {总耗时:8.2f} s")
    print(f"  实际 QPS : {40 / 总耗时:8.1f}")
