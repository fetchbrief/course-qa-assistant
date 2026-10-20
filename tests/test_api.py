"""
接口回归测试 —— 改完代码跑一遍，确认没把原来的功能改坏。

跑法：
    cd C:/Users/admin/Desktop/ai/lanqiao
    .venv/Scripts/python.exe -m pytest tests/ -v

设计原则：
  1. 只用 mock 后端 —— 测试不花钱、不依赖网络
  2. 每条断言对应一个"已知问题"，改坏了立刻能看出来
"""

import os
import sys
import time

os.environ["NO_PROXY"] = ",".join(
    x for x in os.environ.get("NO_PROXY", "127.0.0.1,localhost").replace("[::1]", "").split(",") if x
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

import app as app_module


@pytest.fixture(scope="module")
def client():
    # 强制用 mock 后端：测试不许调真实模型
    app_module.GEN_CFG["backend"] = "mock"
    app_module._CACHE.clear()
    with TestClient(app_module.app) as c:
        yield c


def post(client, question, top_k=3):
    return client.post("/answer", json={"question": question, "top_k": top_k})


# ---------------------------------------------------------------- 基础可用性

def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    # 索引建好了，模型也应该标记为已加载（不信接口里写死的 False）
    assert data["index_loaded"] is True
    assert data["model_loaded"] is True


def test_config_endpoint_returns_effective_values(client):
    r = client.get("/config")
    assert r.status_code == 200
    data = r.json()
    assert data["retrieval"]["sim_threshold"] > 0
    assert data["retrieval"]["chunk_max"] > 0
    assert data["generation"]["system_prompt_chars"] > 0


# ---------------------------------------------------------------- 拒答链路

def test_answerable_question_returns_sources(client):
    r = post(client, "直播断线了怎么办？")
    assert r.status_code == 200
    data = r.json()
    assert data["is_refused"] is False
    assert len(data["sources"]) >= 1
    assert data["sources"][0]["score"] > 0


def test_unrelated_question_is_refused(client):
    """「今天天气怎么样」和知识库完全无关 → 必须拒答"""
    r = post(client, "今天天气怎么样？")
    assert r.status_code == 200
    data = r.json()
    assert data["is_refused"] is True
    assert data["sources"] == []


def test_synonym_question_still_matches(client):
    """同义改写：和文档没有一个共同的字面词，也该匹配上（这是向量的核心价值）"""
    r = post(client, "我想把钱退回来")
    data = r.json()
    assert data["is_refused"] is False, "同义问法不该被拒"
    assert any("退费" in s["title"] or "refund" in s["id"] for s in data["sources"])


# ---------------------------------------------------------------- 参数校验

@pytest.mark.parametrize("bad", [
    {"question": ""},                    # 空
    {"question": "   "},                 # 全是空格
    {"question": "x" * 501},             # 超长
    {"question": "正常问题", "top_k": 0},  # top_k 越界
    {"question": "正常问题", "top_k": 9},  # top_k 越界
])
def test_invalid_request_returns_422(client, bad):
    r = client.post("/answer", json=bad)
    assert r.status_code == 422


# ---------------------------------------------------------------- 缓存

def test_cache_makes_repeat_fast(client):
    """同一个问题问第二遍，应该走缓存，明显更快"""
    client.post("/cache/clear")
    q = "作业迟交会扣分吗？"

    t0 = time.perf_counter()
    first = post(client, q)
    first_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    second = post(client, q)
    second_ms = (time.perf_counter() - t0) * 1000

    assert first.json()["answer"] == second.json()["answer"], "缓存命中后答案必须一致"
    assert second_ms < first_ms, f"缓存没生效：第一次 {first_ms:.1f}ms，第二次 {second_ms:.1f}ms"


def test_cache_stats_increment(client):
    client.post("/cache/clear")
    post(client, "账号可以借给同学吗？")
    post(client, "账号可以借给同学吗？")
    data = client.get("/health").json()
    assert data["cache"]["hit"] >= 1
    assert data["cache"]["size"] >= 1


def test_cache_clear(client):
    post(client, "退费多久到账？")
    r = client.post("/cache/clear")
    assert r.status_code == 200
    assert client.get("/health").json()["cache"]["size"] == 0


# ---------------------------------------------------------------- 批量

def test_batch_answer_keeps_order(client):
    qs = ["直播断线了怎么办？", "作业迟交会扣分吗？", "今天天气怎么样？"]
    r = client.post("/batch_answer", json={"questions": qs, "top_k": 3})
    assert r.status_code == 200
    data = r.json()
    assert len(data) == len(qs)
