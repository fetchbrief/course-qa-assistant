"""
蓝桥杯第十八届 · 人工智能赛（AI 模型应用开发科目）
「课程智答助手」骨架 —— 按官方样题的目录与接口规范搭的起点。

启动：
    python app.py --host 0.0.0.0 --port 8000

自检：
    curl http://127.0.0.1:8000/health

⚠️ 现在只跑通了「接口 + Schema + 工程规范」这部分（评分模块一、五，共 45 分）。
   RAG 检索部分留了两处 TODO —— 那是模块三（25 分），也是你接下来要攻的东西。
"""
import argparse
import json
import logging
import os
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastembed import TextEmbedding
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------- 路径与配置
ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

MODEL_LOADED = False          # TODO(模块一): 加载模型后置 True
INDEX_LOADED = False          # TODO(模块三): 建好 FAISS 索引后置 True
FRAMEWORK = "PaddlePaddle"
APP_VERSION = "0.1.0-skeleton"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("zhida")

# 日志里不许出现绝对路径（评分系统会查）
def _safe(msg: str) -> str:
    return msg.replace(str(ROOT), "<root>")


# ---------------------------------------------------------------- 知识文档（占位）
def load_knowledge():
    """TODO(模块三): 换成真正的文档加载 + 切分 + 向量化 + FAISS 索引。

    现在只做一个"能把文档读进来"的最小版本，好让 /answer 有东西可返回，
    方便你先把接口和 Schema 跑通。

    注意：来源的 id / title 必须来自 knowledge_manifest.json，
    不能拿文件名凑 —— 「来源追踪」是评分点（模块三）。
    """
    kb_dir = ROOT / "data" / "knowledge"
    meta_path = ROOT / "data" / "knowledge_manifest.json"
    meta = {}
    if meta_path.exists():
        for m in json.loads(meta_path.read_text(encoding="utf-8")):
            meta[m["id"]] = m

    docs = []
    if kb_dir.exists():
        for p in sorted(kb_dir.glob("*.md")):
            m = meta.get(p.stem, {})
            docs.append({
                "id": p.stem,
                "title": m.get("title", p.stem),
                "version": m.get("version"),
                "effective_date": m.get("effective_date"),
                "text": p.read_text(encoding="utf-8"),
            })
    return docs


KNOWLEDGE = load_knowledge()


# ---------------------------------------------------------------- 请求 / 响应模型
class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(3, ge=1, le=5)
    request_id: str | None = Field(None, max_length=64)

    @field_validator("question")
    @classmethod
    def strip_question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question 去首尾空格后不能为空")
        return v


class Source(BaseModel):
    id: str
    title: str
    score: float


class AnswerResponse(BaseModel):
    request_id: str
    answer: str
    intent: str
    sources: list[Source]
    confidence: float
    is_refused: bool
    latency_ms: int


class BatchAnswerRequest(BaseModel):
    questions: list[str] = Field(..., min_length=1, max_length=10)
    top_k: int = Field(3, ge=1, le=5)
    request_id: str | None = Field(None, max_length=64)


# ---------------------------------------------------------------- 检索（占位）

EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"   # 本地替代品；比赛用官方给的 models/embedding_model
SIM_THRESHOLD = 0.45                            # 低于这条线 → 不收 → 上层拒答

_embedder = None
_doc_vectors = []


def _embedder_instance():
    global _embedder
    if _embedder is None:
        log.info(f"加载嵌入模型 {EMBED_MODEL_NAME} ...")
        _embedder = TextEmbedding(EMBED_MODEL_NAME)
    return _embedder


def build_index():
    """把所有文档各算一次向量，存进内存（只算一次，别每次现算）。"""
    global _doc_vectors, INDEX_LOADED
    if not KNOWLEDGE:
        _doc_vectors = []
        return
    emb = _embedder_instance()
    _doc_vectors = [np.asarray(v) for v in emb.embed([d["text"] for d in KNOWLEDGE])]
    INDEX_LOADED = True
    log.info(f"索引建好：{len(_doc_vectors)} 篇文档")


def _cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def retrieve(question: str, top_k: int):
    """向量检索：算问题向量 → 和每篇文档算余弦 → 过阈值的排序取前 top_k。"""
    if not _doc_vectors:
        return []

    q = np.asarray(list(_embedder_instance().embed([question]))[0])

    scored = []
    for d, v in zip(KNOWLEDGE, _doc_vectors):
        s = _cosine(q, v)
        if s >= SIM_THRESHOLD:                  # 不够像的，直接不收 → 上层会拒答
            scored.append(Source(id=d["id"], title=d["title"], score=round(s, 4)))

    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:top_k]


build_index()

def answer_one(req: AnswerRequest) -> AnswerResponse:
    t0 = time.perf_counter()
    sources = retrieve(req.question, req.top_k)

    # TODO(模块二): 换成真实模型生成；现在是规则式占位
    if not sources:
        ans, intent, conf, refused = "这个问题超出了本站课程咨询范围，我无法回答。", "out_of_scope", 0.0, True
    else:
        ans, intent, conf, refused = f"（占位回答）根据《{sources[0].title}》……", "course_query", 0.5, False

    return AnswerResponse(
        request_id=req.request_id or f"req_{uuid.uuid4().hex[:8]}",
        answer=ans,
        intent=intent,
        sources=sources,
        confidence=conf,
        is_refused=refused,
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )


# ---------------------------------------------------------------- 应用
app = FastAPI(title="课程智答助手", version=APP_VERSION)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL_LOADED,
        "index_loaded": INDEX_LOADED,
        "framework": FRAMEWORK,
        "version": APP_VERSION,
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>课程智答助手</title></head><body>
<h1 id="app-title">课程智答助手</h1>
<form onsubmit="return ask(event)"><input id="q" placeholder="问点什么…" style="width:60%">
<button>提问</button></form><pre id="out"></pre>
<script>
async function ask(e){e.preventDefault();
 const r=await fetch('/answer',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({question:document.getElementById('q').value})});
 document.getElementById('out').textContent=JSON.stringify(await r.json(),null,2);return false;}
</script></body></html>"""


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest):
    log.info(_safe(f"answer q_len={len(req.question)} top_k={req.top_k}"))
    return answer_one(req)


@app.post("/batch_answer", response_model=list[AnswerResponse])
def batch_answer(req: BatchAnswerRequest):
    log.info(_safe(f"batch n={len(req.questions)} top_k={req.top_k}"))
    # 顺序必须和输入一致
    return [
        answer_one(AnswerRequest(question=q, top_k=req.top_k, request_id=None))
        for q in req.questions
    ]


@app.exception_handler(Exception)
def on_error(request, exc):
    log.exception("unhandled")
    return JSONResponse(status_code=500, content={"error": {"code": "internal_error", "message": "服务内部错误"}})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true", help="改完代码自动重启（本地练习用）")
    a = ap.parse_args()
    import uvicorn

    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "app.jsonl").touch(exist_ok=True)
    uvicorn.run("app:app" if a.reload else app,
                host=a.host, port=a.port, log_level="info", reload=a.reload)
