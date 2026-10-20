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
import re
import time
import uuid
from pathlib import Path

import httpx
import numpy as np
import yaml
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


# ---------------------------------------------------------------- 配置加载
# 规则：代码里不写死参数，一律从 config/*.yaml 读。
# 文件缺失 → 用代码里的默认值兜底（比直接崩掉好），但打一条警告。

def _load_yaml(rel_path: str) -> dict:
    p = ROOT / rel_path
    if not p.exists():
        log.warning(_safe(f"配置文件不存在，用默认值：{rel_path}"))
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


MODEL_CFG = _load_yaml("config/model.yaml")
GEN_CFG = _load_yaml("config/generation.yaml")
RETR_CFG = _load_yaml("config/retrieval.yaml")


def _load_prompt(rel_path: str) -> str:
    p = ROOT / rel_path
    if not p.exists():
        log.warning(_safe(f"提示词文件不存在：{rel_path}"))
        return ""
    return p.read_text(encoding="utf-8").strip()


SYSTEM_PROMPT = _load_prompt(GEN_CFG.get("system_prompt_file", "prompts/system.txt"))


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

EMBED_MODEL_NAME = MODEL_CFG.get("embedding_model_name", "BAAI/bge-small-zh-v1.5")
SIM_THRESHOLD = float(RETR_CFG.get("sim_threshold", 0.45))   # 低于这条线 → 不收 → 上层拒答
CHUNK_MAX = int(RETR_CFG.get("chunk_max", 400))
CHUNK_OVERLAP = int(RETR_CFG.get("chunk_overlap", 80))
DEFAULT_TOP_K = int(RETR_CFG.get("top_k_default", 3))

_embedder = None
_chunks = []             # [{"id","title","text"}]  一段一条
_chunk_vectors = []


def _embedder_instance():
    global _embedder, MODEL_LOADED
    if _embedder is None:
        log.info(f"加载嵌入模型 {EMBED_MODEL_NAME} ...")
        _embedder = TextEmbedding(EMBED_MODEL_NAME)
        MODEL_LOADED = True
    return _embedder


def split_doc(text: str):
    """把一篇文档切成若干段。

    规则（按顺序）：
      1. 有 '## 小标题' → 按小标题切
      2. 没小标题 → 整篇当一段
      3. 某段仍超过 CHUNK_MAX → 按长度硬切，相邻段重叠 CHUNK_OVERLAP 字
    """
    pieces = []
    if "\n## " in text or text.startswith("## "):
        parts = re.split(r"\n?## ", text)
        head = parts[0].strip()
        if head:
            pieces.append(("", head))
        for p in parts[1:]:
            label, _, body = p.partition("\n")
            body = (label + "\n" + body).strip()
            if body:
                pieces.append(("## " + label.strip(), body))
    else:
        pieces.append(("", text.strip()))

    out = []
    for label, body in pieces:
        if len(body) <= CHUNK_MAX:
            out.append((label, body))
        else:
            start = 0
            while start < len(body):
                out.append((label, body[start:start + CHUNK_MAX]))
                start += CHUNK_MAX - CHUNK_OVERLAP
    return out


def build_index():
    """把每篇文档切成段，每段算一次向量，存进内存（只算一次）。"""
    global _chunks, _chunk_vectors, INDEX_LOADED
    _chunks, texts = [], []
    for d in KNOWLEDGE:
        for label, body in split_doc(d["text"]):
            _chunks.append({
                "id": d["id"],
                "title": f"{d['title']} · {label}" if label else d["title"],
                "text": body,
            })
            texts.append(body)

    if not texts:
        _chunk_vectors = []
        return

    emb = _embedder_instance()
    _chunk_vectors = [np.asarray(v) for v in emb.embed(texts)]
    INDEX_LOADED = True
    log.info(f"索引建好：{len(_chunks)} 段（来自 {len(KNOWLEDGE)} 篇文档）")


def _cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def retrieve(question: str, top_k: int):
    """向量检索：问题向量 → 和每一【段】算余弦 → 过阈值 → 按文档去重 → 取前 top_k。"""
    if not _chunk_vectors:
        return []

    q = np.asarray(list(_embedder_instance().embed([question]))[0])

    scored = []
    for ch, v in zip(_chunks, _chunk_vectors):
        s = _cosine(q, v)
        if s >= SIM_THRESHOLD:                  # 不够像的，直接不收 → 上层会拒答
            scored.append(Source(id=ch["id"], title=ch["title"], score=round(s, 4)))

    scored.sort(key=lambda x: x.score, reverse=True)

    # 同一篇文档可能有多段命中，只保留分最高的那段
    seen, deduped = set(), []
    for s in scored:
        if s.id in seen:
            continue
        seen.add(s.id)
        deduped.append(s)
    return deduped[:top_k]


build_index()

# 段标题 → 段正文。检索只返回标题和分数，生成时要拿正文拼 Prompt。
_CHUNK_TEXT = {c["title"]: c["text"] for c in _chunks}

# 模型说这句话时，把它当"拒答"处理（和 system prompt 里的说法保持一致）
REFUSAL_MARK = "超出了本站课程咨询范围"


# ---------------------------------------------------------------- 缓存
# 测出来：检索不到 1ms，模型调用 500~1000ms。所以瓶颈不在检索，在模型。
# → 优化方向不是让检索更快，是「少调模型」。缓存就是干这个的。

from collections import OrderedDict

CACHE_TTL_S = float(RETR_CFG.get("cache_ttl_s", 300))
CACHE_MAX = int(RETR_CFG.get("cache_max", 256))

_CACHE: "OrderedDict[str, tuple]" = OrderedDict()
_CACHE_STATS = {"hit": 0, "miss": 0}


def _cache_key(question: str, top_k: int) -> str:
    return f"{top_k}|{question}"


def _cache_get(key: str):
    item = _CACHE.get(key)
    if item is None:
        _CACHE_STATS["miss"] += 1
        return None
    ts, val = item
    if time.time() - ts > CACHE_TTL_S:      # 过期 → 当没有
        _CACHE.pop(key, None)
        _CACHE_STATS["miss"] += 1
        return None
    _CACHE.move_to_end(key)                 # 标记为"最近用过"
    _CACHE_STATS["hit"] += 1
    return val


def _cache_put(key: str, val):
    _CACHE[key] = (time.time(), val)
    _CACHE.move_to_end(key)
    while len(_CACHE) > CACHE_MAX:          # 超容量 → 淘汰最久没用的
        _CACHE.popitem(last=False)


# ---------------------------------------------------------------- 生成

def build_user_prompt(question: str, sources: list[Source]) -> str:
    """把问题和检索到的资料拼成一段话，送给模型。"""
    资料块 = []
    for i, s in enumerate(sources, 1):
        正文 = _CHUNK_TEXT.get(s.title, "").strip()
        资料块.append(f"[{i}] 来源：{s.title}\n{正文}")
    return (
        "【知识文档】\n"
        + "\n\n".join(资料块)
        + f"\n\n【用户问题】\n{question}\n\n"
        + "请根据上面的文档回答，并标明来源编号。文档里没有的，直接说不知道。"
    )


def _generate_mock(question: str, sources: list[Source]) -> str:
    """不联网的假生成：只用来验证链路通不通，不作为最终回答。"""
    正文 = _CHUNK_TEXT.get(sources[0].title, "").strip()
    return f"（mock 后端，非真实回答）根据《{sources[0].title}》：{正文[:80]}…"


def _generate_anthropic(system: str, user: str) -> str:
    """调 Anthropic 兼容接口。密钥只从环境变量读，不落盘、不进日志。"""
    env_name = GEN_CFG.get("api_key_env", "ANTHROPIC_API_KEY")
    api_key = os.environ.get(env_name)
    if not api_key:
        raise RuntimeError(f"环境变量 {env_name} 未设置")

    base = str(GEN_CFG.get("base_url", "")).rstrip("/")
    payload = {
        "model": GEN_CFG.get("model"),
        "max_tokens": int(GEN_CFG.get("max_new_tokens", 256)),
        "temperature": float(GEN_CFG.get("temperature", 0.1)),
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    with httpx.Client(timeout=float(GEN_CFG.get("timeout_s", 30))) as client:
        r = client.post(f"{base}/v1/messages", json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()

    # ⚠️ 推理型模型会先吐一个 type="thinking" 的块（思考过程），再吐 type="text" 的块（最终回答）。
    # 只取 text 块，否则会把思考过程当回答返回；而且 max_tokens 太小时可能只输出思考、一个字答案都没有。
    parts = [
        b.get("text", "")
        for b in data.get("content", [])
        if isinstance(b, dict) and b.get("type") == "text"
    ]
    return "".join(parts).strip()


def generate(question: str, sources: list[Source]) -> str:
    """按配置选择后端生成回答。"""
    backend = GEN_CFG.get("backend", "mock")
    if backend == "mock":
        return _generate_mock(question, sources)
    if backend == "anthropic":
        return _generate_anthropic(SYSTEM_PROMPT, build_user_prompt(question, sources))
    raise RuntimeError(f"未知的 backend：{backend}")


def answer_one(req: AnswerRequest) -> AnswerResponse:
    t0 = time.perf_counter()
    key = _cache_key(req.question, req.top_k)

    # ① 先查缓存：命中就直接返回，连检索都省了
    cached = _cache_get(key)
    if cached is not None:
        ans, intent, conf, refused, sources = cached
        return AnswerResponse(
            request_id=req.request_id or f"req_{uuid.uuid4().hex[:8]}",
            answer=ans,
            intent=intent,
            sources=list(sources),
            confidence=conf,
            is_refused=refused,
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )

    # ② 没命中 → 正常走一遍
    sources = retrieve(req.question, req.top_k)

    if not sources:
        ans, intent, conf, refused = "这个问题超出了本站课程咨询范围，我无法回答。", "out_of_scope", 0.0, True
        # ↑ 检索为空 = 直接拒答
    else:
        try:
            ans = generate(req.question, sources)
            if not ans:
                raise RuntimeError("模型返回空内容")
            # 模型自己说"超出范围"时，响应字段也要跟着是 True —— 否则前端看到的和实际内容对不上
            if REFUSAL_MARK and REFUSAL_MARK in ans:
                intent, conf, refused = "out_of_scope", 0.0, True
            else:
                intent, conf, refused = "course_query", 0.5, False
        except Exception:
            # 生成挂了不能返回 500 —— 降级成"暂时答不了"，保住接口可用性
            log.exception("生成失败，降级")
            ans, intent, conf, refused = "生成服务暂时不可用，请稍后再试。", "generation_error", 0.0, True

    # ③ 写缓存。失败结果不缓存 —— 否则一次抖动会被记住 5 分钟
    if intent != "generation_error":
        _cache_put(key, (ans, intent, conf, refused, list(sources)))

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
        "generation_backend": GEN_CFG.get("backend", "mock"),
        "cache": {
            "size": len(_CACHE),
            "hit": _CACHE_STATS["hit"],
            "miss": _CACHE_STATS["miss"],
            "ttl_s": CACHE_TTL_S,
        },
        "framework": FRAMEWORK,
        "version": APP_VERSION,
    }


@app.post("/cache/clear")
def clear_cache():
    """知识库更新后要清缓存，否则用户会在 TTL 内一直拿到旧答案。"""
    n = len(_CACHE)
    _CACHE.clear()
    log.info(f"缓存已清空，原有 {n} 条")
    return {"cleared": n}


@app.get("/config")
def show_config():
    """把「当前真正生效的配置」暴露出来。

    为什么需要这个接口：配置文件改了但没生效，是最难查的一类问题
    （改错了文件 / 缓存 / 拼写错）。看一眼这里就能确认。
    """
    return {
        "model": {
            "embedding_model_name": EMBED_MODEL_NAME,
            "device": MODEL_CFG.get("device"),
            "precision": MODEL_CFG.get("precision"),
            "max_context_tokens": MODEL_CFG.get("max_context_tokens"),
        },
        "retrieval": {
            "sim_threshold": SIM_THRESHOLD,
            "chunk_max": CHUNK_MAX,
            "chunk_overlap": CHUNK_OVERLAP,
            "top_k_default": DEFAULT_TOP_K,
        },
        "generation": {
            "max_new_tokens": GEN_CFG.get("max_new_tokens"),
            "temperature": GEN_CFG.get("temperature"),
            "top_p": GEN_CFG.get("top_p"),
            "system_prompt_chars": len(SYSTEM_PROMPT),
        },
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
