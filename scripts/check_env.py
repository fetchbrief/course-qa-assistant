"""环境自检：跑一遍就知道环境对不对。评分系统也查这个文件在不在。"""
import json, sys, platform
from pathlib import Path

out = {
    "python": sys.version.split()[0],
    "platform": platform.system(),
    "checks": {},
}
try:
    import fastapi; out["checks"]["fastapi"] = fastapi.__version__
except Exception as e: out["checks"]["fastapi"] = f"FAIL: {e}"
try:
    import uvicorn; out["checks"]["uvicorn"] = uvicorn.__version__
except Exception as e: out["checks"]["uvicorn"] = f"FAIL: {e}"

root = Path(__file__).resolve().parent.parent
out["checks"]["knowledge_dir"] = (root / "data" / "knowledge").exists()

Path("env_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2))
