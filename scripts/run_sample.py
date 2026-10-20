"""
跑官方样题的 5 道公开例题，看能拿几分。

跑法：
    cd C:/Users/admin/Desktop/ai/lanqiao
    .venv/Scripts/python.exe scripts/run_sample.py

评分规则（官方原文）：
    关键事实准确   5 分
    数值与条件完整 2 分
    来源文档命中   2 分
    每题满分 9 分，5 题满分 45 分

⚠️ 注意：官方样题用的是它自己的知识库（refund_policy_002 / course_python_101 /
   certificate_policy_001），我们的知识库是自己写的，内容不完全对得上。
   所以「分数低」不等于「代码差」，要分开看。
"""

import os
import sys

os.environ["NO_PROXY"] = ",".join(
    x for x in os.environ.get("NO_PROXY", "127.0.0.1,localhost").replace("[::1]", "").split(",") if x
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module

BACKEND = os.environ.get("SAMPLE_BACKEND", "mock")

# 官方样题附录里的公开例题
SAMPLES = [
    {
        "q": "报名后还没开课，7 天内可以退吗？",
        "facts": ["7 日内", "尚未开课", "可申请全额退款"],
        "src": "refund_policy_002",
    },
    {
        "q": "Python 基础班周几直播？",
        "facts": ["每周二、周四", "19:30"],
        "src": "course_python_101",
    },
    {
        "q": "学完以后结业证书怎么拿？",
        "facts": ["完成度不低于 90%", "测试不低于 80 分"],
        "src": "certificate_policy_001",
    },
    {
        "q": "退废申请入口在哪？",   # 故意的错别字「退废」
        "facts": ["识别为退费", "学习中心入口"],
        "src": "refund_policy_002",
    },
    {
        "q": "你们学校食堂几点关门？",
        "facts": ["应拒答", "不返回来源"],
        "src": None,
    },
]


def main():
    app_module.GEN_CFG["backend"] = BACKEND
    app_module._CACHE.clear()

    print(f"生成后端：{BACKEND}")
    print(f"知识库：{len(app_module._chunks)} 段，来自 {len(app_module.KNOWLEDGE)} 篇文档")
    print("=" * 78)

    results = []
    for i, s in enumerate(SAMPLES, 1):
        r = app_module.answer_one(app_module.AnswerRequest(question=s["q"]))
        results.append((s, r))

        print(f"\n【第 {i} 题】{s['q']}")
        print(f"  应含事实：{' / '.join(s['facts'])}")
        print(f"  标准来源：{s['src'] or '（应拒答）'}")
        print(f"  ── 系统回答 ──")
        print(f"  {r.answer}")
        print(f"  拒答：{r.is_refused}　来源：{[x.id for x in r.sources]}")

    # 简单自动比对
    print("\n" + "=" * 78)
    print("自动比对结果（只查关键字和来源 id，不等于人工评分）")
    print("=" * 78)
    total = 0
    for s, r in results:
        hits = [f for f in s["facts"] if f.replace(" ", "") in r.answer.replace(" ", "")]
        fact_ok = len(hits) == len(s["facts"])

        if s["src"] is None:
            src_ok = r.is_refused and not r.sources
        else:
            src_ok = any(x.id == s["src"] for x in r.sources)

        score = (5 if fact_ok else 0) + (2 if fact_ok and len(hits) >= 2 else 0) + (2 if src_ok else 0)
        total += score
        print(f"  {s['q'][:18]:<20} 事实 {len(hits)}/{len(s['facts'])}　来源 {'OK' if src_ok else 'NG'}　估分 {score}/9")
        if not fact_ok:
            print(f"      缺：{[f for f in s['facts'] if f not in hits]}")

    print(f"\n  合计估分：{total}/45")


if __name__ == "__main__":
    main()
