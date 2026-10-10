# -*- coding: utf-8 -*-
"""多模型中医安全问答对比跑批（零依赖，仅标准库）。

输出 out/answers_<时间戳>.jsonl：
  第一行 meta，记录本次跑批的全部条件（时间／模型／参数／题量），
  之后每行一次调用，含原始回答全文、耗时、token 用量。

用法：
  python run_compare.py --list
  python run_compare.py --dry-run
  python run_compare.py --providers deepseek --limit 2
  python run_compare.py
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

BASE = pathlib.Path(__file__).resolve().parent
QUESTIONS = BASE / "questions_paired.jsonl"
OUTDIR = BASE / "out"

# model 只是占位默认值：各平台改名频繁，请以你控制台里实际显示的模型名为准。
PROVIDERS: dict[str, dict[str, str]] = {
    "deepseek": {"base_url": "https://api.deepseek.com/v1", "env": "DEEPSEEK_API_KEY", "model": "deepseek-chat"},
    "zhipu": {"base_url": "https://open.bigmodel.cn/api/paas/v4", "env": "ZHIPU_API_KEY", "model": "glm-4.6"},
    "moonshot": {"base_url": "https://api.moonshot.cn/v1", "env": "MOONSHOT_API_KEY", "model": "kimi-k2-0905-preview"},
    "dashscope": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "env": "DASHSCOPE_API_KEY", "model": "qwen-plus"},
    "ark": {"base_url": "https://ark.cn-beijing.volces.com/api/v3", "env": "ARK_API_KEY", "model": "填你的推理接入点ID"},
}

# 固定参数：温度 0 保证可复现。推理型模型不接受 temperature，脚本会自动剔除。
PARAMS: dict[str, object] = {"temperature": 0, "max_tokens": 1500, "stream": False}
TIMEOUT = 180


def load_questions() -> list[dict]:
    items: list[dict] = []
    with QUESTIONS.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def call_api(cfg: dict, question: str) -> tuple[dict, float]:
    key = os.environ.get(cfg["env"], "").strip()
    if not key:
        raise RuntimeError(f"环境变量 {cfg['env']} 未设置")

    model = cfg["model"]
    params = dict(PARAMS)
    if "reasoner" in model.lower() or "thinking" in model.lower():
        params.pop("temperature", None)

    payload = {"model": model, "messages": [{"role": "user", "content": question}], **params}
    req = urllib.request.Request(
        cfg["base_url"].rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )

    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    elapsed = time.perf_counter() - started

    choice = (raw.get("choices") or [{}])[0]
    text = (choice.get("message") or {}).get("content") or ""
    return {
        "returned_model": raw.get("model", ""),
        "answer": text,
        "finish_reason": choice.get("finish_reason", ""),
        "usage": raw.get("usage", {}),
        "raw_ok": True,
    }, elapsed


def main() -> int:
    ap = argparse.ArgumentParser(description="多模型对比跑批")
    ap.add_argument("--providers", help="逗号分隔，如 deepseek,zhipu；默认全部已配 key 的")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 道题")
    ap.add_argument("--dry-run", action="store_true", help="不联网，只打印将发送的内容")
    ap.add_argument("--list", action="store_true", help="列出 provider 与 key 状态后退出")
    args = ap.parse_args()

    questions = load_questions()
    if args.limit:
        questions = questions[: args.limit]

    if args.providers:
        names = [n.strip() for n in args.providers.split(",") if n.strip()]
    else:
        names = list(PROVIDERS)

    ready = {n: bool(os.environ.get(PROVIDERS[n]["env"], "").strip()) for n in names}

    if args.list:
        groups = len({q["group"] for q in questions})
        print(f"题库：{len(questions)} 道（{groups} 组配对）")
        for n in names:
            mark = "OK " if ready[n] else "缺 key"
            cfg = PROVIDERS[n]
            print(f"  [{mark}] {n:10s} model={cfg['model']:32s} env={cfg['env']}")
        return 0

    use = [n for n in names if ready[n]]
    if not use:
        print("没有可用的 provider（key 都没配）。先用 --list 看需要哪些环境变量。", file=sys.stderr)
        return 2

    if args.dry_run:
        for n in use:
            cfg = PROVIDERS[n]
            print(f"[{n}] POST {cfg['base_url'].rstrip('/')}/chat/completions  model={cfg['model']}")
        print(f"将发送 {len(use)} 模型 × {len(questions)} 题 = {len(use) * len(questions)} 次调用")
        print("示例题：", questions[0]["qid"], questions[0]["question"][:60], "...")
        return 0

    OUTDIR.mkdir(exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTDIR / f"answers_{stamp}.jsonl"

    meta = {
        "kind": "meta",
        "run_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "providers": {n: PROVIDERS[n]["model"] for n in use},
        "params": PARAMS,
        "question_file": QUESTIONS.name,
        "question_count": len(questions),
        "note": "A/B 版各自独立请求，无上下文串扰；temperature 固定 0。",
    }

    total = len(use) * len(questions)
    done = 0
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
        fh.flush()
        for q in questions:
            for n in use:
                done += 1
                prefix = f"[{done}/{total}] {n} {q['qid']}"
                base_row = {
                    "kind": "answer",
                    "provider": n,
                    "sent_model": PROVIDERS[n]["model"],
                    "qid": q["qid"],
                    "group": q["group"],
                    "variant": q["variant"],
                    "must_have": q.get("must_have", []),
                    "question": q["question"],
                }
                try:
                    result, elapsed = call_api(PROVIDERS[n], q["question"])
                    base_row.update(result)
                    base_row["latency_s"] = round(elapsed, 2)
                    print(f"{prefix}  完成 {elapsed:.1f}s  {len(result['answer'])} 字")
                except urllib.error.HTTPError as exc:
                    body = exc.read().decode("utf-8", "replace")[:400]
                    base_row.update({"raw_ok": False, "error": f"HTTP {exc.code}: {body}"})
                    print(f"{prefix}  失败 HTTP {exc.code}")
                except Exception as exc:  # noqa: BLE001
                    base_row.update({"raw_ok": False, "error": repr(exc)})
                    print(f"{prefix}  失败 {exc!r}")
                fh.write(json.dumps(base_row, ensure_ascii=False) + "\n")
                fh.flush()

    print(f"\n已写入：{out_path}")
    print("下一步：打开 README 第三节的判读表，逐题记必检项命中与错误类型。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
