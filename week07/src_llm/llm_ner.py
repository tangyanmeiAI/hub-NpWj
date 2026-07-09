"""
使用大模型 API 做 NER：zero-shot vs few-shot 对比
适配数据格式：[{ "tokens": [...], "ner_tags": [...] }] BIO 标注
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time
import random
import argparse
import re
from pathlib import Path
from collections import defaultdict

from openai import OpenAI

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"
LOG_DIR = ROOT / "outputs" / "logs"

ENTITY_TYPE_ZH = {
    "LOC": "地址名称", "ORG": "机构名称", "PER": "人名"
}
ENTITY_TYPES_EN = list(ENTITY_TYPE_ZH.keys())

# ===================== 核心：BIO 解析 =====================
def bio_to_entity_spans(tokens, ner_tags):
    entities = []
    current_text = None
    current_type = None
    start_idx = None

    for idx, (token, tag) in enumerate(zip(tokens, ner_tags)):
        if tag.startswith("B-"):
            if current_text is not None:
                entities.append((current_text, current_type, start_idx, idx - 1))
            current_text = token
            current_type = tag.split("-")[1]
            start_idx = idx
        elif tag.startswith("I-") and current_text is not None:
            current_text += token
        elif tag == "O":
            if current_text is not None:
                entities.append((current_text, current_type, start_idx, idx - 1))
                current_text = None
                current_type = None
                start_idx = None
    if current_text is not None:
        entities.append((current_text, current_type, start_idx, len(tokens) - 1))
    return entities

# ===================== API 客户端 =====================
def build_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise EnvironmentError("请设置环境变量 DASHSCOPE_API_KEY")
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

# ===================== 读取标准 span =====================
def gold_spans_from_record(record: dict) -> set:
    tokens = record["tokens"]
    ner_tags = record["ner_tags"]
    spans = bio_to_entity_spans(tokens, ner_tags)
    return set(spans)

# ===================== 解析模型输出 =====================
def pred_spans_from_response(text: str, response_text: str) -> set:
    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if not json_match:
        return set()
    try:
        obj = json.loads(json_match.group())
    except json.JSONDecodeError:
        return set()

    entities = obj.get("entities", [])
    if not isinstance(entities, list):
        return set()

    spans = set()
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        surface = str(ent.get("text", "")).strip()
        etype = str(ent.get("type", "")).strip()
        if not surface or etype not in ENTITY_TYPES_EN:
            continue
        idx = text.find(surface)
        if idx == -1:
            continue
        spans.add((surface, etype, idx, idx + len(surface) - 1))
    return spans

# ===================== 计算 F1 =====================
def compute_span_f1(all_golds: list[set], all_preds: list[set]) -> dict:
    tp = sum(len(g & p) for g, p in zip(all_golds, all_preds))
    pred_total = sum(len(p) for p in all_preds)
    gold_total = sum(len(g) for g in all_golds)
    p = tp / pred_total if pred_total else 0.0
    r = tp / gold_total if gold_total else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1}

# ===================== Prompt =====================
SYSTEM_PROMPT = """你是一个命名实体识别（NER）专家。
实体类型：
- LOC：地址名称
- ORG：机构名称
- PER：人名

输出严格 JSON，不要多余文字：
{"entities": [{"text": "实体文本", "type": "LOC/ORG/PER"}, ...]}
无实体输出：{"entities": []}
"""

FEW_SHOT_EXAMPLES = [
    {
        "text": "相比之下，青岛海牛队和广州松日队的雨中之战虽然也是0∶0，但乏善可陈。",
        "output": '{"entities": [{"text": "青岛海牛队", "type": "ORG"}, {"text": "广州松日队", "type": "ORG"}]}'
    },
    {
        "text": "我们变而以书会友，以书结缘，把欧美、港台流行的食品类图谱、画册、工具书汇集一堂。",
        "output": '{"entities": [{"text": "欧美", "type": "LOC"}, {"text": "港台", "type": "LOC"}]}'
    },
    {
        "text": "国正学长的文章与诗词，早就读过一些，很是喜欢。",
        "output": '{"entities": [{"text": "国正", "type": "PER"}]}'
    },
]

def zero_shot_prompt(text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]

def few_shot_prompt(text: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": ex["text"]})
        messages.append({"role": "assistant", "content": ex["output"]})
    messages.append({"role": "user", "content": text})
    return messages

# ===================== API 调用 =====================
def call_api(client: OpenAI, messages: list[dict], model: str) -> str:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=0.0, max_tokens=512,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            if attempt < 2:
                time.sleep(1)
            else:
                print(f"API 失败：{e}")
                return ""
    return ""

# ===================== 采样数据 =====================
def sample_records(n: int, seed: int = 42) -> list[dict]:
    with open(DATA_DIR / "validation.json", "r", encoding="utf-8") as f:
        records = json.load(f)
    random.seed(seed)
    if len(records) > n:
        records = random.sample(records, n)
    return records

# ===================== 主函数 =====================
def main():
    args = parse_args()
    client = build_client()
    records = sample_records(args.n_samples)
    print(f"采样 {len(records)} 条验证集")

    zero_shot_golds = []
    zero_shot_preds = []
    few_shot_golds = []
    few_shot_preds = []
    detail_records = []

    for i, record in enumerate(records, 1):
        text = "".join(record["tokens"])
        gold = gold_spans_from_record(record)

        zs_resp = call_api(client, zero_shot_prompt(text), args.model)
        zs_pred = pred_spans_from_response(text, zs_resp)

        fs_resp = call_api(client, few_shot_prompt(text), args.model)
        fs_pred = pred_spans_from_response(text, fs_resp)

        zero_shot_golds.append(gold)
        zero_shot_preds.append(zs_pred)
        few_shot_golds.append(gold)
        few_shot_preds.append(fs_pred)

        detail_records.append({
            "text": text,
            "gold": [{"text": s, "type": t} for s, t, _, _ in gold],
            "zero_shot": [{"text": s, "type": t} for s, t, _, _ in zs_pred],
            "few_shot": [{"text": s, "type": t} for s, t, _, _ in fs_pred],
        })

        if i % 10 == 0 or i == len(records):
            print(f"已处理 {i}/{len(records)}")

    zs_metrics = compute_span_f1(zero_shot_golds, zero_shot_preds)
    fs_metrics = compute_span_f1(few_shot_golds, few_shot_preds)

    print("\n" + "=" * 60)
    print(f"LLM NER 对比结果（模型：{args.model}）")
    print("=" * 60)
    print(f"{'方案':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 52)
    print(f"{'Zero-shot':<20} {zs_metrics['precision']:>10.4f} {zs_metrics['recall']:>10.4f} {zs_metrics['f1']:>10.4f}")
    print(f"{'Few-shot (3例)':<20} {fs_metrics['precision']:>10.4f} {fs_metrics['recall']:>10.4f} {fs_metrics['f1']:>10.4f}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "model": args.model,
        "n_samples": len(records),
        "zero_shot": zs_metrics,
        "few_shot": fs_metrics,
        "detail": detail_records,
    }

    out_path = LOG_DIR / "eval_llm.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存 → {out_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="LLM NER 对比")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--model", type=str, default="qwen-plus")
    return parser.parse_args()

if __name__ == "__main__":
    main()