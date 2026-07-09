"""
加载 SFT checkpoint（LoRA / 全量微调），在验证集上评估 NER entity-level F1，
与 BERT+CRF 和 LLM API（zero/few-shot）多方对比

适配数据格式：[{ "tokens": [...], "ner_tags": [...] }]
"""

import os
import argparse
import json
import random
import re
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / "data" / "peoples_daily"
MODEL_PATH  = "D:/AI/pretrain_models/Qwen2-0.5B-Instruct"
ADAPTER_DIR = ROOT / "outputs" / "sft_adapter"
LOG_DIR     = ROOT / "outputs" / "logs"

# 只保留你数据里的 3 类
ENTITY_TYPES = ["LOC", "ORG", "PER"]

SYSTEM_PROMPT = (
    "你是一个命名实体识别助手。从文本中识别命名实体，以 JSON 格式输出。\n"
    "实体类型（英文标识）：LOC（地址名称）、ORG（机构名称）、PER（人名）\n"
    '输出格式（严格遵守，不输出其他内容）：{"entities": [{"text": "实体文本", "type": "实体类型"}]}\n'
    '无实体时输出：{"entities": []}'
)

# ══════════════════════════════════════════════════════════════════════════════
# 核心：BIO 解析（和 llm_ner.py / train_sft.py 完全一致）
# ══════════════════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════════════════
# 模型加载
# ══════════════════════════════════════════════════════════════════════════════
def load_model(model_path: str, ckpt_dir: str, device: torch.device):
    ckpt_path = Path(ckpt_dir)
    is_lora   = (ckpt_path / "adapter_config.json").exists()

    if is_lora:
        if not PEFT_AVAILABLE:
            raise ImportError("加载 LoRA adapter 需要 peft 库：pip install peft>=0.14.0")
        print(f"检测到 LoRA adapter，加载 base model: {model_path}")
        tokenizer  = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, str(ckpt_path))
        model = model.merge_and_unload()
    else:
        print(f"加载全量微调模型: {ckpt_dir}")
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        )

    model.to(device).eval()
    print(f"模型加载完成！\n")
    return model, tokenizer

# ══════════════════════════════════════════════════════════════════════════════
# 推理与解析
# ══════════════════════════════════════════════════════════════════════════════
def generate_ner(text: str, model, tokenizer, device: torch.device, max_new_tokens=256):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": text},
    ]
    encoding = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    prompt_len = input_ids.shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    new_tokens = output_ids[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def gold_spans_from_record(record: dict) -> set:
    tokens = record["tokens"]
    ner_tags = record["ner_tags"]
    spans = bio_to_entity_spans(tokens, ner_tags)
    return set(spans)

def pred_spans_from_output(text: str, raw_output: str) -> set:
    json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if not json_match:
        return set()
    try:
        obj = json.loads(json_match.group())
    except json.JSONDecodeError:
        return set()

    entities = obj.get("entities", [])
    spans = set()
    for ent in entities:
        surface = str(ent.get("text", "")).strip()
        etype = str(ent.get("type", "")).strip()
        if not surface or etype not in ENTITY_TYPES:
            continue
        idx = text.find(surface)
        if idx == -1:
            continue
        spans.add((surface, etype, idx, idx + len(surface) - 1))
    return spans

def compute_span_f1(all_golds: list[set], all_preds: list[set]) -> dict:
    tp = sum(len(g & p) for g, p in zip(all_golds, all_preds))
    pred_total = sum(len(p) for p in all_preds)
    gold_total = sum(len(g) for g in all_golds)
    p = tp / pred_total if pred_total else 0.0
    r = tp / gold_total if gold_total else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1}

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(description="LLM SFT NER 评估（BIO 格式）")
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--ckpt_dir", default=str(ADAPTER_DIR))
    parser.add_argument("--data_dir", default=str(DATA_DIR))
    parser.add_argument("--n_samples", default=100, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--demo", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载 BIO 格式验证集
    with open(Path(args.data_dir) / "validation.json", encoding="utf-8") as f:
        val_data = json.load(f)

    random.seed(args.seed)
    n = 5 if args.demo else args.n_samples
    samples = random.sample(val_data, min(n, len(val_data)))
    print(f"评估样本数: {len(samples)}\n")

    model, tokenizer = load_model(args.model_path, args.ckpt_dir, device)

    all_golds, all_preds = [], []
    detail_records = []
    parse_fail = 0
    t0 = time.time()

    for i, record in enumerate(samples, 1):
        text = "".join(record["tokens"])
        g_set = gold_spans_from_record(record)
        raw = generate_ner(text, model, tokenizer, device)
        p_set = pred_spans_from_output(text, raw)

        if not re.search(r"\{.*entities.*\}", raw, re.DOTALL):
            parse_fail += 1

        all_golds.append(g_set)
        all_preds.append(p_set)
        detail_records.append({
            "text": text,
            "gold": [{"text": s, "type": t} for s, t, *_ in g_set],
            "pred": [{"text": s, "type": t} for s, t, *_ in p_set],
            "raw_output": raw,
        })

    metrics = compute_span_f1(all_golds, all_preds)
    elapsed = time.time() - t0

    print(f"\n" + "="*50)
    print(f"📊 SFT 模型评估结果")
    print(f"="*50)
    print(f"样本数      : {len(samples)}")
    print(f"Precision   : {metrics['precision']:.4f}")
    print(f"Recall      : {metrics['recall']:.4f}")
    print(f"F1          : {metrics['f1']:.4f}")
    print(f"JSON 失败   : {parse_fail} 条")
    print(f"耗时        : {elapsed:.1f}s\n")

    out_path = LOG_DIR / "eval_sft.json"
    LOG_DIR.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "detail": detail_records}, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()