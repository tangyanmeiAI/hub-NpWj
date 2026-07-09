"""
LLM SFT（监督微调）训练脚本 — 基于 LoRA 高效微调 Qwen2-0.5B-Instruct 做 NER
适配数据格式：[{ "tokens": [...], "ner_tags": [...] }] BIO 标注
"""

import os
import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# ====================== 配置 ======================
ROOT       = Path(__file__).parent
DATA_DIR   = ROOT / "data" / "peoples_daily"
MODEL_PATH = "D:/AI/pretrain_models/Qwen2-0.5B-Instruct"
OUTPUT_DIR = ROOT / "outputs"

ENTITY_MAP = {
    "LOC": "地址",
    "ORG": "机构",
    "PER": "人名"
}

SYSTEM_PROMPT = (
    "你是一个命名实体识别助手。从文本中识别命名实体，以 JSON 格式输出。\n"
    "实体类型：LOC（地址）、ORG（机构）、PER（人名）\n"
    '输出格式：{"entities": [{"text": "实体文本", "type": "实体类型"}]}\n'
    '无实体输出：{"entities": []}'
)

# ================ 核心：适配 BIO 格式 ================
def bio_to_entities(tokens, ner_tags):
    entities = []
    current_entity = None
    current_type = None

    for idx, (token, tag) in enumerate(zip(tokens, ner_tags)):
        if tag.startswith("B-"):
            if current_entity is not None:
                entities.append({"text": current_entity, "type": current_type})
            current_entity = token
            current_type = tag.split("-")[1]
        elif tag.startswith("I-") and current_entity is not None:
            current_entity += token
        elif tag == "O":
            if current_entity is not None:
                entities.append({"text": current_entity, "type": current_type})
                current_entity = None
                current_type = None

    if current_entity is not None:
        entities.append({"text": current_entity, "type": current_type})

    return entities

def bio_record_to_target(record):
    tokens = record["tokens"]
    ner_tags = record["ner_tags"]
    entities = bio_to_entities(tokens, ner_tags)
    return json.dumps({"entities": entities}, ensure_ascii=False)

# ================ 数据集 =================
class SFTDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = "".join(item["tokens"])
        target = bio_record_to_target(item)

        prompt_text = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        response_ids = self.tokenizer.encode(target, add_special_tokens=False) + [self.tokenizer.eos_token_id]

        input_ids = (prompt_ids + response_ids)[: self.max_length]
        labels = ([-100] * len(prompt_ids) + response_ids)[: self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def collate_fn(batch, pad_id):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids_list, labels_list, mask_list = [], [], []
    for item in batch:
        n = len(item["input_ids"])
        pad = max_len - n
        input_ids_list.append(torch.cat([item["input_ids"], torch.full((pad,), pad_id, dtype=torch.long)]))
        labels_list.append(torch.cat([item["labels"], torch.full((pad,), -100, dtype=torch.long)]))
        mask_list.append(torch.cat([torch.ones(n, dtype=torch.long), torch.zeros(pad, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(input_ids_list),
        "labels": torch.stack(labels_list),
        "attention_mask": torch.stack(mask_list),
    }

# ================ 训练主逻辑 =================
def parse_args():
    parser = argparse.ArgumentParser(description="LLM SFT NER 训练（适配 BIO 格式）")
    parser.add_argument("--model_path", default=str(MODEL_PATH))
    parser.add_argument("--data_dir", default=str(DATA_DIR))
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--num_train",   default=-1,   type=int,
                        help="训练样本数，-1 使用全部 10748 条（默认）")
    parser.add_argument("--epochs",      default=3,    type=int)
    parser.add_argument("--batch_size",  default=4,    type=int)
    parser.add_argument("--grad_accum",  default=4,    type=int)
    parser.add_argument("--lr",          default=None, type=float,
                        help="学习率；默认 LoRA=2e-4，全量=2e-5（自动判断）")
    parser.add_argument("--max_length",  default=256,  type=int,
                        help="序列最大长度；NER 的 JSON 输出比分类长，建议 256")
    # 全量微调开关
    parser.add_argument("--full_ft",     action="store_true",
                        help="全量微调：跳过 LoRA，更新所有 495M 参数（需显存 ≥ 16GB）")
    # LoRA 超参（full_ft 时忽略）
    parser.add_argument("--lora_r",      default=8,    type=int)
    parser.add_argument("--lora_alpha",  default=16,   type=int)
    parser.add_argument("--seed",        default=42,   type=int)
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.lr is None:
        args.lr = 2e-5 if args.full_ft else 2e-4

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / ("sft_full_ckpt" if args.full_ft else "sft_adapter")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode_str = "全量微调" if args.full_ft else "LoRA 微调"
    print(f"设备: {device} | 模式: {mode_str}")

    # ========== 加载 BIO 格式数据 ==========
    with open(data_dir / "train.json", encoding="utf-8") as f:
        train_raw = json.load(f)
    with open(data_dir / "validation.json", encoding="utf-8") as f:
        val_raw = json.load(f)

    if args.num_train > 0:
        train_raw = random.sample(train_raw, min(args.num_train, len(train_raw)))
    print(f"训练集: {len(train_raw)} | 验证集: {len(val_raw[:300])}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    train_dataset = SFTDataset(train_raw, tokenizer, args.max_length)
    val_dataset = SFTDataset(val_raw[:300], tokenizer, args.max_length)
    collate = lambda b: collate_fn(b, tokenizer.pad_token_id)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size*2, shuffle=False, collate_fn=collate)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if not args.full_ft:
        if not PEFT_AVAILABLE:
            raise Exception("请安装 peft：pip install peft")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        for p in model.parameters():
            p.requires_grad_(True)

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best_val = float("inf")
    logs = []

    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        optimizer.zero_grad()
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} train")
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            loss = out.loss
            ntok = (labels != -100).sum().item()
            total_loss += loss.item() * ntok
            total_tokens += ntok
            (loss / args.grad_accum).backward()

            if (step+1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        train_loss = total_loss / max(total_tokens, 1)

        model.eval()
        val_loss, val_tok = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                ntok = (labels != -100).sum().item()
                val_loss += out.loss.item() * ntok
                val_tok += ntok
        val_loss = val_loss / max(val_tok, 1)

        elapsed = time.time() - t0
        print(f"Epoch {epoch} | train={train_loss:.4f} val={val_loss:.4f} | {elapsed:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"→ 最优模型保存: {ckpt_dir}")

        logs.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "time": elapsed})

    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    with open(log_dir / "train_sft.json", "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)

    print("\n训练完成！")
    print(f"最优 val loss: {best_val:.4f}")
    print(f"模型: {ckpt_dir}")

if __name__ == "__main__":
    main()
