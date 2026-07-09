"""
NER 数据集类：直接使用 BIO 标签 + BERT 子词对齐
适配数据格式：
[{
  "tokens": ["我","爱","北","京"],
  "ner_tags": ["O","O","B-LOC","I-LOC"]
}]
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast #word_ids() 仅 Fast 分词器可用


ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"

# 你要识别的实体类型
ENTITY_TYPES = [
    "LOC", "ORG", "PER"
]


def build_label_schema() -> tuple[list[str], dict[str, int], dict[int, str]]:
    """构建 BIO 标签体系"""
    labels = ["O"]
    for etype in ENTITY_TYPES:
        labels.append(f"B-{etype}")
        labels.append(f"I-{etype}")

    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return labels, label2id, id2label


class NERDataset(Dataset):
    """
    直接适配 BIO 格式 JSON：tokens + ner_tags
    """
    def __init__(
        self,
        records: list,
        tokenizer: BertTokenizerFast,
        label2id: dict,
        max_length: int = 128,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records[idx]
        tokens = row["tokens"]         # 直接用字符列表
        ner_tags = row["ner_tags"]     # 直接用 BIO 标签

        # 标签转 id
        char_labels = [self.label2id[t] for t in ner_tags]

        # BERT 编码（按字符分，自动对齐 word_ids）
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # 子词标签对齐：非第一个子词设为 -100
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        prev_wid = None

        for wid in word_ids:
            if wid is None:
                aligned_labels.append(-100)
            elif wid != prev_wid:
                aligned_labels.append(char_labels[wid] if wid < len(char_labels) else -100)
                prev_wid = wid
            else:
                aligned_labels.append(-100)

        labels_tensor = torch.tensor(aligned_labels, dtype=torch.long)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": labels_tensor,
        }


def load_records(split: str, data_dir: Optional[Path] = None) -> list:
    d = data_dir or DATA_DIR
    with open(d / f"{split}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataloaders(
    tokenizer: BertTokenizerFast,
    label2id: dict,
    batch_size: int = 8,
    max_length: int = 128,
    data_dir: Optional[Path] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:

    train_records = load_records("train", data_dir)
    val_records = load_records("validation", data_dir)
    test_records = load_records("test", data_dir)

    train_ds = NERDataset(train_records, tokenizer, label2id, max_length)
    val_ds = NERDataset(val_records, tokenizer, label2id, max_length)
    test_ds = NERDataset(test_records, tokenizer, label2id, max_length)

    print(f"训练集：{len(train_ds)} | 验证集：{len(val_ds)} | 测试集：{len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
