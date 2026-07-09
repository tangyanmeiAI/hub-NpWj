# USAGE_GUIDE.md — 代码调用与测试指南

## 一、环境准备

```bash
pip install torch transformers peft>=0.14.0 scikit-learn matplotlib tqdm openai datasets requests

# API key（LLM API 对比脚本需要）
export DASHSCOPE_API_KEY="sk-xxx"
```

本项目使用以下两个预训练模型：

```
pretrain_models/
├── bert-base-chinese/          ← src/ 使用
└── Qwen2-0.5B-Instruct/        ← src_llm/ 使用（SFT 训练和评估）
```

---

## 二、数据准备

数据已下载完毕，无需重复执行。如需重新下载：

```bash
cd src
python download_data.py
```

**预期输出：**
```
下载 AFQMC（clue/afqmc）...
  train       :  34,334 条  正样本  10,573  负样本  23,761
  validation  :   4,316 条  正样本   1,338  负样本   2,978
  test        :   3,861 条  （CLUE 竞赛格式，test 标签未公开，不用于评估）

下载 LCQMC ...    → data/lcqmc/（学生自主练习）
下载 BQ Corpus ... → data/bq_corpus/（学生自主练习）
  train       :  68,960 条  正样本  34,438 (49.9%)  负样本  34,522 (50.1%)
  validation  :   8,620  条  正样本  4,329 (50.2%)  负样本   4,291 (49.8%)
  test        :   8,620  条  正样本  4,382 (50.8%)  负样本   4,238 (49.2%)
---

## 三、数据探索

```bash
cd src
python explore_data.py
```

生成 4 张图到 `outputs/figures/`：

| 图表文件 | 内容 | 教学重点 |
|---------|------|---------|
| `label_distribution.png` | 正/负样本数量 | 类别不均衡（31% vs 69%）|
| `char_length_distribution.png` | 字符长度分布 | max_length=32 已覆盖 98.4% |
| `length_diff_distribution.png` | 正/负样本长度差 | 无 length bias，数据质量好 |
| `token_length_distribution.png` | BERT Token 长度分布 | Token ≈ 字符（中文特性）|

---

## 四、训练 BiEncoder（表示型，重点）

### 4.1 CosineEmbeddingLoss 训练

```bash
cd src
python train_biencoder.py --loss cosine
```

默认参数：`--pool mean --num_hidden_layers 4 --epochs 3 --batch_size 32 --lr 2e-5 --margin 0.3`

**内部流程：**
1. 加载 AFQMC train/val（PairDataset，sentence1 / sentence2 / label）
2. 每个 step：encode(s1) → emb_a，encode(s2) → emb_b，label 0→-1 / 1→+1
3. `F.cosine_embedding_loss(emb_a, emb_b, cos_target, margin=0.3)`
4. 每个 epoch 末：val 集计算余弦相似度 → 枚举 101 个阈值 → 取 F1 最高
5. 保存 val_f1 最优的 checkpoint

**预期输出（每 epoch）：**
```
Epoch 1/3 | train_loss=0.1234 | val_acc=0.7500 val_f1=0.7234 threshold=0.73 | 35s
  ✓ 新最优模型已保存 → .../outputs/checkpoints/biencoder_cosine_best.pt
```


设备: xpu
Loss 类型: cosine  池化策略: mean  BERT 层数: 4  Epochs: 3
BatchSize:32 梯度累积:1 等效批次:32

DataLoader 构建中...
  train :  68,960 条,  2155 batch
  val   :   8,620 条,   270 batch
  test  :   8,620 条,   270 batch  (AFQMC test 无正样本，仅供参考)

构建模型...
模型: BiEncoder (pool=mean, layers=4)
参数量: 45.6M  (BERT 骨干: 45.6M)
总训练步数: 6465  Warmup 步数: 646
Epoch 1/3 | train_loss=0.2841 | val_acc=0.7494 val_f1=0.7494 threshold=0.64 | 878s
  ✓ 新最优模型已保存 → C:\AI\week8 文本匹配问题\文本匹配项目\outputs\checkpoints\biencoder_cosine_best.pt  (val_f1=0.7494)
Epoch 2/3 | train_loss=0.2396 | val_acc=0.7584 val_f1=0.7577 threshold=0.63 | 898s
  ✓ 新最优模型已保存 → C:\AI\week8 文本匹配问题\文本匹配项目\outputs\checkpoints\biencoder_cosine_best.pt  (val_f1=0.7577)
Epoch 3/3 | train_loss=0.2327 | val_acc=0.7628 val_f1=0.7627 threshold=0.66 | 961s
  ✓ 新最优模型已保存 → C:\AI\week8 文本匹配问题\文本匹配项目\outputs\checkpoints\biencoder_cosine_best.pt  (val_f1=0.7627)

训练完成。最优 val_f1=0.7627
训练日志 → C:\AI\week8 文本匹配问题\文本匹配项目\outputs\logs\biencoder_cosine_log.json
最优 checkpoint → C:\AI\week8 文本匹配问题\文本匹配项目\outputs\checkpoints\biencoder_cosine_best.pt

### 4.2 TripletLoss 训练

```bash
python train_biencoder.py --loss triplet --margin 0.3
```
设备: xpu
Loss 类型: triplet  池化策略: mean  BERT 层数: 4  Epochs: 3
BatchSize:32 梯度累积:1 等效批次:32

DataLoader 构建中...
  TripletDataset: 构建 34,438 个三元组
  triplet train :  34,438 三元组,  1077 batch
  val (pair)    :   8,620 对,       270 batch

构建模型...
模型: BiEncoder (pool=mean, layers=4)
参数量: 45.6M  (BERT 骨干: 45.6M)
总训练步数: 3231  Warmup 步数: 323
Epoch 1/3 | train_loss=0.1963 | val_acc=0.7487 val_f1=0.7484 threshold=0.53 | 660s
  ✓ 新最优模型已保存 → C:\AI\week8 文本匹配问题\文本匹配项目\outputs\checkpoints\biencoder_triplet_best.pt  (val_f1=0.7484)
Epoch 2/3 | train_loss=0.1267 | val_acc=0.7720 val_f1=0.7720 threshold=0.56 | 661s
  ✓ 新最优模型已保存 → C:\AI\week8 文本匹配问题\文本匹配项目\outputs\checkpoints\biencoder_triplet_best.pt  (val_f1=0.7720)
Epoch 3/3 | train_loss=0.1138 | val_acc=0.7749 val_f1=0.7749 threshold=0.56 | 651s
  ✓ 新最优模型已保存 → C:\AI\week8 文本匹配问题\文本匹配项目\outputs\checkpoints\biencoder_triplet_best.pt  (val_f1=0.7749)

训练完成。最优 val_f1=0.7749
训练日志 → C:\AI\week8 文本匹配问题\文本匹配项目\outputs\logs\biencoder_triplet_log.json
最优 checkpoint → C:\AI\week8 文本匹配问题\文本匹配项目\outputs\checkpoints\biencoder_triplet_best.pt


**内部流程（与 cosine 的差异）：**
1. 使用 TripletDataset：从 10,573 个正例对构建三元组（anchor, positive, negative）
2. 每个 step：`F.triplet_margin_loss(emb_a, emb_p, emb_n, margin=0.3)`
3. 评估仍用 PairDataset（余弦相似度 + 阈值搜索）

**验证两种 Loss 的差异：** 对比 `outputs/logs/biencoder_cosine_log.json` 与 `biencoder_triplet_log.json` 中的 `val_f1` 曲线。

### 4.3 参数建议（课堂演示 vs 学生练习）

| 场景 | 推荐参数 |
|------|---------|
| 课堂快速演示 | `--num_hidden_layers 4 --epochs 3 --batch_size 32` |
| 学生完整训练 | `--num_hidden_layers 12 --epochs 5 --batch_size 16` |
| 池化策略对比 | `--pool cls` vs `--pool mean` vs `--pool max` |

---

## 五、训练 CrossEncoder（交互型，对比）

```bash
cd src
python train_crossencoder.py
```

默认参数：`--num_hidden_layers 4 --epochs 3 --batch_size 32`

**与 BiEncoder 的关键差异：**
- 输入：`tokenizer(sentence1, sentence2)` 生成 `[CLS] s1 [SEP] s2 [SEP]`
- 评估：直接 `argmax(logits)`，无需阈值搜索
- 训练更慢（max_length=128 vs BiEncoder 的 64）

**预期输出：**
```
Epoch 1/3 | train_loss=0.4812 train_acc=0.7234 | val_acc=0.7681 val_f1=0.7412 | 52s
```


设备: xpu
BERT 层数: 4  Epochs: 3  Batch size: 32

DataLoader 构建中...
  train :  68,960 条,  2155 batch
  val   :   8,620 条,   270 batch
  test  :   8,620 条,   270 batch

构建模型...
模型: CrossEncoder (layers=4)
参数量: 45.6M  (BERT 骨干: 45.6M)
总训练步数: 6465  Warmup 步数: 646
Epoch 1/3 | train_loss=0.5318 train_acc=0.7232 | val_acc=0.8037 val_f1=0.8035 | 868s
  ✓ 新最优模型已保存 → C:\AI\week8 文本匹配问题\文本匹配项目\outputs\checkpoints\crossencoder_best.pt  (val_f1=0.8035)
Epoch 2/3 | train_loss=0.4096 train_acc=0.8139 | val_acc=0.8341 val_f1=0.8341 | 873s
  ✓ 新最优模型已保存 → C:\AI\week8 文本匹配问题\文本匹配项目\outputs\checkpoints\crossencoder_best.pt  (val_f1=0.8341)
Epoch 3/3 | train_loss=0.3585 train_acc=0.8433 | val_acc=0.8477 val_f1=0.8477 | 927s
  ✓ 新最优模型已保存 → C:\AI\week8 文本匹配问题\文本匹配项目\outputs\checkpoints\crossencoder_best.pt  (val_f1=0.8477)

训练完成。最优 val_f1=0.8477
训练日志 → C:\AI\week8 文本匹配问题\文本匹配项目\outputs\logs\crossencoder_log.json
最优 checkpoint → C:\AI\week8 文本匹配问题\文本匹配项目\outputs\checkpoints\crossencoder_best.pt
---

## 六、评估（加载 checkpoint）

```bash
cd src
# BiEncoder
python evaluate.py --model_type biencoder  --ckpt ../outputs/checkpoints/biencoder_cosine_best.pt

              precision    recall  f1-score   support

              不相似       0.77      0.74      0.76      4291
                相似       0.75      0.78      0.77      4329

    accuracy                           0.76      8620
   macro avg       0.76      0.76      0.76      8620
weighted avg       0.76      0.76      0.76      8620

# CrossEncoder
python evaluate.py --model_type crossencoder  --ckpt ../outputs/checkpoints/crossencoder_best.pt


CrossEncoder 评估结果（validation，8620 条）
  Accuracy: 0.8477
  F1      : 0.8477

              precision    recall  f1-score   support

               不相似       0.84      0.86      0.85      4291
                 相似       0.85      0.84      0.85      4329

    accuracy                           0.85      8620
   macro avg       0.85      0.85      0.85      8620
weighted avg       0.85      0.85      0.85      8620

---
**BiEncoder 额外输出：** 相似度分布图 `outputs/figures/biencoder_validation_sim_dist.png`

---

## 七、方法对比（三种训练方式）

```bash
cd src
# 确保三种方法都已训练完（各 1 epoch）后运行
python compare_methods.py
```

**前提：** `outputs/checkpoints/` 下需要有以下三个文件：
- `biencoder_cosine_best.pt`（`python train_biencoder.py --loss cosine`）
- `biencoder_triplet_best.pt`（`python train_biencoder.py --loss triplet`）
- `crossencoder_best.pt`（`python train_crossencoder.py`）

**输出示例（4 层 × 1 epoch）：**
```
方法                              Accuracy  F1(weighted)    额外信息
BiEncoder (CosineEmbeddingLoss)    0.6643        0.6505  threshold=0.55
BiEncoder (TripletLoss)            0.6569        0.6286  threshold=0.84
CrossEncoder (CrossEntropyLoss)    0.6921        0.5703          argmax
```


表格
Epoch	训练 loss	验证 acc	验证 F1	最优阈值	耗时
1	0.2510	0.6654	0.6497	0.57	476s
2	0.2216	0.6659	0.6662	0.51	483s
3	0.2137	0.6722	0.6755	0.51	462s


✅ 识别到 Intel Arc XPU 核显，使用XPU加速
设备: xpu  评估集: validation

=======================================================
加载 biencoder_cosine ...
模型: BiEncoder (pool=mean, layers=4)
参数量: 45.6M  (BERT 骨干: 45.6M)

=======================================================
加载 biencoder_triplet ...
模型: BiEncoder (pool=mean, layers=4)
参数量: 45.6M  (BERT 骨干: 45.6M)

=======================================================
加载 crossencoder ...
模型: CrossEncoder (layers=4)
参数量: 45.6M  (BERT 骨干: 45.6M)

=================================================================
方法                              Accuracy  F1(weighted)  额外信息
-----------------------------------------------------------------
  biencoder_cosine                0.7628        0.7627  threshold=0.66
  biencoder_triplet               0.7749        0.7749  threshold=0.56
  crossencoder                    0.8477        0.8477          argmax

─────────────────────────────────────────────────────────────────
结论速览：
  最高 Accuracy : crossencoder (0.8477)
  最高 F1       : crossencoder  (0.8477)

  Cosine vs Triplet (Δ):
    Accuracy: +0.0122  F1: +0.0123
    → TripletLoss 更优，三元组对语义距离的约束更精确


**几个值得注意的规律：**
- CrossEncoder Accuracy 最高但 F1 最低：1 epoch 训练不足时倾向于预测多数类（负类），
  accuracy 因此虚高——这本身就是一个教学点（accuracy ≠ F1）
- CosineEmbeddingLoss 优于 TripletLoss：AFQMC 正样本只有 10K 条，TripletLoss
  三元组数量受限；数据量更大时（如 LCQMC）Triplet 的优势会更明显
- 生成图表：`method_comparison_bar.png`（柱状对比）+ `biencoder_sim_distributions.png`（分布对比）

---

## 八、Bad Case 分析与优化方向

```bash
cd src
# 分析 BiEncoder（CosineEmbeddingLoss）的错误案例
python analyze_badcases.py

✅ 识别到 Intel Arc XPU 核显，使用XPU加速
加载 checkpoint: C:\AI\week8 文本匹配问题\文本匹配项目\outputs\checkpoints\biencoder_cosine_best.pt
数据集: validation.jsonl  共 8,620 条
模型: BiEncoder (pool=mean, layers=4)
参数量: 45.6M  (BERT 骨干: 45.6M)
整体准确率: 0.7628  错误数: 2045

============================================================
Bad Case 汇总  (共 2045 个错误)
────────────────────────────────────────────────────────────
  FP 假阳性（预测相似，实际不同）: 1098 条
    其中高置信度错误  (Δscore>0.15): 587 条
    其中临界错误     (Δscore≤0.15): 511 条
  FN 假阴性（预测不同，实际相似）:  947 条
    其中高置信度错误  (Δscore>0.15): 445 条
    其中临界错误     (Δscore≤0.15): 502 条

────────────────────────────────────────────────────────────
Bad Case 语言特征分析：

  【FP（假阳性）】共 1098 条
    长度差     : 均值=30.1  中位=4
    s1 长度    : 均值=11.5
    s2 长度    : 均值=37.0
    字符 Jaccard: 均值=0.229  （1=完全重叠，0=无共同字符）

  【FN（假阴性）】共 947 条
    长度差     : 均值=7.0  中位=5
    s1 长度    : 均值=12.4
    s2 长度    : 均值=12.2
    字符 Jaccard: 均值=0.187  （1=完全重叠，0=无共同字符）

============================================================

  FP 高置信度错误（score最高的5条） (展示 5 条)：
    score=0.995  | '可以重新打电话吗？'
                  | '可以主动打电话过去吗'

    score=0.993  | '自己打电话确认可以吗'
                  | '现在可以打电话过来给我吗？'

    score=0.992  | '下午两点打的电话，现在银行还没打电话来确认'
                  | '上午通过qq申请过了，下午没接电话，什么时候重新来电呢'

    score=0.992  | '前面打电话没接到'
                  | '没接到电话'

    score=0.991  | '怎样去取消借款'
                  | '借款申请中如何取消借款'


  FP 临界错误（5条） (展示 5 条)：
    score=0.705  | '为什么绑定银行。说对方银行处理失败'
                  | '为什么总是失败'

    score=0.710  | '我还想看看贷款合同'
                  | '合同是什么'

    score=0.795  | '怎么可以通过审批'
                  | '那怎么才可以获得审批'

    score=0.774  | '为什么我的微信无法借款'
                  | '怎么没开放？'

    score=0.749  | '每天都是身份证输入错误多次'
                  | '我咋借不出来。上次输错身份证后四位。能帮我解开吗'


  FN 高置信度错误（score最低的5条） (展示 5 条)：
    score=-0.118  | 'QQ和微信是一个银行借钱口吗'
                  | '为什么微信上面有微粒贷图标，怎么QQ上面没显示了？'

    score=-0.047  | '什么时候还款综合评估未通过是什么意思'
                  | '8月22日借2万还5个月，第一期什么时候还款'

    score=-0.029  | '可以给我资格么？'
                  | '/微笑，我怎么还是没有微粒贷'

    score=0.000  | '你好，我申请了贷款，什么时候能下来'
                  | '为何还没有入账。'

    score=0.009  | '关注这么久了，还是没成为你们的受邀客户'
                  | '可以给我资格么？'


  FN 临界错误（5条） (展示 5 条)：
    score=0.593  | '可以先第一个月的利息，第二个月一次性还清么？这样利息怎么收？'
                  | '微粒贷怎么计息'

    score=0.620  | '我还款的银行卡注销了'
                  | '我想更换农行卡的，不支持，我换了工商卡，可以了，/呲牙'

    score=0.626  | '为什么我绑不上银行卡'
                  | '总是绑定失败'

    score=0.584  | '可不可以用新卡还款'
                  | '如何换银行卡还款'

    score=0.616  | '我怎么查额度？'
                  | '额度'

  图表已保存 → C:\AI\week8 文本匹配问题\文本匹配项目\outputs\figures\biencoder_badcase_dist.png

============================================================
优化方向建议（基于当前 bad case 分析）
============================================================

【1】数据层面
  ├─ 难负样本增强（Hard Negative Mining）
  │    当前负样本是随机采样，FP 案例中很多是"话题相关但语义不同"的句对。
  │    → 用训练好的 BiEncoder 在大规模数据中挖掘相似度高但标签为 0 的对，
  │      加入训练集，提升负例的区分难度。
  │
  ├─ 数据增强（正样本扩充）
  │    AFQMC 正样本只有 10K 条（31%），TripletLoss 三元组因此受限。
  │    → 对正样本做同义改写（换词、调序），扩充正例数量。
  │      可用 LLM API 批量生成改写句。
  │
  └─ 跨数据集迁移
       LCQMC / BQ Corpus（已下载）包含更多样的问句对。
       → 先在 LCQMC（238K 对）上预训练 Sentence-BERT，再 fine-tune 到 AFQMC，
         利用大数据集的语义泛化能力。

【2】模型层面（FP 字符重叠不高，主要是语义理解不足）
  ├─ 增加 BERT 层数（4 → 8 → 12 层）
  │    浅层 BERT 对语义的建模能力有限，更深层能捕捉更细粒度的语义差异。
  │
  └─ 换用金融领域预训练模型
       FinBERT / MacBERT / RoBERTa-Chinese 在金融/客服语料上有更好的初始化，
       AFQMC 中很多错误源于领域术语理解不准确。

【3】训练策略层面（针对 FN：词汇重叠低但语义相同的同义句）
  ├─ SimCSE 对比学习预训练
  │    同一句话 dropout 两次得到两个正例，大 batch 内其他句子为负例。
  │    这种方式能让模型学到"用不同词说同一个意思"的不变性。
  │
  └─ 调小 TripletLoss 的 margin（0.3 → 0.1）
       如果正例本身语义就不太相似（同义但换了词），过大的 margin 反而
       要求 sim(a,p) 比 sim(a,n) 高出太多，训练信号消失。

【4】评估与部署层面
  ├─ 阈值校准
  │    当前阈值在 val 集上网格搜索，但 val 正负比（31:69）和线上分布可能不同。
  │    → 收集真实线上日志，按实际分布重新校准阈值（Platt scaling 等）。
  │
  ├─ 两阶段级联（最实用的工程改进）
  │    BiEncoder（召回 Top-K）→ CrossEncoder（精排 Top-1）
  │    这是 rag_annual_report Reranker 的完整版。
  │    → 可用当前两个 checkpoint 直接组合，无需重新训练。
  │
  └─ 训练更多 epoch + 全量 12 层
       本次演示用 4 层 × 1 epoch 快速验证，完整训练预计提升 5~10 个 F1 点。
       → 建议学生实验：4 层 vs 12 层，1 epoch vs 5 epoch 的 2×2 消融。

# 分析 CrossEncoder
python analyze_badcases.py --model_type crossencoder \
  --ckpt ../outputs/checkpoints/crossencoder_best.pt

✅ 识别到 Intel Arc XPU 核显，使用XPU加速
加载 checkpoint: ..\outputs\checkpoints\crossencoder_best.pt
数据集: validation.jsonl  共 8,620 条
模型: CrossEncoder (layers=4)
参数量: 45.6M  (BERT 骨干: 45.6M)

整体准确率: 0.8477  错误数: 1313

============================================================
Bad Case 汇总  (共 1313 个错误)
────────────────────────────────────────────────────────────
  FP 假阳性（预测相似，实际不同）:  617 条
    其中高置信度错误  (Δscore>0.15): 408 条
    其中临界错误     (Δscore≤0.15): 209 条
  FN 假阴性（预测不同，实际相似）:  696 条
    其中高置信度错误  (Δscore>0.15): 476 条
    其中临界错误     (Δscore≤0.15): 220 条

────────────────────────────────────────────────────────────
Bad Case 语言特征分析：

  【FP（假阳性）】共 617 条
    长度差     : 均值=5.6  中位=4
    s1 长度    : 均值=12.1
    s2 长度    : 均值=11.7
    字符 Jaccard: 均值=0.269  （1=完全重叠，0=无共同字符）

  【FN（假阴性）】共 696 条
    长度差     : 均值=7.3  中位=5
    s1 长度    : 均值=12.5
    s2 长度    : 均值=13.2
    字符 Jaccard: 均值=0.170  （1=完全重叠，0=无共同字符）

============================================================

  FP 高置信度错误（score最高的5条） (展示 5 条)：
    score=0.997  | '为啥我看不了额度'
                  | '为什么看不到额度'

    score=0.996  | '我的微众银行为何登陆不了'
                  | '银行下载了，怎么不能登陆'

    score=0.991  | '再借款失败'
                  | '我为什么借款失败'

    score=0.987  | '不满足审批要求无法借款是什么意思啊'
                  | '上面显示无法借贷什么意思'

    score=0.987  | '能取消吗'
                  | '电话确认能取消吗'


  FP 临界错误（5条） (展示 5 条)：
    score=0.541  | '我还想看看贷款合同'
                  | '合同是什么'

    score=0.533  | '可以更改还款日日期吗'
                  | '银行卡变更了怎么更改今天还款日了'

    score=0.559  | '我今天有一笔款到期，能否延迟两天'
                  | '延迟一天还款'

    score=0.637  | '为啥评分不足'
                  | '为什么现在说信用不够'

    score=0.521  | '还款银行卡变更了怎么改'
                  | '卡号变更如何还款'


  FN 高置信度错误（score最低的5条） (展示 5 条)：
    score=0.008  | '为什么苹果手机打不开不能使用安卓的手机可以'
                  | 'iPad上微信能申请微粒贷？'

    score=0.009  | '身份验证后多久到账'
                  | '你们最短时间能审批'

    score=0.011  | '未满足要求？哪些要求'
                  | '怎么审批'

    score=0.012  | '可以绑定几张银行卡'
                  | '非绑定向微众卡转帐吗'

    score=0.012  | '什么人可以申请'
                  | '为什么我无法开通'


  FN 临界错误（5条） (展示 5 条)：
    score=0.395  | '可以先第一个月的利息，第二个月一次性还清么？这样利息怎么收？'
                  | '微粒贷怎么计息'

    score=0.386  | '你好，从微众银行转出的金额有限制吗，比如我微众上有30万，可以一次实时转到绑定的银行卡吗？'
                  | '工行购买限额多少'

    score=0.435  | '刚才手机没有信号'
                  | '不好意思，错过了审核电话'

    score=0.385  | '为什么一部分的人有这个微粒贷，一部分人又没有'
                  | "'为啥我不能袋"

    score=0.396  | '刚才我办理了贷款，但点了确实借款就什么提示都没有？'
                  | '为什么我贷'

  图表已保存 → C:\AI\week8 文本匹配问题\文本匹配项目\outputs\figures\crossencoder_badcase_dist.png

============================================================
优化方向建议（基于当前 bad case 分析）
============================================================

【1】数据层面
  ├─ 难负样本增强（Hard Negative Mining）
  │    当前负样本是随机采样，FP 案例中很多是"话题相关但语义不同"的句对。
  │    → 用训练好的 BiEncoder 在大规模数据中挖掘相似度高但标签为 0 的对，
  │      加入训练集，提升负例的区分难度。
  │
  ├─ 数据增强（正样本扩充）
  │    AFQMC 正样本只有 10K 条（31%），TripletLoss 三元组因此受限。
  │    → 对正样本做同义改写（换词、调序），扩充正例数量。
  │      可用 LLM API 批量生成改写句。
  │
  └─ 跨数据集迁移
       LCQMC / BQ Corpus（已下载）包含更多样的问句对。
       → 先在 LCQMC（238K 对）上预训练 Sentence-BERT，再 fine-tune 到 AFQMC，
         利用大数据集的语义泛化能力。

【2】模型层面（FP 字符重叠不高，主要是语义理解不足）
  ├─ 增加 BERT 层数（4 → 8 → 12 层）
  │    浅层 BERT 对语义的建模能力有限，更深层能捕捉更细粒度的语义差异。
  │
  └─ 换用金融领域预训练模型
       FinBERT / MacBERT / RoBERTa-Chinese 在金融/客服语料上有更好的初始化，
       AFQMC 中很多错误源于领域术语理解不准确。

【3】训练策略层面（针对 FN：词汇重叠低但语义相同的同义句）
  ├─ SimCSE 对比学习预训练
  │    同一句话 dropout 两次得到两个正例，大 batch 内其他句子为负例。
  │    这种方式能让模型学到"用不同词说同一个意思"的不变性。
  │
  └─ 调小 TripletLoss 的 margin（0.3 → 0.1）
       如果正例本身语义就不太相似（同义但换了词），过大的 margin 反而
       要求 sim(a,p) 比 sim(a,n) 高出太多，训练信号消失。

【4】评估与部署层面
  ├─ 阈值校准
  │    当前阈值在 val 集上网格搜索，但 val 正负比（31:69）和线上分布可能不同。
  │    → 收集真实线上日志，按实际分布重新校准阈值（Platt scaling 等）。
  │
  ├─ 两阶段级联（最实用的工程改进）
  │    BiEncoder（召回 Top-K）→ CrossEncoder（精排 Top-1）
  │    这是 rag_annual_report Reranker 的完整版。
  │    → 可用当前两个 checkpoint 直接组合，无需重新训练。
  │
  └─ 训练更多 epoch + 全量 12 层
       本次演示用 4 层 × 1 epoch 快速验证，完整训练预计提升 5~10 个 F1 点。
       → 建议学生实验：4 层 vs 12 层，1 epoch vs 5 epoch 的 2×2 消融。


# 展示更多案例
python analyze_badcases.py --n_cases 10
```

**输出内容：**
1. **FP/FN 汇总**：按错误类型和置信度分级（高置信度错误 vs 临界错误）
2. **语言特征分析**：长度差、字符 Jaccard 相似度（揭示错误根因）
3. **典型案例展示**：高置信度错误最具教学价值
4. **优化方向建议**：数据、模型、训练策略、部署四个层面
5. **Score 分布图**：`biencoder_badcase_dist.png`（正确 vs 错误的分数分布）

**关键发现（实测结果）：**
```
FP (假阳性) 567条：字符 Jaccard 均值=0.506  → 词汇高度重叠但语义不同
FN (假阴性) 880条：字符 Jaccard 均值=0.388  → 换了表达方式，词汇重叠低
```
这直接指向两个优化方向：FP → 增大 margin + 难负样本挖掘；FN → SimCSE 对比学习 / 更多层数

---

## 九、LLM zero-shot 对比（API 方式）

```bash
cd src_llm
export DASHSCOPE_API_KEY="sk-xxx"
python llm_compare.py --num_samples 100 --model qwen-plus
```

**说明：**
- 默认只评估 100 条（约消耗 ¥0.1），足够展示效果差异
- 输出包含 Accuracy / F1（正例），以及与 BERT 的对比表
- 结果自动保存到 `outputs/logs/llm_compare_results.json`，供 evaluate_sft.py 读取

---
LLM 评估结果（qwen-plus，100 条样本）
  准确率 (Accuracy)  : 0.7300
  正例精确率         : 0.8000
  正例召回率         : 0.2424
  正例 F1            : 0.3721
  有效预测数         : 100
  解析失败数         : 0




## 十、LLM SFT 指令微调（LoRA / 全量微调）

```bash
cd src_llm

# ── LoRA 微调（默认，推荐演示）──────────────────────────────────────────────
python train_sft.py                        # 5000 条，3 epoch（快速演示）
python train_sft.py --num_train -1         # 全部 34K 条
python train_sft.py --epochs 1             # 1 epoch 快速验证

# ── 全量微调（需显存 ≥ 16GB）────────────────────────────────────────────────
python train_sft.py --full_ft --lr 2e-5
```

**完整参数说明**：

```bash
python train_sft.py \
  --num_train 5000    \  # 训练样本数，-1 用全部 34K 条
  --epochs 3          \  # 训练轮数
  --batch_size 4      \  # 每步 batch 大小
  --grad_accum 4      \  # 梯度累积，等效 batch = 16
  --max_length 128    \  # 句对 + 模板 < 128，无需 256
  --lora_r 8          \  # LoRA rank（仅 LoRA 模式有效）
  --full_ft              # 切换为全量微调（默认 LoRA）
```

**两种模式对比**：

| 维度 | LoRA（默认）| 全量微调（`--full_ft`）|
|------|------------|----------------------|
| 可训练参数 | ~1.08M（0.22%）| 495M（100%）|
| 默认学习率 | 2e-4（自动）| 2e-5（需手动指定）|
| checkpoint 目录 | `outputs/sft_adapter/` | `outputs/sft_full_ckpt/` |
| 日志文件 | `outputs/logs/train_sft.json` | `outputs/logs/train_full_ft.json` |

> **类别平衡说明**：AFQMC 正负比 31:69。`train_sft.py` 默认开启**正负平衡采样**（各取 `num_train//2` 条），避免模型退化为全预测负例（F1=0 的教学反例在第一版实测中出现过，与 CrossEncoder 1-epoch 是同一问题）。

---

## 十一、SFT 模型评估

```bash
cd src_llm

python evaluate_sft.py                                     # 评估 LoRA（默认，200 条）
python evaluate_sft.py --ckpt_dir ../outputs/sft_full_ckpt  # 评估全量微调
python evaluate_sft.py --demo                              # 5 条快速演示
python evaluate_sft.py --num_samples 500                   # 更多样本
```

`evaluate_sft.py` 自动识别 checkpoint 类型，并读取 `llm_compare_results.json` 和 BERT 训练日志做多方对比。

**实测输出**（LoRA，平衡采样 5000 条，1 epoch，200 条评估，seed=42）：

```
样本数      : 200（有效: 200，parse_fail: 0）
Accuracy    : 0.6400
F1(weighted): 0.6535
F1(正例)    : 0.5556
均值耗时    : 0.10s/条（GPU）

多方对比（AFQMC validation 集，所有方案均使用 Accuracy + F1，直接可比）
  BiEncoder + CosineEmbeddingLoss    F1(pos) = 0.6505
  BiEncoder + TripletLoss            F1(pos) = 0.6286
  CrossEncoder + CrossEntropyLoss    F1(pos) = 0.5703
  Qwen API zero-shot                 F1(pos) = ?（运行 llm_compare.py 后获取）
  Qwen2-0.5B SFT（LoRA，5K 平衡）   F1(pos) = 0.5556
```

> SFT 与 BERT 方法使用完全相同的 Accuracy + F1 指标，无评估标准差异，数字可直接比较。

---

## 十、作为模块调用

```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import BertTokenizer
from model import build_biencoder

# 初始化
BERT_PATH = "E:/badou/项目材料准备/pretrain_models/bert-base-chinese"
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

# 加载训练好的模型
ckpt  = torch.load("outputs/checkpoints/biencoder_cosine_best.pt",
                   map_location=device, weights_only=False)
model = build_biencoder(BERT_PATH, pool="mean", num_hidden_layers=4).to(device)
model.load_state_dict(ckpt["state_dict"])
model.eval()

# 单次推理
def encode(text):
    enc = tokenizer(text, max_length=64, truncation=True,
                    padding="max_length", return_tensors="pt")
    return model.encode(
        enc["input_ids"].to(device),
        enc["attention_mask"].to(device),
        enc["token_type_ids"].to(device),
    )

s1 = "花呗怎么还款"
s2 = "如何偿还花呗账单"
emb1 = encode(s1)
emb2 = encode(s2)
sim  = F.cosine_similarity(emb1, emb2).item()
threshold = ckpt["threshold"]  # 训练时在 val 集搜出的最优阈值
print(f"相似度: {sim:.4f}  阈值: {threshold:.2f}  预测: {'相似' if sim >= threshold else '不相似'}")
```

---

## 十一、调试与常见问题

**Q: `OMP: Error #15: Initializing libiomp5md.dll`**
> 已在所有脚本顶部加 `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` 修复

**Q: `OSError: Repo id must use alphanumeric chars`**
> transformers 将相对路径当成 HuggingFace repo ID 验证。已用 `Path(__file__).parent.parent` 构造绝对路径解决

**Q: BiEncoder 评估时 val_f1 很低（< 0.6）**
> 可能原因：(1) 学习率偏高（试 `--lr 1e-5`）；(2) epoch 不够（试 `--epochs 5`）；(3) margin 过大（试 `--margin 0.1`）

**Q: AFQMC test 集评估结果异常（全预测为 0）**
> 正常现象——test 集标签在 CLUE 竞赛中未公开，label=-1。评估请用 `--split validation`

**Q: TripletLoss 训练 loss 不下降**
> AFQMC 三元组约 1 万条（正例数量限制），数据量较小。可调小 `--margin 0.1` 或改用全量 12 层

**Q: LLM 对比脚本报 `DASHSCOPE_API_KEY` 未设置**
> 运行 `export DASHSCOPE_API_KEY="sk-xxx"` 后再执行

**Q: train_sft.py 报 `ModuleNotFoundError: No module named 'peft'`**
> LoRA 微调需要 peft 库：`pip install peft>=0.14.0`。全量微调（`--full_ft`）不需要。

**Q: evaluate_sft.py 报 checkpoint 目录不存在**
> 需先运行 `train_sft.py` 完成训练。LoRA 保存到 `outputs/sft_adapter/`，全量保存到 `outputs/sft_full_ckpt/`。

**Q: SFT 的 parse_fail 率较高**
> 说明训练不足。文本匹配的 TARGET 只有 3~5 token，一般训练 1 epoch 就能稳定输出。检查 SYSTEM_PROMPT 是否与 LABEL_MAP 一致（均使用【相似】/【不相似】）。
