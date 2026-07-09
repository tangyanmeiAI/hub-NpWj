"""
train_chinese_cls_rnn.py
中文句子关键词分类 —— 简单 RNN 版本

任务：设计一个以文本为输入的多分类任务，实验一下用RNN，LSTM等模型的跑通训练。
    如果不知道怎么设计，可以选择如下任务:对一个任意包含“你”字的五个字的文本，
    “你”在第几位，就属于第几类。
模型：Embedding → RNN → 取最后隐藏状态 → Linear
优化：Adam (lr=1e-3)   损失：CrossEntropyLoss   无需 GPU，CPU 即可运行
"""

# 导入需要的工具包
import random               # 用来随机选句子、打乱数据
import torch                # PyTorch深度学习框架
import torch.nn as nn       # 用来搭建神经网络层
from torch.utils.data import Dataset, DataLoader  # 数据集加载工具

# ─── 超参数：训练前人为设定的参数 ───────────────────────────
SEED = 42                   # 固定随机种子，保证每次运行结果一样
N_SAMPLES = 1000            # 总共生成1000句训练数据
MAXLEN = 5                  # 句子固定长度：5个字（我们任务就是5字句）
EMBED_DIM = 64              # 把每个字变成64维向量
HIDDEN_DIM = 64             # RNN隐藏层神经元数量
LR = 1e-3                   # 学习率：控制模型更新步子大小
BATCH_SIZE = 64             # 一次喂给模型64句话
EPOCHS = 20                 # 把全部数据训练20轮
TRAIN_RATIO = 0.8           # 80%数据用来训练，20%用来验证

# 固定随机种子，让实验可复现
random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 定义5类句子模板：你在第1~5位，对应标签0~4 ───────────
TEMPLATES_POS = [
    # 标签0 → 你在第1位
    ["你温软如初","你满目星河","你奔赴山海","你自成星光","你人间理想","你清风明月","你温柔至极","你岁岁安然","你来日方长","你浅遇深藏"],
    # 标签1 → 你在第2位
    ["唯你是星河","知你意难平","念你渡余生","等你赴流年","望你常安乐","随你踏山河","予你一世安","寻你千万里","懂你半生情","盼你皆顺遂"],
    # 标签2 → 你在第3位
    ["清风你如故","山河你无恙","流年你安好","风月你同行","浮生你相伴","人间你值得","朝夕你相依","繁华你初心","江南你相逢","星辰你入梦"],
    # 标签3 → 你在第4位
    ["山河皆是你","满眼皆是你","岁岁等候你","余生守护你","清风也念你","风月皆思你","人间恰逢你","回首仍念你","此生偏爱你","流年静待你"],
    # 标签4 → 你在第5位
    ["清风只为你","余生只为你","山河等候你","温柔赠予你","人间奔赴你","星河奔赴你","岁月安渡你","流年守护你","风月眷恋你","红尘遇见你"]
]

# 功能：生成数据集，每类句子数量均衡
def build_dataset(n=N_SAMPLES):
    data = []                          # 空列表，用来存（句子，标签）
    num_per_class = n // 5             # 5类，每类多少样本
    for label in range(5):             # 遍历5个类别：0,1,2,3,4
        for _ in range(num_per_class):  # 每类生成固定数量
            sent = random.choice(TEMPLATES_POS[label])  # 随机选一句该类句子
            data.append((sent, label))  # 把（句子，标签）存入数据集
    random.shuffle(data)               # 打乱所有数据顺序
    return data                        # 返回最终数据集

# 功能：构建词表：给每个字分配一个唯一数字ID
def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}    # 预设：填充0，未知字1
    for sent, _ in data:               # 遍历所有句子
        for ch in sent:                # 遍历句子里每个字
            if ch not in vocab:        # 如果字不在词表里
                vocab[ch] = len(vocab) # 给它分配一个新数字
    return vocab                       # 返回字→数字的字典

# 功能：把句子转成数字序列（模型只能看懂数字）
def encode(sent, vocab, maxlen=MAXLEN):
    ids  = [vocab.get(ch, 1) for ch in sent]  # 字→数字，找不到用1代替
    ids  = ids[:maxlen]                       # 句子超过5字就截断
    ids += [0] * (maxlen - len(ids))          # 不足5字补0
    return ids                                # 返回数字列表

# ─── 2. 定义数据集类：让模型能批量读取数据 ─────────────────
class TextDataset(Dataset):
    # 初始化：把所有句子转成数字，标签单独存
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]  # 所有句子→数字
        self.y = [lb for _, lb in data]               # 所有标签

    # 返回数据集总长度
    def __len__(self):
        return len(self.y)

    # 根据索引取一条数据（模型批量读取时用）
    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),  # 数字序列
            torch.tensor(self.y[i], dtype=torch.long),  # 标签
        )

# ─── 3. 定义RNN模型：核心分类模型 ─────────────────────────
class KeywordRNN(nn.Module):
    # 初始化模型层
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()                          # 固定写法
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)  # 字→向量
        self.rnn       = nn.RNN(embed_dim, hidden_dim, batch_first=True)     # RNN层
        self.fc        = nn.Linear(hidden_dim, 5)   # 全连接层：输出5分类结果

    # 前向传播：数据怎么走
    def forward(self, x):
        e = self.embedding(x)              # 输入数字 → 64维向量
        out, _ = self.rnn(e)               # 向量送入RNN，得到每一步输出
        feat = out[:, -1, :]               # 取RNN最后一步隐藏状态（关键）
        logits = self.fc(feat)             # 送入全连接层，得到5分类分数
        return logits                      # 返回原始分数（CrossEntropy会自动算概率）

# ─── 4. 评估函数：计算模型准确率 ───────────────────────────
def evaluate(model, loader):
    model.eval()                          # 切换到评估模式（不更新参数）
    correct = total = 0                   # 正确数、总数归零
    with torch.no_grad():                 # 不计算梯度，省内存、提速
        for X, y in loader:                # 遍历所有测试数据
            logits = model(X)              # 模型预测
            pred    = logits.argmax(dim=1) # 取概率最大的类别作为预测结果
            correct += (pred == y).sum().item()  # 统计正确数量
            total   += len(y)              # 统计总数量
    return correct / total                # 返回准确率

# ─── 5. 训练函数：整个训练流程 ────────────────────────────
def train():
    print("生成数据集...")
    data  = build_dataset(N_SAMPLES)      # 生成1000条数据
    vocab = build_vocab(data)             # 构建词表
    print(f"  样本数：{len(data)}，词表大小：{len(vocab)}")

    split      = int(len(data) * TRAIN_RATIO)  # 按8:2切分数据
    train_data = data[:split]             # 训练集
    val_data   = data[split:]             # 验证集

    # 加载成批量数据
    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TextDataset(val_data,   vocab), batch_size=BATCH_SIZE)

    # 创建模型、损失函数、优化器
    model     = KeywordRNN(vocab_size=len(vocab))
    criterion = nn.CrossEntropyLoss()     # 5分类专用损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # 优化器

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量：{total_params:,}\n")

    # 开始循环训练 EPOCHS 轮
    for epoch in range(1, EPOCHS + 1):
        model.train()                     # 切换到训练模式
        total_loss = 0.0                  # 记录本轮总损失

        # 遍历每一批数据
        for X, y in train_loader:
            pred = model(X)               # 模型前向计算
            loss = criterion(pred, y)     # 计算损失
            optimizer.zero_grad()         # 清空上一轮梯度
            loss.backward()               # 反向传播，算梯度
            optimizer.step()              # 更新模型参数
            total_loss += loss.item()     # 累加损失

        # 计算平均损失和验证集准确率
        avg_loss = total_loss / len(train_loader)
        val_acc  = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    # 训练结束，输出最终准确率
    print(f"\n最终验证准确率：{evaluate(model, val_loader):.4f}")

    # 做5个句子的实际测试
    print("\n--- 推理示例 ---")
    model.eval()
    test_sents = [
        '你温软如初',    # 第1位
        '予你一世安',    # 第2位
        '流年你安好',    # 第3位
        '清风也念你',    # 第4位
        '温柔赠予你',    # 第5位
    ]
    with torch.no_grad():
        for sent in test_sents:
            ids   = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            pred = model(ids).argmax().item()
            print(f"句子：{sent} → 预测「你」在第 {pred+1} 位")

# 程序入口：运行train函数
if __name__ == '__main__':
    train()
