import torch
import torch.nn as nn
import numpy as np

"""
    ======================
    1. 定义多分类模型
    尝试完成一个多分类任务的训练:一个随机向量，哪一维数字最大就属于第几类。
    ======================
"""
class TorchModel(nn.Module):
    def __init__(self, input_size, class_num):
        super(TorchModel, self).__init__()
        # 线性层：输出维度 = 类别数量
        self.linear = nn.Linear(input_size, class_num)
        # 多分类交叉熵损失（自带softmax）
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y_true=None):
        # 前向传播：直接输出原始分数，不要softmax
        logits = self.linear(x)

        # 如果传入标签，就计算损失
        if y_true is not None:
            return self.loss(logits, y_true)
        
        # 推理：返回预测类别
        return torch.argmax(logits, dim=-1)

"""
    ======================
    2. 造数据（随机向量 + 自动打标签）
    ======================
"""
def build_data(input_size, class_num, sample_num=1000):
    # 生成随机向量
    X = torch.randn(sample_num, input_size)
    # 哪一维最大，标签就是几 → 自动生成标签
    Y = torch.argmax(X, dim=-1)
    return X, Y

"""
    ======================
    3. 开始训练
    ======================
"""
if __name__ == "__main__":
    # 配置
    input_size = 5    # 输入5维向量
    class_num = 5     # 5分类
    epochs = 100      # 训练轮数
    lr = 0.1          # 学习率

    # 构建模型、数据、优化器
    model = TorchModel(input_size, class_num)
    X, Y = build_data(input_size, class_num)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    print("开始训练...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()   # 梯度清零
        loss = model(X, Y)      # 前向传播 + 算损失
        loss.backward()         # 反向传播
        optimizer.step()        # 更新参数

        # 每10轮打印一次
        if epoch % 10 == 0:
            pred = model(X)
            acc = (pred == Y).sum().item() / len(Y)
            print(f"轮数：{epoch} | 损失：{loss.item():.4f} | 准确率：{acc:.2%}")

    # 测试一下
    print("\n===== 测试 =====")
    test_x = torch.tensor([[0.1, 0.5, 0.2, 0.8, 0.1]], dtype=torch.float32)
    pred = model(test_x)
    print("输入向量：", test_x.numpy())
    print("最大值位置：", torch.argmax(test_x, dim=-1).item())
    print("模型预测类别：", pred.item())
