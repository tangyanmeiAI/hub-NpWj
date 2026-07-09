import torch

L = 4

# 1. 构造下三角 mask
mask = torch.tril(torch.ones(L, L))
print("mask:\n", mask)

# 2. 随机分数
scores = torch.randn(L, L)
print("\n原始 scores:\n", scores)

# 3. 遮罩未来位置 → -inf
scores_masked = scores.masked_fill(mask == 0, float('-inf'))
print("\n遮罩后 scores:\n", scores_masked)


mask = torch.tril(torch.ones(5,5))
type = 'B-PER'
print(type[2:])