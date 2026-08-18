import torch
import matplotlib.pyplot as plt

# 1. 定义 4 个转换规则的参数 (a, b, c, d, e, f)
rules = torch.tensor([
    [0.00, 0.00, 0.00, 0.16, 0.00, 0.00],  # 规则 1: 茎干 (1%)
    [0.85, 0.04, -0.04, 0.85, 0.00, 1.60],  # 规则 2: 主叶 (85%)
    [0.20, -0.26, 0.23, 0.22, 0.00, 1.60],  # 规则 3: 左叶 (7%)
    [-0.15, 0.28, 0.26, 0.24, 0.00, 0.44]  # 规则 4: 右叶 (7%)
])

# 定义每个规则被选中的概率
probabilities = torch.tensor([0.01, 0.85, 0.07, 0.07])

# 2. 发挥 PyTorch 的张量魔法：一次性初始化 200,000 个点
num_points = 200000
points = torch.zeros((num_points, 2))

# 3. 迭代计算
# 迭代 50 次足以让所有的点都被“吸”入到分形的轨迹上
for _ in range(50):
    # 掷骰子：同时为 20 万个点抽取规则
    rule_indices = torch.multinomial(probabilities, num_samples=num_points, replacement=True)

    # 提取对应的参数
    selected_rules = rules[rule_indices]
    a = selected_rules[:, 0]
    b = selected_rules[:, 1]
    c = selected_rules[:, 2]
    d = selected_rules[:, 3]
    e = selected_rules[:, 4]
    f = selected_rules[:, 5]

    # 取出当前的 X 和 Y 坐标
    x = points[:, 0]
    y = points[:, 1]

    # 核心张量运算：利用广播机制(Broadcasting)同时计算所有新坐标
    new_x = a * x + b * y + e
    new_y = c * x + d * y + f

    # 更新坐标池
    points[:, 0] = new_x
    points[:, 1] = new_y

# 4. 绘图渲染
# 注意：因为没有使用 GPU，这里直接 .numpy() 即可，不需要 .cpu() 了
final_points = points.numpy()

plt.figure(figsize=(6, 10), facecolor='black')
plt.scatter(final_points[:, 0], final_points[:, 1], s=0.05, color='limegreen', marker='.')
plt.title("Barnsley Fern (PyTorch Tensors)")
plt.axis('off')
plt.tight_layout()
plt.show()