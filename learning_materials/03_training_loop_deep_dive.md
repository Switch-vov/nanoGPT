# 第03章：训练循环完全指南 - 理解梯度下降

> **学习目标**：深入理解GPT如何通过梯度下降学习  
> **难度等级**：🌿 进阶  
> **预计时间**：35-45分钟  
> **前置知识**：第01章配置参数、第02章数据加载

---

## 🎯 你将学到什么

学完本章，你将能够：
- ✅ 理解梯度下降的数学原理和直觉
- ✅ 掌握完整训练循环的每个步骤
- ✅ 理解前向传播、反向传播的工作原理
- ✅ 知道梯度累积、梯度裁剪的作用
- ✅ 理解AdamW优化器如何更新参数
- ✅ 会调试训练问题（loss不降、NaN等）

---

## 💭 开始之前：机器学习到底是怎么"学习"的？

想象你在教一个学生做数学题：

```
❌ 传统编程方式：
  你告诉他："遇到这种题，用这个公式"
  → 只能解决你教过的题型

✅ 机器学习方式：
  1. 学生先瞎猜答案
  2. 你告诉他对错
  3. 他根据错误调整思路
  4. 重复2-3，直到学会
  → 可以举一反三！
```

**这就是梯度下降的本质：通过不断试错来学习！**

在深入代码前，我们必须理解**梯度下降**的本质。

---

## 📚 第一部分：梯度下降的直觉理解

### 🏔️ 比喻：下山找最低点

想象你在一座大雾弥漫的山上，目标是找到山脚（最低点）：

```
          ⛰️
        /    \
       /      \
      /   你🚶  \
     /          \
    /            \
   /              \
  /________________\
      山脚(目标)
```

**问题：大雾弥漫，你看不到山脚在哪里！**

**解决方案：**
1. 感受脚下的坡度（计算梯度）
2. 往下坡方向走一小步（参数更新）
3. 重复1-2，直到到达平地（收敛）

**这就是梯度下降！**

---

## 📚 第二部分：数学原理（用最简单例子）

### 例子：训练一个超简单的"模型"

假设我们要预测：给定输入x，预测输出y

```python
# 真实规律：y = 2x
# 但模型不知道，需要学习

# 模型：y_pred = w × x （只有一个参数w）
# 目标：找到 w = 2
```

**训练数据：**
```python
x_train = [1, 2, 3, 4, 5]
y_train = [2, 4, 6, 8, 10]
```

### 训练过程（手工计算）

**初始状态：**
```python
w = 0.5  # 随机初始化（瞎猜）
learning_rate = 0.1
```

**迭代1：**

```python
# 1. 前向传播（预测）
x = 1
y_true = 2
y_pred = w × x = 0.5 × 1 = 0.5

# 2. 计算损失
loss = (y_pred - y_true)² = (0.5 - 2)² = 2.25

# 3. 计算梯度（数学推导）
# loss = (w×x - y)²
# ∂loss/∂w = 2(w×x - y) × x
gradient = 2 × (0.5 - 2) × 1 = -3.0

# 4. 更新参数
w = w - learning_rate × gradient
w = 0.5 - 0.1 × (-3.0) = 0.5 + 0.3 = 0.8
```

**迭代2：**
```python
x = 2
y_true = 4
y_pred = 0.8 × 2 = 1.6
loss = (1.6 - 4)² = 5.76
gradient = 2 × (1.6 - 4) × 2 = -9.6
w = 0.8 - 0.1 × (-9.6) = 0.8 + 0.96 = 1.76
```

**迭代3：**
```python
w = 1.76 → 1.88 → 1.94 → 1.97 → 1.99 → 2.00 ✅
```

**可视化：**
```
w的变化过程：

     2.0 |                      ⭐ (目标)
         |                   ●
     1.5 |              ●
         |         ●
     1.0 |    ●
         | ●
     0.5 |●
         |
     0.0 |__________________________________
          0   1   2   3   4   5   6   7   (迭代次数)
```

---

## 📚 第三部分：GPT训练的完整流程

### 🔥 单次迭代的详细步骤

让我们用实际数值演示一次完整的训练迭代：

```python
# 假设我们已经完成数据加载
X.shape = [4, 8]   # batch_size=4, block_size=8
Y.shape = [4, 8]

X = tensor([
    [20, 43,  1, 60, 43,  1, 52, 47],  # "he we ni"
    [18, 43, 44, 53, 43,  1, 56, 60],  # "Before w"
    [45, 53, 47, 52, 45,  1, 47, 52],  # "going in"
    [56, 46, 43,  1, 49, 47, 52, 45],  # "the king"
])

Y = tensor([
    [43,  1, 60, 43,  1, 52, 47, 56],  
    [43, 44, 53, 43,  1, 56, 60, 43],  
    [53, 47, 52, 45,  1, 47, 52, 56],  
    [46, 43,  1, 49, 47, 52, 45, 27],  
])
```

---

#### 步骤1：前向传播（Forward Pass）

```python
# 代码（train.py 第300行）
with ctx:
    logits, loss = model(X, Y)
```

**发生了什么？（简化版）**

```python
# 模型内部（简化）
def model(X, Y):
    # 1. 嵌入层：把token ID转换为向量
    # X[0,0] = 20 → embedding[20] = [0.23, -0.45, 0.67, ..., 0.12]
    #                                 (384维向量)
    embeddings = token_embedding(X)  # [4, 8, 384]
    
    # 2. 位置编码：告诉模型每个token的位置
    positions = position_embedding(torch.arange(8))  # [8, 384]
    x = embeddings + positions  # [4, 8, 384]
    
    # 3. Transformer层（6层）
    for layer in transformer_blocks:
        x = layer(x)  # 自注意力 + MLP
    
    # 4. 输出层：预测下一个token的概率
    logits = output_layer(x)  # [4, 8, 65]
    #         ↑ 对于65个字符，每个的概率
    
    # 5. 计算损失
    loss = cross_entropy(logits, Y)
    
    return logits, loss
```

**具体数值示例：**

```python
# 以第一个样本的第一个位置为例
输入: X[0, 0] = 20  (字符 'h')
目标: Y[0, 0] = 43  (字符 'e')

# 经过模型后
logits[0, 0] = [
    0.1,   # 字符0的分数
    0.3,   # 字符1的分数
    ...
    2.8,   # 字符43的分数 (正确答案，期望最高！)
    ...
    0.2,   # 字符64的分数
]  # 共65个分数

# 转换为概率（softmax）
probs[0, 0] = [
    0.01,   # 字符0: 1%
    0.02,   # 字符1: 2%
    ...
    0.35,   # 字符43: 35% ← 最高概率
    ...
    0.01,   # 字符64: 1%
]  # 总和 = 100%

# 理想情况：
ideal_probs = [
    0.00,   # 字符0: 0%
    0.00,   # 字符1: 0%
    ...
    1.00,   # 字符43: 100% ← 应该100%确定
    ...
    0.00,   # 字符64: 0%
]
```

**损失计算（Cross Entropy）：**

```python
# 交叉熵损失公式：
loss = -log(P(正确答案))

# 计算
loss = -log(0.35) = 1.05

# 直觉理解：
如果 P(正确答案) = 0.99 → loss = -log(0.99) = 0.01 (很好！)
如果 P(正确答案) = 0.50 → loss = -log(0.50) = 0.69 (一般)
如果 P(正确答案) = 0.01 → loss = -log(0.01) = 4.61 (很差！)

# 对所有位置、所有样本求平均
total_loss = average([loss₀₀, loss₀₁, ..., loss₃₇])
           = 2.45  # 示例值
```

---

#### 步骤2：梯度累积（Gradient Accumulation）

```python
# 代码（train.py 第292-305行）
for micro_step in range(gradient_accumulation_steps):
    with ctx:
        logits, loss = model(X, Y)
        loss = loss / gradient_accumulation_steps
    X, Y = get_batch('train')
    scaler.scale(loss).backward()
```

**为什么要除以 gradient_accumulation_steps？**

```python
# 假设 gradient_accumulation_steps = 4

# 不除的话：
micro_step 1: loss₁ = 2.5 → backward → 累积梯度 = 2.5
micro_step 2: loss₂ = 2.3 → backward → 累积梯度 = 2.5 + 2.3 = 4.8
micro_step 3: loss₃ = 2.4 → backward → 累积梯度 = 4.8 + 2.4 = 7.2
micro_step 4: loss₄ = 2.6 → backward → 累积梯度 = 7.2 + 2.6 = 9.8

最终梯度 = 9.8  （太大了！是正常的4倍）

# 除以4的话：
micro_step 1: loss₁ = 2.5/4 = 0.625 → backward → 累积梯度 = 0.625
micro_step 2: loss₂ = 2.3/4 = 0.575 → backward → 累积梯度 = 1.200
micro_step 3: loss₃ = 2.4/4 = 0.600 → backward → 累积梯度 = 1.800
micro_step 4: loss₄ = 2.6/4 = 0.650 → backward → 累积梯度 = 2.450

最终梯度 = 2.45  （正确！相当于batch_size=64的平均梯度）
```

---

#### 步骤3：反向传播（Backward Pass）

```python
scaler.scale(loss).backward()
```

**发生了什么？（超详细）**

```python
# 链式法则（Chain Rule）计算梯度

# 假设模型只有3个参数：w1, w2, w3
# 前向传播路径：
X → w1 → h1 → w2 → h2 → w3 → output → loss

# 反向传播计算每个参数的梯度：

# 1. 从loss开始
∂loss/∂loss = 1  # 自己对自己的导数=1

# 2. output层的梯度
∂loss/∂output = ... (由交叉熵公式计算)

# 3. w3的梯度
∂loss/∂w3 = ∂loss/∂output × ∂output/∂w3

# 4. h2的梯度  
∂loss/∂h2 = ∂loss/∂output × ∂output/∂h2

# 5. w2的梯度
∂loss/∂w2 = ∂loss/∂h2 × ∂h2/∂w2

# ... 一直传播到w1

# 结果：每个参数都知道"怎么调整能降低loss"
```

**具体数值示例：**

```python
# 某个参数w的梯度计算

当前值: w = 0.523
loss = 2.45

# 反向传播计算得到：
∂loss/∂w = -0.38

# 含义：
# w增加0.001 → loss减少 0.38×0.001 = 0.00038
# w增加0.1   → loss减少 0.38×0.1   = 0.038

# 所以我们应该增加w！（因为梯度是负的）
```

---

#### 步骤4：梯度裁剪（Gradient Clipping）

```python
# 代码（train.py 第307-309行）
if grad_clip != 0.0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
```

**为什么需要？**

```python
# 问题：梯度爆炸

正常梯度：
w1.grad = 0.23
w2.grad = -0.45
w3.grad = 0.67
...

异常情况（梯度爆炸）：
w1.grad = 0.23
w2.grad = -0.45
w3.grad = 0.67
w4.grad = 234.56  ⚠️ 突然变得很大！
w5.grad = -1892.34 ⚠️ 
w6.grad = 45629.12 ⚠️ 

如果不裁剪，参数更新：
w4 = w4 - 0.001 × 234.56 = w4 - 0.23  (变化太大！)
w5 = w5 - 0.001 × (-1892.34) = w5 + 1.89 (巨变！)
→ 模型崩溃，loss变成NaN
```

**裁剪过程：**

```python
# 1. 计算所有梯度的总范数
grad_norm = sqrt(sum(g² for g in all_gradients))
          = sqrt(0.23² + 0.45² + ... + 45629.12²)
          = 45630.5  # 非常大！

# 2. 设定阈值
grad_clip = 1.0

# 3. 如果超过阈值，按比例缩小
if grad_norm > grad_clip:
    scaling_factor = grad_clip / grad_norm
                   = 1.0 / 45630.5
                   = 0.000022
    
    # 所有梯度乘以这个因子
    w1.grad = 0.23 × 0.000022 = 0.0000051
    w2.grad = -0.45 × 0.000022 = -0.0000099
    ...
    w6.grad = 45629.12 × 0.000022 = 1.00
    
    # 新的grad_norm刚好等于1.0

# 效果：保持梯度的相对方向，但限制大小
```

---

#### 步骤5：参数更新（Optimizer Step）

```python
# 代码（train.py 第311-314行）
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad(set_to_none=True)
```

**AdamW优化器的更新过程：**

```python
# 标准SGD（最简单）：
w = w - learning_rate × gradient
w = 0.523 - 0.001 × (-0.38) = 0.523 + 0.00038 = 0.52338

# AdamW（复杂但强大）：
# 维护两个移动平均：
# m: 梯度的一阶矩（momentum，动量）
# v: 梯度的二阶矩（variance，方差）

# 初始状态
m = 0
v = 0
beta1 = 0.9
beta2 = 0.95
learning_rate = 0.001
weight_decay = 0.1

# 第1步
gradient = -0.38

# 更新m和v
m = beta1 × m + (1-beta1) × gradient
  = 0.9 × 0 + 0.1 × (-0.38)
  = -0.038

v = beta2 × v + (1-beta2) × gradient²
  = 0.95 × 0 + 0.05 × (0.38)²
  = 0.00722

# 偏差修正（前几步很重要）
m_corrected = m / (1 - beta1^t)
            = -0.038 / (1 - 0.9)
            = -0.38

v_corrected = v / (1 - beta2^t)
            = 0.00722 / (1 - 0.95)
            = 0.1444

# 参数更新（加上weight_decay）
w = w - learning_rate × (m_corrected / sqrt(v_corrected) + weight_decay × w)
  = 0.523 - 0.001 × (-0.38/sqrt(0.1444) + 0.1×0.523)
  = 0.523 - 0.001 × (-1.00 + 0.0523)
  = 0.523 + 0.001 × 0.9477
  = 0.52395

# 第2步
gradient = -0.42  # 新的梯度

m = 0.9 × (-0.038) + 0.1 × (-0.42) = -0.0762
v = 0.95 × 0.00722 + 0.05 × (0.42)² = 0.01567

w = ... (继续更新)
```

**为什么AdamW比SGD好？**

```
场景1：稳定下降区域
  SGD: gradient = 0.1 → update = 0.1 × lr
  Adam: 适应性调整，步长稳定

场景2：陡峭区域
  SGD: gradient = 100 → update = 100 × lr (太大！)
  Adam: 检测到方差大 → 自动减小步长 ✅

场景3：平坦区域  
  SGD: gradient = 0.001 → update很小 (太慢)
  Adam: 累积动量 → 继续前进 ✅

场景4：震荡区域
  SGD: gradient = [+50, -48, +52, -49, ...]
  Adam: 动量相互抵消 → 稳定前进 ✅
```

---

## 📚 第四部分：完整训练循环可视化

### 📊 一个完整epoch的loss变化

```python
# 假设训练1000步

步骤    Loss    说明
----    ----    ----
0       4.174   随机初始化，乱猜
10      3.892   开始学习
50      3.245   学到一些模式
100     2.687   越来越好
200     2.234   
500     1.845   接近收敛
1000    1.469   ✅ 训练完成
```

**可视化：**
```
Loss
4.0 |●
    |  ●
3.5 |    ●
    |      ●●
3.0 |         ●●
    |            ●●
2.5 |              ●●●
    |                  ●●●
2.0 |                     ●●●●
    |                         ●●●●●
1.5 |                              ●●●●●●●
    |_________________________________________
    0    200   400   600   800   1000  (步数)
```

### 🔍 参数的变化轨迹

```python
# 某个参数w在训练过程中的变化

步骤    w值     梯度     更新量
----    ---     ----     -----
0       0.523   -0.38    +0.00038
1       0.52338 -0.42    +0.00042
2       0.52380 -0.39    +0.00039
...
100     0.687   -0.12    +0.00012
...
500     0.823   -0.03    +0.00003
...
1000    0.856   -0.001   +0.000001  ← 几乎不变了（收敛）
```

---

## 📚 第五部分：实战调试技巧

### 🔧 打印梯度信息

创建调试脚本：

```python
# debug_training.py

import torch
from model import GPT, GPTConfig

# 创建小模型
config = GPTConfig(
    vocab_size=65,
    n_layer=2,
    n_head=2,
    n_embd=64,
    block_size=16,
)
model = GPT(config)

# 虚拟数据
X = torch.randint(0, 65, (4, 16))
Y = torch.randint(0, 65, (4, 16))

# 前向传播
logits, loss = model(X, Y)
print(f"Loss: {loss.item():.4f}")

# 反向传播
loss.backward()

# 检查梯度
print("\n参数梯度统计：")
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_mean = param.grad.mean().item()
        grad_std = param.grad.std().item()
        grad_max = param.grad.abs().max().item()
        print(f"{name:40s} | mean: {grad_mean:+.6f}, std: {grad_std:.6f}, max: {grad_max:.6f}")

# 输出示例：
# transformer.wte.weight                   | mean: +0.000123, std: 0.002341, max: 0.023451
# transformer.wpe.weight                   | mean: -0.000087, std: 0.001234, max: 0.012345
# transformer.h.0.attn.c_attn.weight       | mean: +0.000456, std: 0.003456, max: 0.034567
# ...
```

### 🎯 监控训练健康度

```python
# 添加到train.py中

# 每100步检查一次
if iter_num % 100 == 0:
    # 1. 检查梯度范数
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    print(f"Gradient norm: {total_norm:.4f}")
    
    # 2. 检查参数更新比例
    # 更新量应该是参数值的0.001-0.01倍
    
    # 3. 检查loss是否为NaN
    if math.isnan(loss.item()):
        print("❌ Loss is NaN! Training crashed!")
        break
    
    # 4. 检查loss是否在下降
    # ...
```

### 🐛 常见问题诊断

```python
# 问题1: Loss不下降
可能原因：
  - learning_rate太小 → 增大到1e-3
  - 模型太小 → 增加n_layer, n_embd
  - 数据有问题 → 检查get_batch()

# 问题2: Loss变成NaN
可能原因：
  - learning_rate太大 → 减小到1e-4
  - 梯度爆炸 → 启用grad_clip=1.0
  - 数值不稳定 → 使用float32而不是float16

# 问题3: 过拟合（train loss << val loss）
解决方案：
  - 增加dropout=0.2
  - 增加weight_decay=0.1
  - 获取更多数据
  - 减小模型

# 问题4: 训练太慢
优化方案：
  - 启用compile=True
  - 增大batch_size
  - 使用多GPU
  - 减小eval_interval
```

---

## 🎓 总结：训练循环完整流程

```
┌─────────────────────────────────────────────┐
│ 1. 初始化                                    │
│    - 创建模型（随机参数）                      │
│    - 创建优化器                               │
│    - 设置学习率                               │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ 2. 主循环 (while iter_num < max_iters)      │
└─────────────────────────────────────────────┘
                    ↓
        ┌───────────────────────┐
        │ 2.1 调整学习率         │
        │  lr = get_lr(iter_num) │
        └───────────────────────┘
                    ↓
        ┌──────────────────────────────────────┐
        │ 2.2 梯度累积循环 (4次)                 │
        │  ┌────────────────────────────────┐  │
        │  │ a) 加载数据: X, Y = get_batch() │  │
        │  │ b) 前向传播: logits, loss = model(X,Y) │
        │  │ c) 反向传播: loss.backward()    │  │
        │  │ d) 累积梯度                      │  │
        │  └────────────────────────────────┘  │
        └──────────────────────────────────────┘
                    ↓
        ┌───────────────────────┐
        │ 2.3 梯度裁剪           │
        │  clip_grad_norm_(...)  │
        └───────────────────────┘
                    ↓
        ┌───────────────────────┐
        │ 2.4 参数更新           │
        │  optimizer.step()      │
        └───────────────────────┘
                    ↓
        ┌───────────────────────┐
        │ 2.5 清空梯度           │
        │  optimizer.zero_grad() │
        └───────────────────────┘
                    ↓
        ┌────────────────────────────┐
        │ 2.6 定期评估 (每2000步)     │
        │  - 计算val loss             │
        │  - 保存checkpoint           │
        └────────────────────────────┘
                    ↓
        ┌───────────────────────┐
        │ 2.7 记录日志           │
        │  - 打印loss             │
        │  - 计算速度             │
        └───────────────────────┘
                    ↓
              iter_num += 1
                    ↓
           达到max_iters? ──No──→ 返回2.1
                    │
                   Yes
                    ↓
              训练完成！✅
```

---

---

## 🎓 总结与检查

### ✅ 知识检查清单

完成学习后，你应该能够：

**基础概念（必须掌握）**
- [ ] 用"下山"比喻解释梯度下降
- [ ] 理解loss是什么，为什么要最小化
- [ ] 知道前向传播和反向传播的区别
- [ ] 理解梯度的含义（参数调整方向）
- [ ] 知道learning_rate的作用

**进阶理解（建议掌握）**
- [ ] 理解梯度累积的工作原理
- [ ] 知道为什么需要梯度裁剪
- [ ] 理解AdamW比SGD好在哪里
- [ ] 能解释交叉熵损失的计算
- [ ] 知道如何诊断训练问题

**实战能力（最终目标）**
- [ ] 能写出调试脚本检查梯度
- [ ] 会监控训练健康度
- [ ] 能解决loss不降、NaN等问题
- [ ] 理解完整训练循环的每个步骤
- [ ] 能优化训练速度和稳定性

### 📊 核心要点总结

```python
训练循环 = 重复以下步骤：

1. 前向传播 (Forward Pass)
   输入 → 模型 → 输出 → 计算loss
   
2. 反向传播 (Backward Pass)
   loss → 链式法则 → 计算每个参数的梯度
   
3. 梯度裁剪 (Gradient Clipping)
   防止梯度爆炸，保持训练稳定
   
4. 参数更新 (Optimizer Step)
   参数 = 参数 - learning_rate × 梯度
   （AdamW有自适应调整）
   
5. 清空梯度 (Zero Grad)
   准备下一次迭代
```

### 🎯 关键公式

```python
# 梯度下降
w_new = w_old - learning_rate × gradient

# 交叉熵损失
loss = -log(P(正确答案))

# 梯度裁剪
if grad_norm > threshold:
    gradient = gradient × (threshold / grad_norm)

# AdamW更新
m = β1 × m + (1-β1) × gradient
v = β2 × v + (1-β2) × gradient²
w = w - lr × (m / √v + weight_decay × w)
```

### 🚀 下一步学习

现在你完全理解了训练循环！你知道了：

✅ 梯度下降的数学原理  
✅ 前向传播计算loss  
✅ 反向传播计算梯度  
✅ 梯度累积的实现  
✅ 梯度裁剪防止爆炸  
✅ AdamW优化器更新参数  
✅ 整个循环如何串联  

**接下来应该学习：**

1. **04_complete_guide_and_experiments.md** - 完整指南与实验
2. **05_model_architecture_deep_dive.md** - 深入理解Transformer
3. **实战练习** - 运行调试脚本，监控训练过程

### 💡 实践建议

1. **运行调试脚本**：查看实际的梯度和参数变化
2. **监控训练**：添加日志，观察loss曲线
3. **实验参数**：尝试不同的learning_rate和优化器
4. **故意制造问题**：设置很大的lr，观察梯度爆炸

---

## 📚 推荐资源

### 📖 延伸阅读
- [梯度下降可视化](https://distill.pub/2017/momentum/)
- [Adam优化器论文](https://arxiv.org/abs/1412.6980)
- [反向传播详解](http://colah.github.io/posts/2015-08-Backprop/)

### 🎥 视频教程
- [3Blue1Brown: 神经网络](https://www.youtube.com/watch?v=aircAruvnKk)
- [Andrej Karpathy: 反向传播](https://www.youtube.com/watch?v=VMj-3S1tku0)

### 🔧 实用工具
- [TensorBoard](https://www.tensorflow.org/tensorboard) - 可视化训练过程
- [Weights & Biases](https://wandb.ai/) - 实验追踪

---

## 🐛 常见问题 FAQ

### Q1: 为什么loss一开始很高？

```python
# 初始状态：参数随机初始化
# 模型完全不知道该预测什么

初始loss ≈ -log(1/vocab_size)
        = -log(1/65)
        = 4.17

# 这是随机猜测的loss
# 随着训练，loss会逐渐下降
```

### Q2: 梯度累积和增大batch_size有什么区别？

```python
# 效果相同：
batch_size = 64, grad_accum = 1
batch_size = 16, grad_accum = 4

# 区别：
梯度累积：
  ✅ 显存占用小
  ❌ 训练速度稍慢（多次前向传播）
  
大batch_size：
  ✅ 训练速度快（一次前向传播）
  ❌ 显存占用大
```

### Q3: 为什么要用AdamW而不是SGD？

```python
SGD的问题：
  - 需要精心调整learning_rate
  - 在不同参数上使用相同的lr
  - 容易卡在平坦区域
  
AdamW的优势：
  ✅ 自适应调整每个参数的lr
  ✅ 有动量，可以越过局部最优
  ✅ 更稳定，更容易收敛
  ✅ 是目前训练LLM的标准选择
```

### Q4: 如何判断训练是否正常？

```python
健康的训练：
  ✅ loss稳定下降
  ✅ 梯度范数在0.1-10之间
  ✅ 参数更新量是参数值的0.001-0.01倍
  ✅ 没有NaN或Inf
  
异常情况：
  ❌ loss不降或震荡 → 调小lr
  ❌ loss变成NaN → 梯度爆炸，启用grad_clip
  ❌ loss下降太慢 → 调大lr或增大模型
  ❌ train loss << val loss → 过拟合，增加正则化
```

---

**恭喜你完成第03章！** 🎉

你现在已经掌握了训练循环的完整原理。这是深度学习的核心，理解了这一章，你就理解了AI如何"学习"。

**准备好了吗？让我们继续前进！** → [04_complete_guide_and_experiments.md](04_complete_guide_and_experiments.md)
