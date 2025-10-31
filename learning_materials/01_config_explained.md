# 第01章：配置参数完全指南 - 从零开始

> **学习目标**：理解GPT训练的所有配置参数，学会根据硬件条件调整配置  
> **难度等级**：🌱 入门  
> **预计时间**：30-40分钟  
> **前置知识**：无需任何基础

---

## 🎯 你将学到什么

学完本章，你将能够：
- ✅ 理解每个配置参数的含义和作用
- ✅ 根据显卡显存调整训练配置
- ✅ 解决"显存不足"、"训练太慢"等常见问题
- ✅ 为自己的项目选择合适的参数

---

## 💭 开始之前：为什么要学配置参数？

想象你要训练一个AI模型，就像培养一个学生：

```
❌ 错误的配置：
  - 一次教100本书（显存爆炸）
  - 学习速度太快（学不进去）
  - 没有复习机制（忘得快）
  
✅ 正确的配置：
  - 循序渐进，每次学一点
  - 速度适中，稳步提升
  - 定期复习，巩固知识
```

**配置参数就是控制"如何教"的关键！**

---

## 📚 第一部分：数据相关参数（最基础）

### 🌱 1.1 batch_size（批次大小）

#### 💡 直观理解

**是什么？**  
每次训练时，同时给模型看多少个样本。

**生活比喻：老师教学生**
```
batch_size = 1：
  老师一次只教一个学生
  ✅ 优点：针对性强，每个学生都能得到关注
  ❌ 缺点：效率低，教完所有学生要很久
  
batch_size = 32：
  老师一次教32个学生
  ✅ 优点：效率高，可以快速教完
  ❌ 缺点：平均化，个体差异被忽略
  
batch_size = 128：
  老师一次教128个学生
  ✅ 优点：非常高效
  ❌ 缺点：需要大教室（显存），个性化更弱
```

#### 📊 具体例子

```python
# 假设我们在训练莎士比亚文本生成
batch_size = 4
block_size = 8  # 每个样本8个字符

# 一次训练迭代可能是这样的：
样本1: "To be or" → 预测下一个字符 "not"
样本2: "not to b" → 预测下一个字符 "e"
样本3: "be that " → 预测下一个字符 "is"
样本4: "is the q" → 预测下一个字符 "u"

# 模型同时处理这4个样本，计算平均损失
```

#### 📈 数值影响对比表

| batch_size | 训练速度 | 显存使用 | 梯度稳定性 | 建议场景 |
|-----------|---------|---------|-----------|----------|
| 1-8       | 🐌 很慢  | 💚 很低  | ⚠️ 不稳定  | 调试代码 |
| 16-32     | 🚶 适中  | 💛 适中  | ✅ 稳定    | **初学者推荐** |
| 64-128    | 🏃 很快  | 💔 很高  | ✅ 很稳定  | 大规模训练 |

#### 🎯 如何选择？

```python
# 决策树
if 你是初学者:
    batch_size = 32  # 平衡点
elif 显存不够:
    batch_size = 8   # 先能跑起来
elif 追求速度:
    batch_size = 64  # 充分利用GPU
else:
    batch_size = 32  # 默认选择
```

---

### 🌱 1.2 block_size（上下文长度）

#### 💡 直观理解

**是什么？**  
模型一次能"看到"多少个字符/词，也叫上下文窗口。

**生活比喻：记忆力**
```
block_size = 8：
  只记得最近8个词（健忘症患者）
  "The quick brown fox jumps over the lazy"
   ^^^^^^^^ 只能看到这8个词
  
block_size = 256：
  能记得256个词（记忆力很好）
  可以理解更长的上下文
  
block_size = 1024：
  能记得1024个词（过目不忘）
  可以理解整篇文章的逻辑
```

#### 📊 实际例子

假设有句子："The quick brown fox jumps over the lazy dog"

```python
# block_size = 4（只看4个词）
输入: ["The", "quick", "brown", "fox"]
预测: "jumps"
问题：看不到后面的"lazy dog"，理解不完整

# block_size = 10（看10个词）
输入: ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
预测: 可以理解完整句子！
```

#### ⚖️ 权衡分析

```
block_size 增大的影响：

✅ 好处：
  - 理解力↑：能看到更多上下文
  - 生成质量↑：文本更连贯
  
❌ 代价：
  - 显存使用↑↑（平方级增长！）
  - 训练时间↑↑
  - 计算复杂度 = O(block_size²)
  
为什么是平方？因为Attention机制！
每个词都要和其他所有词计算关系：
  4个词 → 4×4 = 16次计算
  256个词 → 256×256 = 65,536次计算
```

#### 📐 显存计算

```python
# 显存占用估算
显存 ≈ batch_size × block_size² × hidden_size × 4 bytes

例子：
batch_size = 32
block_size = 256
hidden_size = 768

显存 ≈ 32 × 256² × 768 × 4 / 1024³
     ≈ 1.5 GB（仅激活值）
```

---

### 🌿 1.3 gradient_accumulation_steps（梯度累积）

#### 💡 直观理解

**这是最难理解的概念！让我用最简单的方式解释：**

**核心问题：**  
你的显卡只有8GB显存，但你想用相当于32GB显存的效果。

**解决方案：分期付款训练法**

就像买房子：
- ❌ 一次性付款：需要100万现金（显存爆炸）
- ✅ 分期付款：每月付5万，20个月付清（梯度累积）

#### 📊 详细过程模拟

```python
# 方法1：直接用大batch（显存爆炸！）
batch_size = 64  # 需要32GB显存 ❌
gradient_accumulation_steps = 1

# 方法2：梯度累积（巧妙！）
batch_size = 16  # 只需8GB显存 ✅
gradient_accumulation_steps = 4  # 累积4次
# 等效batch_size = 16 × 4 = 64
```

#### 🎬 动画演示

```
=== 传统训练 (batch_size=64, 显存32GB) ===

步骤1: 读取64个样本 → 前向传播 → 计算loss → 反向传播 → 更新参数
显存: ████████████████ (32GB) ❌ 显存不够！

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

=== 梯度累积训练 (batch_size=16, 累积4次, 显存8GB) ===

步骤1: 读取16个样本 → 前向 → 反向 → 梯度暂存（不更新参数）
显存: ████ (8GB) ✅
梯度1 = [0.1, 0.2, 0.3, ...]

步骤2: 读取16个样本 → 前向 → 反向 → 梯度累加
显存: ████ (8GB) ✅
梯度2 = [0.15, 0.18, 0.25, ...]
累积梯度 = 梯度1 + 梯度2 = [0.25, 0.38, 0.55, ...]

步骤3: 读取16个样本 → 前向 → 反向 → 梯度累加
显存: ████ (8GB) ✅
梯度3 = [0.12, 0.22, 0.28, ...]
累积梯度 = 梯度1 + 梯度2 + 梯度3 = [0.37, 0.60, 0.83, ...]

步骤4: 读取16个样本 → 前向 → 反向 → 梯度累加 → 更新参数！
显存: ████ (8GB) ✅
梯度4 = [0.11, 0.19, 0.27, ...]
累积梯度 = 梯度1 + 梯度2 + 梯度3 + 梯度4 = [0.48, 0.79, 1.10, ...]

最终效果：等同于batch_size=64，但显存只用了8GB！
```

#### 📐 数学原理

```python
# 为什么有效？

平均梯度 = (梯度1 + 梯度2 + 梯度3 + 梯度4) / 4

这等价于：
一次性计算64个样本的平均梯度

数学证明：
梯度(样本1-16) + 梯度(样本17-32) + 梯度(样本33-48) + 梯度(样本49-64)
= 梯度(样本1-64)
```

#### 🎯 实战配置

```python
# 根据显存选择配置

# 显存 4GB（入门显卡）
batch_size = 8
gradient_accumulation_steps = 8
# 等效batch_size = 64

# 显存 8GB（主流显卡）
batch_size = 16
gradient_accumulation_steps = 4
# 等效batch_size = 64

# 显存 16GB（高端显卡）
batch_size = 32
gradient_accumulation_steps = 2
# 等效batch_size = 64

# 显存 24GB+（专业显卡）
batch_size = 64
gradient_accumulation_steps = 1
# 直接用大batch
```

---

## 📚 第二部分：模型架构参数（核心）

### 🌿 2.1 n_layer（层数）

#### 💡 直观理解

**是什么？**  
模型有多少层Transformer块，决定了模型的"深度"。

**生活比喻：学历等级**
```
n_layer = 1：  幼儿园水平
  只能理解最简单的模式
  
n_layer = 6：  高中生水平
  可以理解复杂的语法和简单推理
  （莎士比亚模型用这个）
  
n_layer = 12： 大学生水平
  可以理解抽象概念和复杂推理
  （GPT-2用这个）
  
n_layer = 96： 博士水平
  可以理解非常深层的知识
  （GPT-3用这个）
```

#### 📊 具体影响

```python
# 参数量和训练时间对比（假设n_embd=384, n_head=6）

n_layer = 2:
  参数量: ~10M
  训练时间: 30分钟
  能力: 只能学简单模式
  
n_layer = 6:
  参数量: ~40M
  训练时间: 2小时
  能力: 可以生成连贯文本
  
n_layer = 12:
  参数量: ~124M
  训练时间: 1天
  能力: 可以理解复杂语义
  
n_layer = 24:
  参数量: ~350M
  训练时间: 1周
  能力: 接近人类写作水平
```

#### 🎯 选择建议

```python
# 决策树
if 只是学习/实验:
    n_layer = 4  # 快速验证想法
elif 小数据集（<10MB）:
    n_layer = 6  # 避免过拟合
elif 中等数据集（10-100MB）:
    n_layer = 12  # 平衡点
elif 大数据集（>1GB）:
    n_layer = 24  # 充分利用数据
```

---

### 🌿 2.2 n_head（注意力头数）

#### 💡 直观理解

**是什么？**  
模型同时从多少个不同角度理解文本。

**生活比喻：多角度分析**

想象你在分析一首诗："To be or not to be, that is the question"

```
n_head = 1（单一视角）：
  只从语法角度分析
  "To be" 是动词不定式
  
n_head = 6（多重视角）：
  头1：语法结构（"To be"是动词）
  头2：语义关系（"be"和"question"的关系）
  头3：韵律节奏（重音模式）
  头4：情感色彩（哲学性的疑问）
  头5：上下文连贯（前后逻辑）
  头6：风格特征（莎士比亚风格）
  
最后综合所有视角 → 完整理解
```

#### 📐 技术细节

```python
# 多头注意力的工作原理

n_embd = 384  # 总维度
n_head = 6    # 6个头

每个头的维度 = 384 / 6 = 64

# 维度分配
头1处理：维度 0-63    （负责语法）
头2处理：维度 64-127  （负责语义）
头3处理：维度 128-191 （负责韵律）
头4处理：维度 192-255 （负责情感）
头5处理：维度 256-319 （负责连贯）
头6处理：维度 320-383 （负责风格）

# 最后合并所有头的结果
输出 = Concat(头1, 头2, 头3, 头4, 头5, 头6)
```

#### ⚠️ 重要约束

```python
# n_head 必须能整除 n_embd！

✅ 正确配置：
n_embd = 384, n_head = 6  # 384 / 6 = 64 ✅
n_embd = 768, n_head = 12 # 768 / 12 = 64 ✅

❌ 错误配置：
n_embd = 384, n_head = 5  # 384 / 5 = 76.8 ❌
n_embd = 768, n_head = 10 # 768 / 10 = 76.8 ❌
```

---

### 🌿 2.3 n_embd（嵌入维度）

#### 💡 直观理解

**是什么？**  
用多少个数字来表示一个词。

**生活比喻：描述一个人**

```
n_embd = 2（2个特征）：
  特征1: 身高
  特征2: 体重
  → 信息太少，很多人无法区分
  
n_embd = 10（10个特征）：
  身高、体重、年龄、肤色、发色、
  眼睛颜色、性格、爱好、职业、学历
  → 可以区分大部分人
  
n_embd = 384（384个特征）：
  包含非常细微的特征
  → 可以精确描述每个人的独特性
  
n_embd = 768（GPT-2标准）：
  可以捕捉极其细微的差异
  → 理解词的深层含义和微妙关系
```

#### 📊 存储空间计算

```python
# 嵌入层参数量计算

vocab_size = 50000  # 词汇表大小
n_embd = 768        # 嵌入维度

# 嵌入矩阵大小
embedding_params = vocab_size × n_embd
                 = 50000 × 768
                 = 38,400,000 个参数

# 存储空间（float32）
storage = 38,400,000 × 4 bytes
        = 153,600,000 bytes
        ≈ 147 MB
```

#### 🎯 选择建议

```python
# 常用配置

小模型（快速实验）：
  n_embd = 256
  n_head = 4
  n_layer = 4
  参数量: ~10M

中等模型（平衡）：
  n_embd = 384
  n_head = 6
  n_layer = 6
  参数量: ~40M

大模型（高质量）：
  n_embd = 768
  n_head = 12
  n_layer = 12
  参数量: ~124M（GPT-2标准）
```

---

## 📚 第三部分：优化器参数（关键）

### 🌳 3.1 learning_rate（学习率）

#### 💡 直观理解

**最重要的超参数！**

**生活比喻：学习的步伐**

```
learning_rate = 1.0（太大）：
  学生学得太激进
  今天全懂了，明天全忘了，后天又全懂了...
  结果：来回震荡，永远学不好
  
learning_rate = 0.001（刚好）：
  学生稳步学习
  每天进步一点点，扎实掌握
  结果：✅ 稳定提升
  
learning_rate = 0.000001（太小）：
  学生学得太慢
  每天只记住一个字
  结果：一辈子都学不完
```

#### 📈 可视化

```
Loss（损失值）
  ^
  |
  |  lr太大 → 震荡不收敛
  |    /\    /\    /\
  |   /  \  /  \  /  \
  |  /    \/    \/    \
  |
  |  lr刚好 → 平滑下降 ✅
  |  \
  |   \____
  |        \____
  |            \____
  |                \____
  |
  |  lr太小 → 下降太慢
  |  \
  |   \
  |    \
  |     \
  |      \
  +--------------------------------> 训练步数
       0    1000   2000   3000   4000
```

#### 🎯 经验值

```python
# 从零训练

小模型（<50M参数）：
  learning_rate = 1e-3  # 0.001
  
中等模型（50-500M参数）：
  learning_rate = 6e-4  # 0.0006（GPT-2默认）
  
大模型（>500M参数）：
  learning_rate = 3e-4  # 0.0003

# 微调预训练模型

learning_rate = 1e-4  # 0.0001（避免破坏已学知识）
learning_rate = 5e-5  # 0.00005（更保守）
```

#### 🧪 如何调试

```python
# 实验方法

# 步骤1：尝试默认值
learning_rate = 6e-4
训练1000步，观察loss曲线

# 步骤2：如果loss震荡
learning_rate = 3e-4  # 减半再试

# 步骤3：如果loss下降太慢
learning_rate = 1e-3  # 加倍再试

# 步骤4：如果loss不下降
learning_rate = 1e-4  # 大幅降低
```

---

### 🌳 3.2 weight_decay（权重衰减）

#### 💡 直观理解

**是什么？**  
防止模型"死记硬背"的机制。

**生活比喻：学习方法**

```
没有weight_decay（死记硬背）：
  学生把每个例子都背下来
  
  训练集：
    "To be" → "or not to be"
    "To go" → "or not to go"
  
  测试集：
    "To run" → ？（不知道，没背过）
  
  结果：过拟合，泛化能力差

有weight_decay（理解规律）：
  学生理解了规律，不是死记
  
  学到的规律：
    "To + 动词" 后面通常跟 "or not to + 动词"
  
  测试集：
    "To run" → "or not to run" ✅
  
  结果：泛化能力强
```

#### 📐 数学原理

```python
# 参数更新公式

没有weight_decay：
  新参数 = 旧参数 - learning_rate × 梯度

有weight_decay：
  新参数 = 旧参数 - learning_rate × 梯度 - weight_decay × 旧参数
                                             ↑
                                    这一项让参数不要太大
```

#### 🎯 效果对比

```python
# 实验对比

weight_decay = 0（无正则化）：
  训练loss: 0.5  ✅ 很好
  验证loss: 1.2  ❌ 过拟合
  
weight_decay = 0.1（适度正则化）：
  训练loss: 0.8  ✅ 可接受
  验证loss: 0.9  ✅ 泛化好
  
weight_decay = 0.5（过度正则化）：
  训练loss: 1.5  ❌ 欠拟合
  验证loss: 1.6  ❌ 学不好
```

---

### 🌳 3.3 grad_clip（梯度裁剪）

#### 💡 直观理解

**是什么？**  
防止梯度爆炸，保持训练稳定。

**生活比喻：老师的反馈**

```
没有grad_clip：
  学生答错了题
  老师大喊："你全错了！！！从头学！！！"
  学生：😱 吓傻了，忘记之前学的
  结果：梯度爆炸，训练崩溃
  
有grad_clip：
  学生答错了题
  老师温和说："你有些错误，慢慢改进"
  学生：😊 稳步改进
  结果：训练稳定
```

#### 📐 工作原理

```python
# 梯度裁剪算法

grad_clip = 1.0  # 裁剪阈值

# 步骤1：计算梯度
gradients = [100, 200, 300]  # 梯度爆炸了！

# 步骤2：计算梯度范数（长度）
gradient_norm = sqrt(100² + 200² + 300²)
              = sqrt(10000 + 40000 + 90000)
              = sqrt(140000)
              = 374.17

# 步骤3：判断是否需要裁剪
if gradient_norm > grad_clip:  # 374.17 > 1.0
    # 需要裁剪
    scale = grad_clip / gradient_norm
          = 1.0 / 374.17
          = 0.00267
    
    # 缩小梯度
    gradients = gradients × scale
              = [100, 200, 300] × 0.00267
              = [0.267, 0.534, 0.801]

# 步骤4：使用裁剪后的梯度更新参数
```

#### 🎯 实战效果

```
训练曲线对比：

没有grad_clip：
Loss
  |  /\      /\
  | /  \    /  \
  |/    \  /    \  💥 训练崩溃
  |      \/      \
  +-----------------> Steps

有grad_clip：
Loss
  |  \
  |   \___
  |       \___
  |           \___  ✅ 稳定下降
  |               \___
  +---------------------> Steps
```

---

## 📚 第四部分：学习率调度（进阶）

### 🌳 4.1 warmup_iters（预热步数）

#### 💡 直观理解

**为什么需要warmup？**

想象你早上刚起床：

```
没有warmup（直接开始）：
  闹钟响 → 立即跳起来 → 跑步10公里
  结果：💥 身体受不了，可能受伤
  
有warmup（逐渐加速）：
  闹钟响 → 慢慢起床 → 伸展运动 → 慢跑 → 加速跑
  结果：✅ 状态逐渐进入，不会受伤
```

#### 📈 学习率变化曲线

```
Learning Rate
  ^
  |                /‾‾‾‾‾‾‾‾‾‾‾‾\
  |               /              \
  |              /                \
  |             /                  \___
  |            /                       \___
  |           /                            \___
  |          /                                 \___
  |         /                                      \___
  +---------|---------|-----------------|--------------|---> Steps
          warmup    正常训练           decay         结束
          (2000)                    (600000)
          
阶段1：Warmup（0-2000步）
  learning_rate: 0 → 6e-4
  作用：让模型慢慢适应
  
阶段2：正常训练（2000-600000步）
  learning_rate: 6e-4（保持不变）
  作用：稳定学习
  
阶段3：Decay（600000步后）
  learning_rate: 6e-4 → 6e-5
  作用：精细调整
```

#### 📐 Warmup公式

```python
# 线性warmup

def get_learning_rate(step, warmup_iters, max_lr):
    if step < warmup_iters:
        # warmup阶段：线性增长
        return max_lr * step / warmup_iters
    else:
        # 正常阶段
        return max_lr

# 例子
warmup_iters = 2000
max_lr = 6e-4

步骤0:    lr = 6e-4 × 0/2000    = 0
步骤500:  lr = 6e-4 × 500/2000  = 1.5e-4
步骤1000: lr = 6e-4 × 1000/2000 = 3e-4
步骤2000: lr = 6e-4 × 2000/2000 = 6e-4 ✅
```

---

### 🌳 4.2 lr_decay_iters（衰减步数）

#### 💡 直观理解

**为什么需要衰减？**

学习过程就像爬山：

```
初期（大learning_rate）：
  大步前进，快速接近山顶
  
中期（保持learning_rate）：
  稳步前进，持续攀登
  
后期（小learning_rate）：
  小心翼翼，精确找到最高点
```

#### 📐 Cosine衰减公式

```python
# Cosine Annealing（余弦退火）

def get_lr_with_decay(step, warmup_iters, lr_decay_iters, max_lr, min_lr):
    # 阶段1：Warmup
    if step < warmup_iters:
        return max_lr * step / warmup_iters
    
    # 阶段2：正常训练
    if step < lr_decay_iters:
        return max_lr
    
    # 阶段3：Cosine衰减
    decay_ratio = (step - lr_decay_iters) / (max_iters - lr_decay_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
```

---

## 📚 第五部分：实战计算示例

### 🎯 5.1 莎士比亚配置完整分析

```python
# 配置文件：config/train_shakespeare_char.py

# ===== 数据参数 =====
batch_size = 64                    # 一次处理64个样本
block_size = 256                   # 每个样本256个字符
gradient_accumulation_steps = 1    # 不使用梯度累积

# ===== 模型参数 =====
n_layer = 6                        # 6层Transformer
n_head = 6                         # 6个注意力头
n_embd = 384                       # 384维嵌入
dropout = 0.2                      # 20%的dropout

# ===== 优化器参数 =====
learning_rate = 1e-3               # 0.001
max_iters = 5000                   # 训练5000步
weight_decay = 1e-1                # 0.1
grad_clip = 1.0                    # 梯度裁剪阈值

# ===== 学习率调度 =====
warmup_iters = 100                 # 前100步warmup
lr_decay_iters = 5000              # 5000步后开始衰减
min_lr = 1e-4                      # 最小学习率0.0001
```

### 📊 5.2 计算分析

```python
# 1. 每次迭代处理的token数
tokens_per_iter = batch_size × block_size × gradient_accumulation_steps
                = 64 × 256 × 1
                = 16,384 tokens

# 2. 总共处理的token数
total_tokens = tokens_per_iter × max_iters
             = 16,384 × 5000
             = 81,920,000 tokens
             ≈ 82M tokens

# 3. 模型参数量估算
# 公式：params ≈ 12 × n_layer × n_embd²
params = 12 × 6 × 384²
       = 12 × 6 × 147,456
       = 10,616,832
       ≈ 10.6M 参数

# 4. 显存使用估算
# 模型参数
memory_params = 10.6M × 4 bytes = 42.4 MB

# 梯度
memory_grads = 10.6M × 4 bytes = 42.4 MB

# 优化器状态（AdamW需要2倍参数）
memory_optimizer = 10.6M × 8 bytes = 84.8 MB

# 激活值（前向传播的中间结果）
memory_activations = batch_size × block_size × n_embd × n_layer × 4
                   = 64 × 256 × 384 × 6 × 4
                   = 150,994,944 bytes
                   ≈ 144 MB

# 总显存
total_memory = 42.4 + 42.4 + 84.8 + 144
             ≈ 314 MB

# 实际会更多（还有其他开销），大约需要 500MB-1GB
```

### ⏱️ 5.3 训练时间估算

```python
# 假设硬件：RTX 3060（12GB显存）

# 每步训练时间
time_per_step = 0.5 秒（经验值）

# 总训练时间
total_time = max_iters × time_per_step
           = 5000 × 0.5
           = 2500 秒
           ≈ 42 分钟

# 实际时间可能是 45-60 分钟（包括评估、保存等）
```

---

## 📚 第六部分：配置调优策略

### 🚦 6.1 显存不够怎么办？

**问题：训练时出现 `CUDA out of memory` 错误**

#### 解决方案（按优先级）

```python
# ===== 策略1：减小batch_size（最有效）=====
batch_size = 32  # 从64减到32
# 效果：显存减少约50%
# 代价：训练速度变慢

# ===== 策略2：减小block_size =====
block_size = 128  # 从256减到128
# 效果：显存减少约60%（因为是平方关系）
# 代价：模型看到的上下文变短

# ===== 策略3：使用梯度累积 =====
batch_size = 16
gradient_accumulation_steps = 4  # 等效batch_size=64
# 效果：显存减少75%，效果不变
# 代价：训练速度变慢

# ===== 策略4：减小模型 =====
n_layer = 4      # 从6减到4
n_embd = 256     # 从384减到256
# 效果：显存减少约60%
# 代价：模型能力下降

# ===== 策略5：使用混合精度 =====
dtype = 'float16'  # 从float32改为float16
# 效果：显存减少50%
# 代价：可能有精度损失（通常很小）

# ===== 策略6：减小vocab_size =====
# 如果是字符级模型，这个不能改
# 如果是词级模型，可以减小词汇表
```

#### 决策树

```
显存不足？
  ↓
首先尝试：减小batch_size到16
  ↓
还不够？使用梯度累积
  batch_size = 8
  gradient_accumulation_steps = 8
  ↓
还不够？减小block_size到128
  ↓
还不够？减小模型
  n_layer = 4
  n_embd = 256
  ↓
还不够？使用float16
  dtype = 'float16'
  ↓
还不够？考虑换更大显存的GPU
```

---

### ⚡ 6.2 训练太慢怎么办？

**问题：训练速度很慢，等不及**

#### 解决方案

```python
# ===== 策略1：启用编译（最有效）=====
compile = True
# 效果：提速1.5-2倍
# 代价：首次编译需要等待1-2分钟

# ===== 策略2：增大batch_size =====
batch_size = 128  # 从64增到128
# 效果：GPU利用率更高，提速约1.5倍
# 代价：需要更多显存

# ===== 策略3：减小评估频率 =====
eval_interval = 500  # 从250改到500
# 效果：减少评估时间
# 代价：看不到实时进度

# ===== 策略4：减小日志频率 =====
log_interval = 100  # 从10改到100
# 效果：减少打印开销
# 代价：看不到详细进度

# ===== 策略5：使用更小的模型先验证 =====
n_layer = 4
n_embd = 256
# 效果：快速验证想法
# 代价：模型能力下降
```

---

### 🎯 6.3 过拟合怎么办？

**问题：训练loss很低，但验证loss很高**

```python
训练loss: 0.5  ✅
验证loss: 2.0  ❌ 过拟合了！
```

#### 解决方案

```python
# ===== 策略1：增加dropout =====
dropout = 0.3  # 从0.2增到0.3
# 效果：随机关闭更多神经元，增强泛化
# 代价：训练loss会稍微升高

# ===== 策略2：增加weight_decay =====
weight_decay = 0.2  # 从0.1增到0.2
# 效果：更强的正则化
# 代价：训练loss会稍微升高

# ===== 策略3：减小模型 =====
n_layer = 4      # 从6减到4
n_embd = 256     # 从384减到256
# 效果：更小的模型，泛化更好
# 代价：模型能力下降

# ===== 策略4：获取更多数据 =====
# 这是最根本的解决方案
# 数据越多，过拟合越少

# ===== 策略5：提前停止 =====
# 当验证loss不再下降时停止训练
# 避免过度拟合训练集
```

---

### 📉 6.4 欠拟合怎么办？

**问题：训练loss和验证loss都很高**

```python
训练loss: 2.5  ❌
验证loss: 2.6  ❌ 欠拟合了！
```

#### 解决方案

```python
# ===== 策略1：增大模型 =====
n_layer = 12     # 从6增到12
n_embd = 768     # 从384增到768
# 效果：更强的模型能力
# 代价：需要更多显存和时间

# ===== 策略2：减小正则化 =====
dropout = 0.1    # 从0.2减到0.1
weight_decay = 0.05  # 从0.1减到0.05
# 效果：让模型更容易拟合
# 代价：可能过拟合

# ===== 策略3：增加训练时间 =====
max_iters = 10000  # 从5000增到10000
# 效果：让模型学得更充分
# 代价：时间更长

# ===== 策略4：增大learning_rate =====
learning_rate = 3e-3  # 从1e-3增到3e-3
# 效果：学习更快
# 代价：可能不稳定

# ===== 策略5：检查数据质量 =====
# 确保数据没有问题
# 确保数据量足够
```

---

## 📚 第七部分：实战配置模板

### 🎮 7.1 快速实验配置（5分钟见效果）

```python
# config/quick_experiment.py
# 目标：快速验证想法，看看能不能跑通

# 数据参数
batch_size = 16              # 小batch，节省显存
block_size = 64              # 短上下文，训练快
gradient_accumulation_steps = 1

# 模型参数（超小模型）
n_layer = 2                  # 只有2层
n_head = 2                   # 2个头
n_embd = 128                 # 128维
dropout = 0.0                # 不用dropout

# 训练参数
learning_rate = 1e-3
max_iters = 500              # 只训练500步
eval_interval = 100          # 频繁评估
eval_iters = 20

# 预期结果：5分钟内完成，loss应该能下降
```

### 🏫 7.2 学习配置（1小时高质量）

```python
# config/learning.py
# 目标：学习用，质量不错，时间可接受

# 数据参数
batch_size = 32
block_size = 128
gradient_accumulation_steps = 2  # 等效batch=64

# 模型参数（小模型）
n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.1

# 训练参数
learning_rate = 1e-3
max_iters = 3000
warmup_iters = 100
eval_interval = 250

# 预期结果：1小时完成，能生成不错的文本
```

### 🎯 7.3 标准配置（莎士比亚级别）

```python
# config/standard.py
# 目标：生成高质量文本，可以展示

# 数据参数
batch_size = 64
block_size = 256
gradient_accumulation_steps = 1

# 模型参数（中等模型）
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# 训练参数
learning_rate = 1e-3
max_iters = 5000
warmup_iters = 100
lr_decay_iters = 5000
min_lr = 1e-4

# 预期结果：3-4小时完成，高质量文本生成
```

### 🚀 7.4 生产配置（GPT-2级别）

```python
# config/production.py
# 目标：生产级质量，需要好GPU

# 数据参数
batch_size = 12              # GPT-2原始配置
block_size = 1024            # 长上下文
gradient_accumulation_steps = 40  # 等效batch=480

# 模型参数（GPT-2 124M）
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0                # GPT-2不用dropout

# 训练参数
learning_rate = 6e-4
max_iters = 600000           # 60万步
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

# 预期结果：需要多GPU，数天时间，专业级质量
```

---

## 📚 第八部分：实验建议

### 🔬 8.1 系统性实验

#### 实验1：理解batch_size的影响

```bash
# 创建实验脚本
cat > experiment_batch_size.sh << 'EOF'
#!/bin/bash

echo "实验1：batch_size=16"
python train.py config/train_shakespeare_char.py \
  --batch_size=16 \
  --max_iters=1000 \
  --out_dir=out_batch16

echo "实验2：batch_size=32"
python train.py config/train_shakespeare_char.py \
  --batch_size=32 \
  --max_iters=1000 \
  --out_dir=out_batch32

echo "实验3：batch_size=64"
python train.py config/train_shakespeare_char.py \
  --batch_size=64 \
  --max_iters=1000 \
  --out_dir=out_batch64

echo "对比结果："
echo "查看训练时间和最终loss"
EOF

chmod +x experiment_batch_size.sh
./experiment_batch_size.sh
```

#### 实验2：理解learning_rate的影响

```bash
# 学习率实验
python train.py config/train_shakespeare_char.py \
  --learning_rate=1e-4 --max_iters=1000 --out_dir=out_lr_1e4

python train.py config/train_shakespeare_char.py \
  --learning_rate=1e-3 --max_iters=1000 --out_dir=out_lr_1e3

python train.py config/train_shakespeare_char.py \
  --learning_rate=1e-2 --max_iters=1000 --out_dir=out_lr_1e2

# 观察：哪个收敛最好？哪个震荡？哪个太慢？
```

#### 实验3：理解模型大小的影响

```bash
# 小模型
python train.py config/train_shakespeare_char.py \
  --n_layer=2 --n_embd=128 --max_iters=2000 --out_dir=out_small

# 中模型
python train.py config/train_shakespeare_char.py \
  --n_layer=4 --n_embd=256 --max_iters=2000 --out_dir=out_medium

# 大模型
python train.py config/train_shakespeare_char.py \
  --n_layer=6 --n_embd=384 --max_iters=2000 --out_dir=out_large

# 对比：训练时间、最终loss、生成质量
```

---

## 🎓 总结与检查

### ✅ 知识检查清单

完成学习后，你应该能够：

**基础概念（必须掌握）**
- [ ] 解释batch_size、block_size、gradient_accumulation_steps的含义
- [ ] 理解n_layer、n_head、n_embd如何影响模型
- [ ] 知道learning_rate、weight_decay、grad_clip的作用
- [ ] 能够根据显存大小调整配置

**进阶理解（建议掌握）**
- [ ] 理解梯度累积的工作原理
- [ ] 知道为什么需要warmup和lr_decay
- [ ] 能够诊断过拟合和欠拟合
- [ ] 会设计实验验证参数影响

**实战能力（最终目标）**
- [ ] 能够为新项目选择合适的配置
- [ ] 会根据训练情况调整参数
- [ ] 能够解决常见的训练问题
- [ ] 会进行系统性的参数实验

### 📊 参数速查表

| 参数 | 增大效果 | 减小效果 | 推荐值 |
|------|---------|---------|--------|
| **batch_size** | 显存↑, 速度↑, 稳定↑ | 显存↓, 速度↓, 噪声↑ | 32-64 |
| **block_size** | 显存↑↑, 理解力↑ | 显存↓↓, 理解力↓ | 128-256 |
| **learning_rate** | 可能不收敛 | 训练太慢 | 1e-3 |
| **n_layer** | 能力↑, 时间↑↑ | 能力↓, 时间↓ | 4-6 |
| **n_embd** | 能力↑, 显存↑ | 能力↓, 显存↓ | 256-384 |
| **dropout** | 泛化↑, 拟合↓ | 泛化↓, 拟合↑ | 0.1-0.2 |
| **weight_decay** | 正则化↑ | 正则化↓ | 0.1 |
| **grad_clip** | 更保守 | 可能爆炸 | 1.0 |

### 🚀 下一步学习

现在你已经掌握了配置参数，接下来应该学习：

1. **02_data_loading_deep_dive.md** - 理解数据是如何加载的
2. **03_training_loop_deep_dive.md** - 理解训练循环的工作原理
3. **05_model_architecture_deep_dive.md** - 深入理解模型架构

### 💡 实践建议

1. **动手实验**：不要只看文档，一定要自己跑实验
2. **记录结果**：记录每次实验的配置和结果
3. **对比分析**：对比不同配置的效果
4. **循序渐进**：从小模型开始，逐步增大

---

## 📚 推荐资源

### 📖 延伸阅读
- [Andrej Karpathy的配置建议](https://github.com/karpathy/nanoGPT)
- [GPT-2论文](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Adam优化器论文](https://arxiv.org/abs/1412.6980)

### 🎥 视频教程
- [Andrej Karpathy: Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)

### 🔧 实用工具
- [显存计算器](https://huggingface.co/docs/transformers/model_memory_anatomy)
- [参数量计算器](https://huggingface.co/spaces/hf-accelerate/model-memory-usage)

---

**恭喜你完成第01章！** 🎉

你现在已经掌握了GPT训练的所有配置参数。这是训练模型的基础，接下来让我们深入理解数据加载的原理。

**准备好了吗？让我们继续前进！** → [02_data_loading_deep_dive.md](02_data_loading_deep_dive.md)
