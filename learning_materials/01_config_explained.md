# 配置参数详解 - 初学者指南

## 1. 数据相关参数

### batch_size (批次大小)

**是什么？**
每次训练时，同时给模型看多少个样本。

**生活比喻：**
- batch_size = 1：老师一次只教一个学生（慢，但精确）
- batch_size = 64：老师一次教64个学生（快，但平均化）

**具体例子：**
```python
batch_size = 4
block_size = 8  # 每个样本8个字符

# 训练数据可能长这样：
样本1: "To be or" → 预测 "not"
样本2: "not to b" → 预测 "e"
样本3: "be that " → 预测 "is"
样本4: "is the q" → 预测 "u"

# 模型一次性处理这4个样本
```

**数值影响：**
| batch_size | 训练速度 | 显存使用 | 梯度噪声 | 建议场景 |
|-----------|---------|---------|---------|----------|
| 1-8       | 慢      | 低      | 高      | 调试/小数据集 |
| 16-32     | 中      | 中      | 中      | **初学者推荐** |
| 64-128    | 快      | 高      | 低      | 大规模训练 |

### block_size (上下文长度)

**是什么？**
模型一次能看到多少个字符/词。

**生活比喻：**
- block_size = 8：只记得最近8个词（健忘）
- block_size = 256：能记得256个词（记忆力好）
- block_size = 1024：能记得1024个词（过目不忘）

**实际例子：**

假设有句子："The quick brown fox jumps over the lazy dog"

```python
# block_size = 4
输入: "The quick brown" → 只看前4个词，预测第5个
模型看不到："fox jumps over the lazy dog"

# block_size = 10
输入: "The quick brown fox jumps over the lazy" → 看前10个词
模型可以看到更多上下文！
```

**权衡：**
```
block_size ↑ → 显存使用 ↑↑ (平方增长!)
           → 训练时间 ↑↑
           → 模型理解力 ↑

计算复杂度 = O(block_size²) 因为self-attention机制
```

### gradient_accumulation_steps (梯度累积)

**这是最难理解的概念！让我用详细例子说明：**

**问题背景：**
你的显卡只有8GB显存，但你想用相当于32GB显存的batch size。

**解决方案：分期付款训练法**

```python
# 方法1：直接用大batch（显存爆炸！）
batch_size = 64  # 需要32GB显存 ❌

# 方法2：梯度累积（巧妙！）
batch_size = 16  # 只需8GB显存
gradient_accumulation_steps = 4  # 累积4次
# 等效batch_size = 16 × 4 = 64 ✅
```

**详细过程模拟：**

```
=== 传统训练 (batch_size=64) ===
步骤1: 读取64个样本 → 前向传播 → 计算loss → 反向传播 → 更新参数
显存: ████████████████ (16GB) ❌ 显存不够！

=== 梯度累积训练 ===
步骤1: 读取16个样本 → 前向 → 反向 → 梯度暂存 (不更新参数)
显存: ████ (4GB)

步骤2: 读取16个样本 → 前向 → 反向 → 梯度累加
显存: ████ (4GB)
累积梯度 = 梯度1 + 梯度2

步骤3: 读取16个样本 → 前向 → 反向 → 梯度累加
显存: ████ (4GB)
累积梯度 = 梯度1 + 梯度2 + 梯度3

步骤4: 读取16个样本 → 前向 → 反向 → 梯度累加 → 更新参数！
显存: ████ (4GB)
累积梯度 = 梯度1 + 梯度2 + 梯度3 + 梯度4

效果：等同于batch_size=64，但显存只用了4GB！
```

**数学原理：**
```
平均梯度 = (梯度1 + 梯度2 + 梯度3 + 梯度4) / 4
等价于一次性计算64个样本的平均梯度
```

---

## 2. 模型架构参数

### n_layer (层数)

**是什么？**
模型有多少层Transformer块。

**生活比喻：**
```
1层：小学生的理解力
6层：高中生的理解力（莎士比亚模型用这个）
12层：大学生的理解力（GPT-2用这个）
96层：博士的理解力（GPT-3用这个）
```

**具体影响：**
```python
n_layer = 6   # 参数量：~40M，训练时间：3分钟
n_layer = 12  # 参数量：~124M，训练时间：4天
n_layer = 24  # 参数量：~350M，训练时间：2周
```

### n_head (注意力头数)

**是什么？**
模型同时关注多少个不同的方面。

**生活比喻：阅读理解的多个角度**
```
n_head = 1：只从一个角度看问题
  "To be or not to be"
  只关注：语法结构

n_head = 6：从6个角度看问题
  角度1：语法结构
  角度2：语义关系
  角度3：韵律节奏
  角度4：情感色彩
  角度5：上下文连贯
  角度6：风格特征
```

**实际计算：**
```python
n_embd = 384  # 总维度
n_head = 6    # 6个头
每个头的维度 = 384 / 6 = 64

# 每个头独立处理一部分信息
头1处理：维度0-63
头2处理：维度64-127
头3处理：维度128-191
...
最后合并所有头的结果
```

### n_embd (嵌入维度)

**是什么？**
用多少个数字来表示一个词。

**类比：描述一个人**
```
n_embd = 2：(身高, 体重)  
  → 信息太少，无法区分很多人

n_embd = 384：(身高, 体重, 年龄, 肤色, 发色, ..., 384个特征)
  → 信息丰富，可以精确描述

n_embd = 768：(GPT-2用这个)
  → 可以捕捉非常细微的差异
```

**存储空间：**
```python
vocab_size = 50000  # 词汇表大小
n_embd = 768

# 嵌入层参数量
embedding_params = vocab_size × n_embd 
                 = 50000 × 768 
                 = 38,400,000 个参数
                 ≈ 150MB (float32)
```

---

## 3. 优化器参数

### learning_rate (学习率)

**最重要的超参数！**

**生活比喻：学习的步伐**
```
learning_rate = 1.0：  
  学生学得太激进，一会儿全懂，一会儿全忘
  结果：无法收敛

learning_rate = 0.001：
  学生学得稳健，每次进步一点点
  结果：✅ 稳定学习

learning_rate = 0.000001：
  学生学得太慢，一辈子都学不完
  结果：训练太慢
```

**可视化：**
```
Loss (损失)
  ^
  |  lr太大 → 震荡
  |    /\  /\  /\
  |   /  \/  \/  \
  |  
  |  lr刚好 → 平滑下降
  |  \
  |   \___
  |       \___
  |           \___
  |
  |  lr太小 → 下降太慢
  |  \
  |   \
  |    \
  |     \
  +------------------------> 训练步数
```

**经验值：**
```python
# 从零训练
learning_rate = 1e-3  # 0.001，小模型
learning_rate = 6e-4  # 0.0006，GPT-2

# 微调预训练模型
learning_rate = 1e-4  # 0.0001，避免破坏已学知识
learning_rate = 5e-5  # 0.00005，更保守
```

### weight_decay (权重衰减)

**是什么？**
防止模型"死记硬背"的机制。

**生活比喻：**
```
没有weight_decay：
  学生把每个例子都死记硬背
  "To be" → "or not to be"
  "To go" → ？（不知道，没背过）
  结果：过拟合

有weight_decay：
  学生理解了规律，不是死记
  学到："To" 后面通常跟动词原形
  "To be" → "or not to be"
  "To go" → "or not to go" ✅
  结果：泛化能力强
```

**数学原理：**
```python
# 每次更新参数时
新参数 = 旧参数 - learning_rate × 梯度 - weight_decay × 旧参数
                                        ↑
                                这一项让参数不要太大
```

### grad_clip (梯度裁剪)

**是什么？**
防止梯度爆炸。

**生活比喻：**
```
没有grad_clip：
  老师突然大喊："你全错了！！！从头学！！！"
  学生：😱 吓傻了，忘记之前学的
  结果：梯度爆炸，训练崩溃

有grad_clip：
  老师温和说："你有些错误，慢慢改进"
  学生：😊 稳步改进
  结果：训练稳定
```

**具体实现：**
```python
grad_clip = 1.0

# 假设计算出的梯度很大
gradient = [100, 200, 300]  # 梯度爆炸了！
gradient_norm = sqrt(100² + 200² + 300²) = 374

# 裁剪
if gradient_norm > grad_clip:
    gradient = gradient * (grad_clip / gradient_norm)
    gradient = [0.27, 0.53, 0.80]  # 被缩小了
```

---

## 4. 学习率调度

### warmup_iters (预热步数)

**为什么需要warmup？**

**没有warmup的问题：**
```
训练开始：
  步骤1: 参数随机初始化
  步骤2: 用大learning_rate更新
  步骤3: 参数剧烈变化 💥
  步骤4: 模型崩溃
```

**有warmup：**
```
训练开始：
  步骤1-100: learning_rate从0慢慢增加到0.001
    步骤1:  lr = 0.00001  (小心翼翼)
    步骤50: lr = 0.0005   (逐渐加速)
    步骤100: lr = 0.001   (达到正常速度)
  
  步骤100+: 开始正常训练
```

**曲线图：**
```
Learning Rate
  ^
  |          /‾‾‾‾‾‾‾‾\
  |         /           \
  |        /             \
  |       /               \___
  |      /                    \___
  |     /                         \___
  |    /                              \___
  +-------|---------|--------------|---------> Steps
        warmup    正常训练        decay
       (2000)                   (600000)
```

---

## 5. 实战计算示例

让我们用莎士比亚配置计算一次完整的训练迭代：

```python
# 配置（来自 train_shakespeare_char.py）
batch_size = 64
block_size = 256
gradient_accumulation_steps = 1
vocab_size = 65  # 莎士比亚数据集有65个不同字符

# 一次迭代处理的token数量
tokens_per_iter = batch_size × block_size × gradient_accumulation_steps
                = 64 × 256 × 1
                = 16,384 tokens

# 显存使用估算（简化）
# 假设每个参数 + 梯度 + 优化器状态 ≈ 12 bytes
model_params = 10_000_000  # 10M参数的小模型
memory_model = 10_000_000 × 12 = 120MB

# 激活值（前向传播的中间结果）
memory_activation = batch_size × block_size × n_embd × n_layer × 4
                  = 64 × 256 × 384 × 6 × 4 bytes
                  = 150MB

# 总显存 ≈ 270MB (实际会更多)
```

---

## 6. 配置调优策略

### 🚦 显存不够怎么办？

**优先级从高到低：**

```python
# 策略1: 减小batch_size（最有效）
batch_size = 32  # 减半 → 显存减半

# 策略2: 减小block_size
block_size = 128  # 减半 → 显存减少60%

# 策略3: 使用梯度累积
gradient_accumulation_steps = 2  # 显存减半，效果不变

# 策略4: 减小模型
n_layer = 4
n_embd = 256

# 策略5: 使用混合精度
dtype = 'float16'  # 显存减半
```

### ⚡ 训练太慢怎么办？

```python
# 策略1: 启用编译（最有效）
compile = True  # 提速1.5-2x

# 策略2: 增大batch_size
batch_size = 128  # GPU利用率更高

# 策略3: 减小eval_interval
eval_interval = 500  # 少做评估，专心训练

# 策略4: 关闭一些日志
log_interval = 100  # 减少打印
```

### 🎯 过拟合怎么办？

```python
# 策略1: 增加dropout
dropout = 0.2  # 随机关闭20%的神经元

# 策略2: 增加weight_decay
weight_decay = 0.1  # 更强的正则化

# 策略3: 减小模型
n_layer = 4  # 更小的模型，泛化更好

# 策略4: 更多数据
# 获取更多训练数据！
```

---

## 🧪 实验建议

创建一个实验脚本来理解每个参数的影响：

```bash
# 实验1：batch_size的影响
python train.py config/train_shakespeare_char.py --batch_size=16 --max_iters=1000
python train.py config/train_shakespeare_char.py --batch_size=32 --max_iters=1000
python train.py config/train_shakespeare_char.py --batch_size=64 --max_iters=1000
# 观察：训练速度和最终loss

# 实验2：learning_rate的影响
python train.py config/train_shakespeare_char.py --learning_rate=1e-4 --max_iters=1000
python train.py config/train_shakespeare_char.py --learning_rate=1e-3 --max_iters=1000
python train.py config/train_shakespeare_char.py --learning_rate=1e-2 --max_iters=1000
# 观察：是否收敛

# 实验3：模型大小的影响
python train.py config/train_shakespeare_char.py --n_layer=2 --n_embd=128 --max_iters=2000
python train.py config/train_shakespeare_char.py --n_layer=4 --n_embd=256 --max_iters=2000
python train.py config/train_shakespeare_char.py --n_layer=6 --n_embd=384 --max_iters=2000
# 观察：模型能力和训练时间
```

---

## 📊 总结表格

| 参数 | 增大后果 | 减小后果 | 初学者推荐值 |
|------|---------|---------|-------------|
| batch_size | 显存↑, 速度↑, 噪声↓ | 显存↓, 速度↓, 噪声↑ | 32-64 |
| block_size | 显存↑↑, 理解力↑ | 显存↓↓, 理解力↓ | 128-256 |
| learning_rate | 不收敛 | 训练慢 | 1e-3 (小模型) |
| n_layer | 能力↑, 时间↑↑ | 能力↓, 时间↓ | 4-6 |
| n_embd | 能力↑, 显存↑ | 能力↓, 显存↓ | 256-384 |
| dropout | 泛化↑, 拟合↓ | 泛化↓, 拟合↑ | 0.1-0.2 |

