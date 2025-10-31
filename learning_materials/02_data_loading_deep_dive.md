# 第02章：数据加载完全指南 - 从文本到张量

> **学习目标**：理解GPT如何读取和处理训练数据  
> **难度等级**：🌱 入门  
> **预计时间**：25-35分钟  
> **前置知识**：第01章配置参数

---

## 🎯 你将学到什么

学完本章，你将能够：
- ✅ 理解文本如何转换成数字
- ✅ 掌握`get_batch()`函数的工作原理
- ✅ 理解"预测下一个token"的本质
- ✅ 知道为什么要随机采样
- ✅ 了解memmap和pin_memory的优化技巧

---

## 💭 开始之前：为什么要学数据加载？

想象你要教一个学生学习：

```
❌ 错误的方式：
  - 一次性把所有书都给他（内存爆炸）
  - 按顺序一页一页地教（死记硬背）
  - 每次都重新拿书（效率低下）

✅ 正确的方式：
  - 需要哪页就拿哪页（内存映射）
  - 随机抽取不同章节（随机采样）
  - 提前准备好下一页（异步加载）
```

**数据加载就是"如何高效地喂数据给模型"！**

---

## 📚 第一部分：从文本到数字（基础）

### 🌱 1.1 原始文本是什么样的？

#### 莎士比亚原文

```
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?
```

这是人类能读懂的文本，但计算机不能直接处理文字，需要转换成数字。

---

### 🌱 1.2 字符级编码

#### 💡 直观理解

**步骤1：找出所有唯一字符**

```python
# 扫描整个莎士比亚文集，找出所有出现过的字符
所有字符 = " !',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# 统计
总共有 65 个不同的字符
```

#### 📊 建立映射表

```python
# 字符 → 整数（stoi: string to integer）
stoi = {
    ' ': 0,   # 空格
    '!': 1,   # 感叹号
    ',': 2,   # 逗号
    ...
    'A': 7,   # 大写A
    'B': 8,   # 大写B
    ...
    'a': 33,  # 小写a
    'b': 34,  # 小写b
    ...
    'z': 64   # 小写z
}

# 整数 → 字符（itos: integer to string）
itos = {
    0: ' ',
    1: '!',
    2: ',',
    ...
    64: 'z'
}
```

#### 🎬 编码过程

```python
# 原始文本
text = "First Citizen"

# 编码（查表）
encoded = []
for char in text:
    encoded.append(stoi[char])

# 结果
"First Citizen"
↓
[20, 47, 53, 55, 56, 1, 19, 47, 56, 47, 58, 43, 52]
 F   i   r   s   t  空格 C   i   t   i   z   e   n
```

---

### 🌱 1.3 保存为二进制文件

#### 为什么要保存？

```python
# 问题：每次训练都重新编码？
# 莎士比亚全集有1,115,394个字符
# 每次编码需要1-2秒
# 训练10000次 = 10000-20000秒 = 3-6小时浪费！

# 解决：编码一次，保存下来
# 以后直接读取，只需0.001秒
```

#### 📁 文件结构

```python
data/shakespeare_char/
├── train.bin       # 训练数据（90%）
├── val.bin         # 验证数据（10%）
└── meta.pkl        # 元数据（stoi, itos映射）

# train.bin 内容（二进制格式）
[20, 47, 53, 55, 56, 1, 19, 47, 56, 47, 58, 43, 52, ...]
 F   i   r   s   t  空格 C   i   t   i   z   e   n  ...

# 大小：1,003,854 个数字
# 存储：每个数字用uint16（2字节）
# 总大小：约2MB
```

---

## 📚 第二部分：get_batch()深度解析（核心）

### 🌿 2.1 完整代码

```python
def get_batch(split):
    """
    获取一个批次的训练数据
    
    参数:
        split: 'train' 或 'val'
    
    返回:
        x: 输入张量 [batch_size, block_size]
        y: 目标张量 [batch_size, block_size]
    """
    # 步骤1: 使用内存映射打开文件
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), 
                        dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), 
                        dtype=np.uint16, mode='r')
    
    # 步骤2: 随机选择起始位置
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # 步骤3: 提取输入和目标
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) 
                     for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) 
                     for i in ix])
    
    # 步骤4: 移动到GPU
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), \
               y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    
    return x, y
```

---

### 🌿 2.2 逐步详解

#### 步骤1：内存映射（memmap）

**💡 什么是memmap？**

想象图书馆借书：

```
方法A：全部搬回家（传统加载）
  - 把整个图书馆的书都搬回家
  - 优点：随时可以看
  - 缺点：家里放不下！

方法B：办借书证（memmap）
  - 需要哪本书就去借哪本
  - 优点：不占用家里空间
  - 缺点：需要去图书馆（但很快！）
```

**📊 技术对比**

```python
# 方法A：全部加载到内存
data = np.fromfile('train.bin', dtype=np.uint16)
# 内存占用：2MB（小文件还好）
# 如果是100GB的文件呢？内存爆炸！

# 方法B：内存映射
data = np.memmap('train.bin', dtype=np.uint16, mode='r')
# 内存占用：几乎为0
# 操作系统会按需加载（智能缓存）
# 访问速度：几乎和内存一样快！
```

**🎯 为什么快？**

```
操作系统的魔法：
1. 第一次访问：从硬盘读取 → 缓存到内存
2. 第二次访问：直接从内存读取（超快！）
3. 内存不够：自动释放不常用的部分

结果：
- 可以处理任意大小的文件
- 速度接近内存访问
- 多进程可以共享同一个文件
```

---

#### 步骤2：随机采样

**💡 为什么要随机？**

```python
# 假设配置
batch_size = 4
block_size = 8
len(data) = 1,000,000

# 生成4个随机起始位置
ix = torch.randint(len(data) - block_size, (batch_size,))
# 例如：ix = [137, 5894, 23091, 88234]
```

**🎲 为什么是 `len(data) - block_size`？**

```python
# 边界检查
len(data) = 1,000,000
block_size = 8

# 如果从位置999,995开始
data[999995:999995+8] = data[999995:1000003]
                                    ↑
                                超出范围了！

# 所以最大起始位置是
max_start = len(data) - block_size
          = 1,000,000 - 8
          = 999,992
```

**📊 顺序 vs 随机**

```python
# 方法A：顺序采样（不好）
batch 1: 位置 [0,    1,    2,    3]
batch 2: 位置 [4,    5,    6,    7]
batch 3: 位置 [8,    9,   10,   11]

问题：
- 模型可能学到batch之间的顺序关系
- 容易过拟合
- 泛化能力差

# 方法B：随机采样（好）✅
batch 1: 位置 [2341, 8123, 9987,  123]
batch 2: 位置 [5432, 1245, 7890, 3456]
batch 3: 位置 [9012, 4567, 2345, 6789]

优点：
- 每次看到的都是新鲜组合
- 不会过拟合
- 泛化能力强
```

---

#### 步骤3：构造输入输出对

**💡 核心：y = x 向右移一位**

这是"预测下一个token"任务的本质！

```python
# 从每个起始位置提取数据
for i in ix:
    x[i] = data[i:i+block_size]      # 输入
    y[i] = data[i+1:i+1+block_size]  # 目标（右移1位）
```

**🎬 具体例子**

```python
# 假设从位置137开始
i = 137

# 提取输入
x = data[137:145]  # 8个字符
  = [20, 43, 1, 60, 43, 1, 52, 47]
  = "he we ni"

# 提取目标（右移1位）
y = data[138:146]  # 从138开始！
  = [43, 1, 60, 43, 1, 52, 47, 56]
  = "e we nig"

# 对比
x: [20, 43,  1, 60, 43,  1, 52, 47]  "he we ni"
y: [43,  1, 60, 43,  1, 52, 47, 56]  "e we nig"
    ↑   ↑   ↑   ↑   ↑   ↑   ↑   ↑
    整体向右移动一位
```

**🎯 预测任务**

```python
# 模型要学习的是：
位置0: 看到 "h" (20) → 预测 "e" (43)
位置1: 看到 "e" (43) → 预测 " " (1)
位置2: 看到 " " (1)  → 预测 "w" (60)
位置3: 看到 "w" (60) → 预测 "e" (43)
...

# 可视化
输入:  [h] [e] [ ] [w] [e] [ ] [n] [i]
        ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓
目标:  [e] [ ] [w] [e] [ ] [n] [i] [g]
```

---

#### 步骤4：移动到GPU

**💡 为什么需要pin_memory？**

```python
# 普通方式（慢）
x = x.to(device)
# CPU → GPU：15ms

# 优化方式（快）
x = x.pin_memory().to(device, non_blocking=True)
# CPU → GPU：3ms
# 加速：5倍！
```

**🔧 技术原理**

```
普通内存（Pageable Memory）：
  - CPU可以随时移动和交换
  - GPU传输需要先复制到临时缓冲区
  - 慢！

锁定内存（Pinned Memory）：
  - 锁定在RAM中，不会被交换
  - GPU可以直接访问（DMA传输）
  - 不需要CPU参与
  - 快！

non_blocking=True：
  - 异步传输
  - CPU不用等待传输完成
  - 可以继续做其他事情
```

---

## 📚 第三部分：完整示例（实战）

### 🌳 3.1 手动模拟get_batch()

让我们用具体数字走一遍完整流程：

```python
# 配置
batch_size = 4      # 一次处理4个样本
block_size = 8      # 每个样本8个字符

# 原始数据（简化表示）
data = "First Citizen: Before we proceed any further, hear me speak."
# 编码后
data = [20, 47, 53, 55, 56, 1, 19, 47, 56, 47, 58, 43, 52, 27, 1, 18, ...]
#       F   i   r   s   t  空格 C   i   t   i   z   e   n   :  空格 B  ...
# 索引: 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  ...
```

#### 步骤1：随机采样

```python
# 生成4个随机起始位置
ix = torch.randint(len(data) - block_size, (batch_size,))
# 假设得到：ix = [0, 6, 14, 20]
```

#### 步骤2：提取数据

```python
# 样本0：从位置0开始
x[0] = data[0:8]   = [20, 47, 53, 55, 56, 1, 19, 47]  # "First Ci"
y[0] = data[1:9]   = [47, 53, 55, 56, 1, 19, 47, 56]  # "irst Cit"

# 样本1：从位置6开始
x[1] = data[6:14]  = [19, 47, 56, 47, 58, 43, 52, 27]  # "Citizen:"
y[1] = data[7:15]  = [47, 56, 47, 58, 43, 52, 27, 1]   # "itizen: "

# 样本2：从位置14开始
x[2] = data[14:22] = [1, 18, 43, 44, 53, 43, 1, 60]    # " Before "
y[2] = data[15:23] = [18, 43, 44, 53, 43, 1, 60, 43]   # "Before w"

# 样本3：从位置20开始
x[3] = data[20:28] = [53, 43, 1, 60, 43, 1, 54, 53]    # "re we pr"
y[3] = data[21:29] = [43, 1, 60, 43, 1, 54, 53, 53]    # "e we pro"
```

#### 步骤3：组装成张量

```python
# 最终形状
x.shape = torch.Size([4, 8])  # [batch_size, block_size]
y.shape = torch.Size([4, 8])

# 具体数值
x = tensor([
    [20, 47, 53, 55, 56,  1, 19, 47],  # "First Ci"
    [19, 47, 56, 47, 58, 43, 52, 27],  # "Citizen:"
    [ 1, 18, 43, 44, 53, 43,  1, 60],  # " Before "
    [53, 43,  1, 60, 43,  1, 54, 53],  # "re we pr"
])

y = tensor([
    [47, 53, 55, 56,  1, 19, 47, 56],  # "irst Cit"
    [47, 56, 47, 58, 43, 52, 27,  1],  # "itizen: "
    [18, 43, 44, 53, 43,  1, 60, 43],  # "Before w"
    [43,  1, 60, 43,  1, 54, 53, 53],  # "e we pro"
])
```

---

### 🌳 3.2 实战：调试脚本

创建一个脚本来查看实际的batch：

```python
# debug_get_batch.py
import os
import pickle
import numpy as np
import torch

# ===== 配置 =====
data_dir = 'data/shakespeare_char'
batch_size = 2
block_size = 20

# ===== 加载元数据 =====
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
itos = meta['itos']  # 整数到字符的映射

# ===== 加载数据 =====
data = np.memmap(os.path.join(data_dir, 'train.bin'), 
                 dtype=np.uint16, mode='r')

print(f"数据集大小: {len(data):,} 个字符")
print(f"Batch大小: {batch_size}")
print(f"Block大小: {block_size}")
print()

# ===== 获取一个batch =====
ix = torch.randint(len(data) - block_size, (batch_size,))
print(f"随机起始位置: {ix.tolist()}\n")

x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) 
                 for i in ix])
y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) 
                 for i in ix])

# ===== 打印每个样本 =====
for i in range(batch_size):
    print("=" * 70)
    print(f"样本 {i + 1}")
    print("=" * 70)
    
    # 解码成文本
    x_text = ''.join([itos[int(idx)] for idx in x[i]])
    y_text = ''.join([itos[int(idx)] for idx in y[i]])
    
    print(f"输入 (x): '{x_text}'")
    print(f"目标 (y): '{y_text}'")
    print()
    
    # 打印前5个预测任务
    print("预测任务（前5个）：")
    for j in range(min(5, len(x[i]))):
        input_char = itos[int(x[i][j])]
        target_char = itos[int(y[i][j])]
        print(f"  位置{j}: 看到 '{input_char}' → 预测 '{target_char}'")
    print()
```

**运行结果示例：**

```bash
$ python debug_get_batch.py

数据集大小: 1,003,854 个字符
Batch大小: 2
Block大小: 20

随机起始位置: [45123, 892345]

======================================================================
样本 1
======================================================================
输入 (x): 'the king to the re t'
目标 (y): 'he king to the re th'

预测任务（前5个）：
  位置0: 看到 't' → 预测 'h'
  位置1: 看到 'h' → 预测 'e'
  位置2: 看到 'e' → 预测 ' '
  位置3: 看到 ' ' → 预测 'k'
  位置4: 看到 'k' → 预测 'i'

======================================================================
样本 2
======================================================================
输入 (x): 'Before we proceed An'
目标 (y): 'efore we proceed Any'

预测任务（前5个）：
  位置0: 看到 'B' → 预测 'e'
  位置1: 看到 'e' → 预测 'f'
  位置2: 看到 'f' → 预测 'o'
  位置3: 看到 'o' → 预测 'r'
  位置4: 看到 'r' → 预测 'e'
```

---

## 📚 第四部分：性能优化（高级）

### 🌳 4.1 数据加载速度分析

```python
import time
import numpy as np

# 测试：memmap vs 全加载
data_file = 'data/shakespeare_char/train.bin'

# 方法1：memmap
t0 = time.time()
data_memmap = np.memmap(data_file, dtype=np.uint16, mode='r')
for _ in range(1000):
    batch = data_memmap[np.random.randint(0, len(data_memmap)-1000, 100)]
t1 = time.time()
print(f"memmap方式: {t1-t0:.3f}秒")

# 方法2：全加载
t0 = time.time()
data_full = np.fromfile(data_file, dtype=np.uint16)
for _ in range(1000):
    batch = data_full[np.random.randint(0, len(data_full)-1000, 100)]
t1 = time.time()
print(f"全加载方式: {t1-t0:.3f}秒")

# 结果（示例）：
# memmap方式: 0.234秒
# 全加载方式: 0.221秒（略快，但占用2MB内存）

# 结论：
# - 小文件：差别不大
# - 大文件（>1GB）：memmap明显优势
# - 多进程：memmap可以共享，全加载不行
```

---

### 🌳 4.2 训练时间分配

```python
# 典型训练迭代的时间分配

总时间: 100ms/iter
├─ 数据加载: 5ms (5%)      ← get_batch()
├─ GPU前向传播: 30ms (30%)
├─ GPU反向传播: 40ms (40%)
├─ 参数更新: 10ms (10%)
└─ 日志记录: 15ms (15%)

结论：
✅ 数据加载不是瓶颈
✅ GPU计算占主要时间
✅ 优化重点应该在模型和算法
```

---

### 🌳 4.3 异步数据加载

**💡 核心思想：CPU和GPU并行工作**

```python
# train.py 中的实现（简化）

for iter in range(max_iters):
    # 关键：在GPU计算时，CPU准备下一个batch
    X, Y = get_batch('train')  # CPU: 5ms
    
    with ctx:
        logits, loss = model(X, Y)  # GPU: 70ms
        # 同时，CPU可以准备下一个batch
    
    loss.backward()  # GPU: 40ms
    optimizer.step()  # GPU: 10ms
```

**📊 时间线对比**

```
传统方式（串行）：
  CPU: 加载batch₁ (5ms)
  GPU: 计算batch₁ (120ms)  ← CPU空闲
  CPU: 加载batch₂ (5ms)    ← GPU空闲
  GPU: 计算batch₂ (120ms)
  
  总时间: (5 + 120) × 2 = 250ms

NanoGPT方式（并行）：
  CPU: 加载batch₁ (5ms)
  GPU: 计算batch₁ (120ms) | CPU: 加载batch₂ (5ms)
  GPU: 计算batch₂ (120ms) | CPU: 加载batch₃ (5ms)
  
  总时间: 5 + 120 × 2 = 245ms
  
节省: 5ms per batch
```

---

### 🌳 4.4 pin_memory详解

```python
# pin_memory的作用

# 方法1：普通传输
x = x.to(device)
# 步骤：
#   1. CPU分配临时缓冲区
#   2. 复制数据到缓冲区
#   3. GPU从缓冲区读取
#   4. 释放缓冲区
# 时间：15ms

# 方法2：锁定内存传输
x = x.pin_memory().to(device, non_blocking=True)
# 步骤：
#   1. 数据已在锁定内存中
#   2. GPU直接读取（DMA）
# 时间：3ms

# 加速：5倍！
```

**⚠️ 注意事项**

```python
# 优点：
✅ 传输速度快5倍
✅ CPU不用等待
✅ 可以异步执行

# 缺点：
❌ 占用更多内存（不能被交换）
❌ 分配速度稍慢

# 建议：
- 训练时使用：✅
- 推理时使用：✅
- 内存紧张时：❌ 不用
```

---

## 🎓 总结与检查

### ✅ 知识检查清单

完成学习后，你应该能够：

**基础概念（必须掌握）**
- [ ] 理解文本如何编码成数字
- [ ] 知道stoi和itos的作用
- [ ] 理解为什么y=x右移一位
- [ ] 能解释随机采样的好处
- [ ] 知道memmap的优势

**进阶理解（建议掌握）**
- [ ] 理解pin_memory的作用
- [ ] 知道异步数据加载的原理
- [ ] 能计算数据加载的时间开销
- [ ] 理解为什么数据加载不是瓶颈

**实战能力（最终目标）**
- [ ] 能写出调试脚本查看batch内容
- [ ] 会分析数据加载的性能
- [ ] 能优化数据加载流程
- [ ] 理解整个数据流动过程

### 📊 核心要点总结

```python
get_batch()的四大职责：

1. 高效读取 (memmap)
   - 不占用内存
   - 按需加载
   - 速度快

2. 随机采样 (randint)
   - 避免过拟合
   - 增强泛化
   - 每次都是新组合

3. 构造训练对 (x, y)
   - y = x右移1位
   - 预测下一个token
   - 自监督学习

4. 异步传输 (pin_memory)
   - CPU和GPU并行
   - 减少等待时间
   - 提升效率
```

### 🚀 下一步学习

现在你已经理解了数据如何加载，接下来应该学习：

1. **03_training_loop_deep_dive.md** - 训练循环如何工作
2. **05_model_architecture_deep_dive.md** - 模型如何处理数据
3. **实战练习** - 运行debug脚本，查看实际数据

### 💡 实践建议

1. **运行调试脚本**：看看真实的batch是什么样的
2. **修改参数**：尝试不同的batch_size和block_size
3. **性能测试**：对比memmap和全加载的速度
4. **可视化**：画出数据流动的完整过程

---

## 📚 推荐资源

### 📖 延伸阅读
- [NumPy memmap文档](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html)
- [PyTorch数据加载教程](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
- [pin_memory详解](https://pytorch.org/docs/stable/notes/cuda.html#use-pinned-memory-buffers)

### 🎥 视频教程
- [Andrej Karpathy: 数据预处理](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=1200s)

### 🔧 实用工具
- [Tokenizer可视化](https://tiktokenizer.vercel.app/)

---

## 🐛 常见问题

### Q1: 为什么不用PyTorch的DataLoader？

```python
# DataLoader的优势：
✅ 多进程加载
✅ 自动batching
✅ 丰富的采样策略

# 但对于我们的场景：
❌ 过于复杂（我们只需要简单的随机采样）
❌ 开销大（创建进程、通信等）
❌ 不够灵活（memmap更适合大文件）

# NanoGPT的get_batch()：
✅ 简单（20行代码）
✅ 高效（性能接近DataLoader）
✅ 灵活（容易修改和调试）
```

### Q2: 如果数据太大，内存放不下怎么办？

```python
# 这正是memmap的优势！

# 例如：100GB的数据集
data = np.memmap('huge_data.bin', dtype=np.uint16, mode='r')
# 内存占用：几乎为0
# 操作系统会智能缓存常用部分

# 实际使用：
batch = data[random_index:random_index+block_size]
# 只加载需要的部分到内存
```

### Q3: 为什么是字符级而不是词级？

```python
# 字符级（NanoGPT使用）：
✅ 词汇表小（65个字符）
✅ 不会有未知词
✅ 可以生成任意词
❌ 序列更长

# 词级（GPT-2/GPT-3使用）：
✅ 序列更短
✅ 训练更快
❌ 词汇表大（50K+）
❌ 有未知词问题

# 选择：
- 学习/实验：字符级 ✅
- 生产应用：词级（BPE）✅
```

---

**恭喜你完成第02章！** 🎉

你现在已经掌握了数据加载的完整流程。数据是模型的"食物"，理解数据加载是理解整个训练过程的关键一步。

**准备好了吗？让我们继续前进！** → [03_training_loop_deep_dive.md](03_training_loop_deep_dive.md)
