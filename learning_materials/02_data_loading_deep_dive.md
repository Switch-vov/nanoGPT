# 数据加载深度解析 - get_batch() 函数

## 🎯 核心问题：神经网络如何"吃"数据？

让我们从最基础的开始。

---

## 第一步：数据是如何准备的？

### 原始文本 → 数字序列

**原始莎士比亚文本：**
```
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.
```

**字符级编码：**
```python
# 1. 找出所有唯一字符
所有字符: " !',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
总共: 65个字符

# 2. 建立映射表
stoi = {' ': 0, '!': 1, ',': 2, ...,'z': 64}
itos = {0: ' ', 1: '!', 2: ',', ...  64: 'z'}

# 3. 编码
"First" → [20, 47, 53, 55, 56]
```

**保存为二进制文件：**
```python
# train.bin 文件内容（简化表示）
[20, 47, 53, 55, 56, 1, 19, 47, 56, 47, 58, 43, 52, ...]
   F   i   r   s   t  (space) C  i  t  i  z  e  n  ...

# 大小: 1,003,854 个数字 (约1MB)
```

---

## 第二步：get_batch() 逐行解析

```python
def get_batch(split):
    # 1. 使用内存映射打开文件（不全部加载到内存）
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), 
                        dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), 
                        dtype=np.uint16, mode='r')
    
    # 2. 随机选择起始位置
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # 3. 提取输入和目标
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) 
                     for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) 
                     for i in ix])
    
    # 4. 移动到GPU
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), \
               y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    
    return x, y
```

---

## 🔍 详细示例：手动模拟 get_batch()

假设我们有以下设置：
```python
batch_size = 4      # 一次处理4个样本
block_size = 8      # 每个样本长度为8
```

### 第1步：原始数据

```python
# train.bin 的内容（用字符表示，便于理解）
data = "First Citizen: Before we proceed any further, hear me speak."

# 编码后的数字（简化，实际是0-64）
data = [20, 47, 53, 55, 56, 1, 19, 47, 56, 47, 58, 43, 52, 27, 1, 18, ...]
        F   i   r   s   t     C   i   t   i   z   e   n   :     B   e  ...
索引:   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  ...
```

### 第2步：随机采样

```python
# 执行：ix = torch.randint(len(data) - block_size, (batch_size,))
len(data) = 1000000
len(data) - block_size = 1000000 - 8 = 999992

# 随机生成4个起始位置
ix = [137, 5894, 23091, 88234]

# 为什么要 len(data) - block_size？
# 因为从位置i开始，要读取i到i+block_size，不能超出文件末尾
```

### 第3步：构造输入输出对

从每个起始位置提取数据：

```python
# 位置137开始
样本0_输入 (x[0]): data[137:145]  = [20, 43, 1, 60, 43, 1, 52, 47]  # "he we ni"
样本0_输出 (y[0]): data[138:146]  = [43, 1, 60, 43, 1, 52, 47, 56]  # "e we nig"
                                                                    ↑ 向右移1位

# 位置5894开始  
样本1_输入 (x[1]): data[5894:5902] = [18, 43, 44, 53, 43, 1, 56, 60]  # "Before w"
样本1_输出 (y[1]): data[5895:5903] = [43, 44, 53, 43, 1, 56, 60, 43]  # "efore we"

# 位置23091开始
样本2_输入 (x[2]): data[23091:23099] = [...]
样本2_输出 (y[2]): data[23092:23100] = [...]

# 位置88234开始
样本3_输入 (x[3]): data[88234:88242] = [...]
样本3_输出 (y[3]): data[88235:88243] = [...]
```

### 第4步：形状和维度

```python
# 最终返回的张量形状
x.shape = torch.Size([4, 8])  # [batch_size, block_size]
y.shape = torch.Size([4, 8])

# 具体数值（用实际字符标注）
x = tensor([
    [20, 43,  1, 60, 43,  1, 52, 47],  # "he we ni"
    [18, 43, 44, 53, 43,  1, 56, 60],  # "Before w"
    [45, 53, 47, 52, 45,  1, 47, 52],  # "going in"  
    [56, 46, 43,  1, 49, 47, 52, 45],  # "the king"
])

y = tensor([
    [43,  1, 60, 43,  1, 52, 47, 56],  # "e we nig"
    [43, 44, 53, 43,  1, 56, 60, 43],  # "efore we"
    [53, 47, 52, 45,  1, 47, 52, 56],  # "going in" (shifted)
    [46, 43,  1, 49, 47, 52, 45, 27],  # "he king:"
])
```

---

## 🎓 为什么这样设计？

### 1. 为什么 y = x 向右移一位？

**这就是"预测下一个词"任务的本质！**

```
输入:  "To be or not to be that is"
           ↓
模型预测每个位置的下一个字符：

位置0: "T" → 预测 "o"
位置1: "o" → 预测 " "
位置2: " " → 预测 "b"
位置3: "b" → 预测 "e"
...

所以 x = "To be or not to be that is"
    y = "o be or not to be that is "
           ↑ 整体向右移动一位
```

**可视化：**
```
x: [T][o][ ][b][e][ ][o][r]
    ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓
y: [o][ ][b][e][ ][o][r][ ]
    ↑  
    预测目标
```

### 2. 为什么随机采样，不按顺序？

**方法A：顺序采样**
```python
# 每次batch都是连续的
batch 1: 位置[0,    1,    2,    3]
batch 2: 位置[4,    5,    6,    7]
batch 3: 位置[8,    9,   10,   11]

问题：模型可能学到batch之间的顺序关系
     → 过拟合
```

**方法B：随机采样** ✅
```python
# 每次batch都是随机的
batch 1: 位置[2341, 8123, 9987, 123]
batch 2: 位置[5432, 1245, 7890, 3456]
batch 3: 位置[9012, 4567, 2345, 6789]

优点：每次看到的都是新鲜组合
     → 泛化能力强
```

### 3. 为什么用 memmap，不直接加载？

**方法A：全部加载到内存**
```python
data = np.load('train.bin')  # 一次性加载1GB数据
# 内存占用：1GB

问题：
- 如果数据集有100GB怎么办？
- 内存不够！
```

**方法B：内存映射** ✅
```python
data = np.memmap('train.bin', dtype=np.uint16, mode='r')
# 内存占用：几乎0，操作系统按需加载

优点：
- 可以处理任意大小的文件
- 访问速度几乎和内存一样快（OS缓存）
- 多进程可以共享同一个文件
```

---

## 🔬 实战：打印一个batch的实际内容

让我们写一个调试脚本：

```python
# debug_get_batch.py
import os
import pickle
import numpy as np
import torch

# 设置
data_dir = 'data/shakespeare_char'
batch_size = 2
block_size = 20

# 加载元数据
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
itos = meta['itos']  # 整数到字符的映射

# 加载数据
data = np.memmap(os.path.join(data_dir, 'train.bin'), 
                 dtype=np.uint16, mode='r')

# 获取一个batch
ix = torch.randint(len(data) - block_size, (batch_size,))
print(f"随机起始位置: {ix.tolist()}\n")

x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) 
                 for i in ix])
y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) 
                 for i in ix])

# 打印每个样本
for i in range(batch_size):
    print(f"{'='*60}")
    print(f"样本 {i}")
    print(f"{'='*60}")
    
    # 打印数字
    print(f"x (数字): {x[i].tolist()}")
    print(f"y (数字): {y[i].tolist()}")
    print()
    
    # 解码成文本
    x_text = ''.join([itos[int(idx)] for idx in x[i]])
    y_text = ''.join([itos[int(idx)] for idx in y[i]])
    
    print(f"x (文本): '{x_text}'")
    print(f"y (文本): '{y_text}'")
    print()
    
    # 打印对应关系
    print("预测任务：")
    for j in range(len(x[i])):
        input_char = itos[int(x[i][j])]
        target_char = itos[int(y[i][j])]
        print(f"  看到 '{input_char}' → 预测 '{target_char}'")
    print()
```

**运行结果示例：**
```
随机起始位置: [45123, 89234]

============================================================
样本 0
============================================================
x (数字): [56, 46, 43, 1, 49, 47, 52, 45, 1, 56, 53, 1, 56, 46, 43, 1, 53, 43, 1, 56]
y (数字): [46, 43, 1, 49, 47, 52, 45, 1, 56, 53, 1, 56, 46, 43, 1, 53, 43, 1, 56, 46]

x (文本): 'the king to the re t'
y (文本): 'he king to the re th'

预测任务：
  看到 't' → 预测 'h'
  看到 'h' → 预测 'e'
  看到 'e' → 预测 ' '
  看到 ' ' → 预测 'k'
  看到 'k' → 预测 'i'
  ...

============================================================
样本 1
============================================================
x (数字): [18, 43, 44, 53, 43, 1, 60, 43, 1, 53, 43, 45, 43, 43, 42, 1, 39, 52, 63]
y (数字): [43, 44, 53, 43, 1, 60, 43, 1, 53, 43, 45, 43, 43, 42, 1, 39, 52, 63, 1]

x (文本): 'Before we proceed Any'
y (文本): 'efore we proceed Any '

预测任务：
  看到 'B' → 预测 'e'
  看到 'e' → 预测 'f'
  看到 'f' → 预测 'o'
  看到 'o' → 预测 'r'
  ...
```

---

## 🧮 性能分析

### 数据加载速度

```python
import time

# 测试1：memmap vs 全加载
# memmap方式
t0 = time.time()
data = np.memmap('train.bin', dtype=np.uint16, mode='r')
for _ in range(1000):
    batch = data[np.random.randint(0, len(data)-1000, 100)]
t1 = time.time()
print(f"memmap: {t1-t0:.3f}秒")

# 全加载方式
t0 = time.time()
data = np.fromfile('train.bin', dtype=np.uint16)
for _ in range(1000):
    batch = data[np.random.randint(0, len(data)-1000, 100)]
t1 = time.time()
print(f"全加载: {t1-t0:.3f}秒")

# 结果（示例）：
# memmap: 0.234秒
# 全加载: 0.221秒 (略快，但占用1GB内存！)
```

### 数据IO vs GPU计算

```python
# 训练中的时间分配（典型情况）
总时间: 100ms/iter

  - 数据加载: 5ms (5%)    ← get_batch()
  - GPU前向传播: 30ms (30%)
  - GPU反向传播: 40ms (40%)
  - 参数更新: 10ms (10%)
  - 日志记录: 15ms (15%)

结论：数据加载不是瓶颈！
```

---

## 💡 进阶技巧

### 1. 异步数据加载

在train.py中有这样一行（第303行）：
```python
# immediately async prefetch next batch while model is doing the forward pass
X, Y = get_batch('train')
```

**为什么在前向传播循环内部调用？**

```python
# 时间线分析：

传统方式（慢）：
  1. CPU: 加载数据 (5ms)
  2. GPU: 计算 (70ms)
  3. CPU: 加载数据 (5ms)  ← 等待，GPU空闲！
  4. GPU: 计算 (70ms)
  
  总时间: (5+70) × 2 = 150ms

NanoGPT方式（快）：
  1. CPU: 加载batch₁ (5ms)
  2. GPU: 计算batch₁ (70ms)  同时→  CPU: 加载batch₂ (5ms)
  3. GPU: 计算batch₂ (70ms)  同时→  CPU: 加载batch₃ (5ms)
  
  总时间: 5 + 70 × 2 = 145ms
  
节省: 5ms per batch!
```

### 2. pin_memory 的作用

```python
x = x.pin_memory().to(device, non_blocking=True)
```

**是什么？**
```
普通内存：CPU可以随时移动和交换
Pinned内存：锁定在RAM中，不会被交换到硬盘

为什么有用？
- GPU可以直接访问pinned内存（DMA传输）
- 非常快！不需要CPU参与
```

**性能对比：**
```python
# 不用pin_memory
x.to(device)  # 15ms

# 用pin_memory
x.pin_memory().to(device, non_blocking=True)  # 3ms

加速: 5倍！
```

---

## 🎯 总结

### get_batch() 的核心职责：

1. ✅ 高效读取大文件（memmap）
2. ✅ 随机采样（避免过拟合）
3. ✅ 构造输入输出对（预测下一个token）
4. ✅ 异步传输到GPU（pin_memory）

### 关键设计哲学：

> **"简单但不简陋"** 
> - 只有20行代码
> - 但包含了4个重要优化
> - 性能接近专业数据加载器

### 下一步：

现在你理解了数据如何"喂"给模型，接下来我们会看到：
- 模型如何"消化"这些数据（forward pass）
- 如何计算loss
- 如何更新参数

准备好继续了吗？
