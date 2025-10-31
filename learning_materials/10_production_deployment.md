# 第10章：生产级部署实战指南 - 从零到上线

> **学习目标**：掌握从训练到部署的完整工程流程  
> **难度等级**：🌳🌳🌳 进阶（工程实战）  
> **预计时间**：6-8小时  
> **前置知识**：前9章基础，特别是05模型架构、08分布式训练、09模型优化

---

## 🎯 你将学到什么

学完本章，你将能够：
- ✅ 理解从模型到服务的完整流程
- ✅ 能够构建高性能的API服务
- ✅ 掌握Docker容器化部署
- ✅ 理解Kubernetes编排的基本原理
- ✅ 会配置监控和日志系统
- ✅ 能够进行性能优化和成本控制
- ✅ 知道如何处理生产环境的常见问题

---

## 💭 开始之前：为什么要学生产部署？

### 🤔 训练好模型≠能用的产品

想象你学会了做一道美味的菜：

```
只会做菜（训练模型）:
  ✅ 在家里做给自己吃
  ❌ 无法服务更多人
  ❌ 没有持续供应能力
  ❌ 无法应对高峰期
  
开餐厅（生产部署）:
  ✅ 标准化流程
  ✅ 能同时服务100+顾客
  ✅ 质量稳定可控
  ✅ 成本优化盈利
```

### 📊 从实验到生产的鸿沟

```python
实验阶段（训练模型）:
  环境: Jupyter Notebook
  用户: 只有你自己
  数据: 几个测试样本
  错误: 重启就好
  成本: 不计较
  
生产阶段（部署服务）:
  环境: 24/7运行的服务器
  用户: 成千上万的真实用户
  数据: 每秒上百个请求
  错误: 每次宕机都是损失
  成本: 精打细算
  
这个鸿沟就是本章要填补的！
```

### 🎯 本章的实战项目

我们将构建一个**代码补全助手**（类似GitHub Copilot的简化版）：

```
完整流程：
  ┌─────────────┐
  │ 数据准备    │ ← 收集Python代码
  └──────┬──────┘
         ↓
  ┌─────────────┐
  │ 模型训练    │ ← 在代码数据上训练GPT
  └──────┬──────┘
         ↓
  ┌─────────────┐
  │ 模型优化    │ ← 量化、加速
  └──────┬──────┘
         ↓
  ┌─────────────┐
  │ API服务     │ ← 构建REST API
  └──────┬──────┘
         ↓
  ┌─────────────┐
  │ 容器化      │ ← Docker打包
  └──────┬──────┘
         ↓
  ┌─────────────┐
  │ 编排部署    │ ← Kubernetes管理
  └──────┬──────┘
         ↓
  ┌─────────────┐
  │ 监控运维    │ ← 确保稳定运行
  └─────────────┘

最终目标：
  ✅ 支持100+并发用户
  ✅ 响应延迟 < 200ms
  ✅ 99.9% 可用性
  ✅ 成本 < $0.01/1K tokens
```

**学完之后**：
- ✅ 你能独立部署一个AI服务
- ✅ 理解生产环境的工程挑战
- ✅ 会优化性能和控制成本
- ✅ 能处理真实世界的问题

---

## 📚 第一部分：数据准备（万丈高楼平地起）

### 💡 1.1 为什么数据准备如此重要？

**生活比喻：**

```
做菜需要食材:
  好食材（高质量数据） → 美味的菜（好模型）
  坏食材（低质量数据） → 难吃的菜（差模型）
  
代码补全也一样:
  高质量代码 → 智能的补全助手
  随机代码 → 胡乱建议
```

### 📊 1.2 数据收集策略

```python
数据来源（优先级排序）:

1. 自己的项目代码 ⭐⭐⭐⭐⭐
   ✅ 质量可控
   ✅ 符合自己的编码风格
   ✅ 没有法律问题
   
2. 开源项目 ⭐⭐⭐⭐
   ✅ 高质量
   ✅ 多样性好
   ⚠️ 注意许可证
   
3. 公开数据集 ⭐⭐⭐
   ✅ 现成可用
   ❌ 可能不符合需求

数据量指南:
  最小可行: 10MB代码 (~200个文件)
  推荐: 100MB代码 (~2000个文件)
  理想: 1GB+ 代码
```

### 🔧 1.3 数据收集脚本（逐步理解）

```python
# collect_code.py - 第一步：遍历文件
import os
import tiktoken
import numpy as np

def collect_python_files(root_dir):
    """
    收集所有Python文件
    
    工作原理：
      1. 递归遍历目录
      2. 跳过无用的文件夹（__pycache__等）
      3. 读取每个.py文件
      4. 过滤太短的文件
    """
    code_files = []
    
    # os.walk会递归遍历所有子目录
    for root, dirs, files in os.walk(root_dir):
        # 过滤掉不需要的目录
        # dirs[:] 是就地修改，会影响os.walk的行为
        dirs[:] = [d for d in dirs if d not in [
            '.venv',        # 虚拟环境
            '__pycache__',  # Python缓存
            'node_modules', # Node.js依赖
            '.git'          # Git目录
        ]]
        
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        code = f.read()
                        # 过滤太短的文件（可能是空文件或只有import）
                        if len(code) > 100:
                            code_files.append(code)
                except Exception as e:
                    # 某些文件可能无法读取，跳过即可
                    print(f"跳过文件 {path}: {e}")
                    continue
    
    return code_files
```

```python
# 第二步：准备训练数据
def prepare_dataset(code_files, output_dir='data/python_code'):
    """
    将代码转换为训练数据
    
    步骤：
      1. 合并所有代码文件
      2. 分割训练/验证集（9:1）
      3. Tokenize（转换为数字）
      4. 保存为二进制文件（节省空间和加载时间）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 合并所有代码，用分隔符连接
    separator = '\n\n# ================\n\n'
    data = separator.join(code_files)
    print(f"📊 总字符数: {len(data):,}")
    
    # 2. 分割训练/验证集
    n = len(data)
    split_idx = int(n * 0.9)  # 90%训练，10%验证
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"📊 训练字符数: {len(train_data):,}")
    print(f"📊 验证字符数: {len(val_data):,}")
    
    # 3. Tokenize（使用GPT-2的tokenizer）
    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    
    print(f"🔢 训练tokens: {len(train_ids):,}")
    print(f"🔢 验证tokens: {len(val_ids):,}")
    
    # 4. 保存为二进制文件
    # 使用uint16因为GPT-2词汇表大小是50257
    train_ids_np = np.array(train_ids, dtype=np.uint16)
    val_ids_np = np.array(val_ids, dtype=np.uint16)
    
    train_ids_np.tofile(f'{output_dir}/train.bin')
    val_ids_np.tofile(f'{output_dir}/val.bin')
    
    print(f"✅ 数据已保存到 {output_dir}/")
    
    # 5. 保存元信息
    meta = {
        'num_files': len(code_files),
        'train_tokens': len(train_ids),
        'val_tokens': len(val_ids),
        'vocab_size': 50257
    }
    import json
    with open(f'{output_dir}/meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

# 运行脚本
if __name__ == '__main__':
    # 替换为你的项目路径
    root_directory = '/path/to/your/python/projects'
    
    print("🔍 开始收集Python文件...")
    code_files = collect_python_files(root_directory)
    print(f"✅ 收集了 {len(code_files)} 个Python文件")
    
    print("\n📦 开始准备数据集...")
    prepare_dataset(code_files)
    print("\n🎉 数据准备完成！")
```

### 🚀 1.4 运行数据收集

```bash
# 安装依赖
pip install tiktoken numpy

# 运行脚本
python collect_code.py

# 预期输出：
"""
🔍 开始收集Python文件...
✅ 收集了 1,234 个Python文件

📦 开始准备数据集...
📊 总字符数: 5,678,901
📊 训练字符数: 5,111,011
📊 验证字符数: 567,890
🔢 训练tokens: 1,234,567
🔢 验证tokens: 137,174
✅ 数据已保存到 data/python_code/

🎉 数据准备完成！
"""
```

### ✅ 1.5 数据质量检查清单

```python
检查项目：
  □ 文件数量 > 100个
  □ 训练tokens > 100K
  □ 没有乱码或编码错误
  □ train.bin和val.bin都生成了
  □ 文件大小合理（几MB到几百MB）

常见问题:
  Q: 收集了很多但tokens很少？
  A: 可能包含大量注释或空行，这是正常的
  
  Q: 某些文件读取失败？
  A: 跳过即可，不影响整体质量
  
  Q: 需要多少数据？
  A: 100K tokens可以开始，1M tokens效果更好
```

---

## 📚 第二部分：模型训练（让模型学会写代码）

### 💡 2.1 训练策略：从预训练模型开始

**为什么不从头训练？**

```
从头训练（Training from scratch）:
  时间: 需要数周
  数据: 需要TB级数据
  成本: 数万美元
  结果: 可能还不如预训练模型
  
从预训练模型微调（Fine-tuning）:
  时间: 几个小时 ✅
  数据: MB级数据就够 ✅
  成本: 几美元 ✅
  结果: 往往更好 ✅

就像学外语：
  从头学 = 婴儿学说话（需要多年）
  微调 = 成年人学新词汇（几个月就能用）
```

### 🔧 2.2 创建训练配置（逐参数理解）

```python
# config/train_code_assistant.py

import time

# ===== 输出和日志配置 =====
out_dir = 'out-code-assistant'  # 模型保存目录
eval_interval = 500              # 每500步评估一次
eval_iters = 100                 # 评估时用100个batch
log_interval = 10                # 每10步打印日志

# ===== 数据集配置 =====
dataset = 'python_code'          # 使用我们准备的数据集
gradient_accumulation_steps = 4  # 梯度累积（小显存必备）
batch_size = 8                   # 每个GPU的batch大小
block_size = 512                 # 上下文长度（代码需要更长）

"""
为什么block_size=512？
  - 代码通常比自然语言需要更长的上下文
  - 需要看到完整的函数定义才能补全
  - 512是性能和显存的折中
"""

# ===== 模型架构配置 =====
n_layer = 12   # Transformer层数
n_head = 12    # 注意力头数
n_embd = 768   # 嵌入维度
dropout = 0.1  # Dropout比例

"""
这是GPT-2 Small的架构（124M参数）
  - 足够强大，能理解复杂代码
  - 不太大，单GPU可以训练
  - 推理速度快
"""

# ===== 微调设置（关键！）=====
init_from = 'gpt2'  # 从GPT-2预训练模型开始

"""
为什么从GPT-2开始？
  1. 已经理解基本语法
  2. 已经学会了一些编程概念
  3. 只需要适应我们的代码风格
"""

learning_rate = 5e-5  # 学习率（比从头训练小10倍！）
max_iters = 5000      # 训练步数
lr_decay_iters = 5000 # 学习率衰减步数
min_lr = 5e-6         # 最小学习率

"""
微调的学习率原则：
  - 要小！避免破坏预训练知识
  - 通常是从头训练的1/10
  - 5e-5是个经验好值
"""

# ===== 优化器配置 =====
weight_decay = 0.1  # L2正则化
grad_clip = 1.0     # 梯度裁剪
decay_lr = True     # 启用学习率衰减
warmup_iters = 100  # Warmup步数

# ===== 系统配置 =====
device = 'cuda'     # 使用GPU
dtype = 'float16'   # 混合精度训练
compile = True      # PyTorch 2.0编译加速
```

### 📊 2.3 理解训练过程

```python
训练流程（每一步都在做什么）:

步骤1: 加载预训练模型
  └─ 从Hugging Face下载GPT-2权重
  └─ 初始化loss ≈ 3.0

步骤2: 在代码数据上微调
  ├─ Iter 0-100:   快速适应（warmup）
  │  └─ loss: 3.0 → 2.5
  │
  ├─ Iter 100-2000: 主要学习阶段
  │  └─ loss: 2.5 → 1.5
  │
  └─ Iter 2000-5000: 精细调整
     └─ loss: 1.5 → 1.2

步骤3: 评估和保存
  └─ 每500步评估val loss
  └─ 保存最佳模型

预期效果:
  ✅ Train loss: 1.2-1.5
  ✅ Val loss: 1.5-1.8
  ✅ 能生成合理的代码补全
```

### 🚀 2.4 开始训练

```bash
# 单GPU训练
python train.py config/train_code_assistant.py

# 预期输出（每10步打印一次）:
"""
iter 0: loss 3.2145, time 1234.56ms
iter 10: loss 2.9876, time 234.56ms
iter 20: loss 2.7654, time 234.12ms
...
iter 500: train loss 2.1234, val loss 2.3456
📊 已训练 10% | 预计剩余时间: 1.5小时
...
iter 1000: train loss 1.8765, val loss 2.1234
📊 已训练 20% | 预计剩余时间: 1.2小时
...
iter 5000: train loss 1.2345, val loss 1.5678
✅ 训练完成！最佳val loss: 1.5234
💾 模型已保存到: out-code-assistant/ckpt.pt
"""
```

### ⏱️ 2.5 训练时间估算

```python
硬件配置 vs 训练时间:

单个RTX 3060 (12GB):
  ├─ 每步: ~500ms
  ├─ 5000步: ~40分钟
  └─ 💡 适合学习和实验

单个RTX 3090 (24GB):
  ├─ 每步: ~300ms
  ├─ 5000步: ~25分钟
  └─ 💡 性价比之选

单个A100 (40GB):
  ├─ 每步: ~150ms
  ├─ 5000步: ~12分钟
  └─ 💡 专业选择

显存不够怎么办？
  1. 减小batch_size到4或2
  2. 减小block_size到256
  3. 使用梯度累积
```

---

## 📚 第三部分：分布式加速（多卡训练）

### 💡 3.1 为什么需要分布式训练？

**时间就是金钱：**

```
单GPU训练:
  RTX 3060: 40分钟
  成本: 电费 ~$0.5
  
4×GPU训练:
  RTX 3060×4: 10-12分钟 (3-4x加速)
  成本: 电费 ~$0.6
  
结论: 多花20%成本，节省70%时间！
```

### 📊 3.2 分布式训练原理

```python
数据并行（Data Parallel）:

单GPU:
  GPU 0: 处理 batch 1-8
  
4×GPU (每个GPU处理不同数据):
  GPU 0: 处理 batch 1-8
  GPU 1: 处理 batch 9-16
  GPU 2: 处理 batch 17-24
  GPU 3: 处理 batch 25-32
  
  ↓ 计算梯度
  ↓ 同步梯度（All-Reduce）
  ↓ 更新模型
  
结果: 4倍数据吞吐量！
```

### 🔧 3.3 DDP配置（数据并行）

```python
# config/train_code_assistant_ddp.py

# 继承单GPU配置
exec(open('config/train_code_assistant.py').read())

# ===== DDP优化 =====
batch_size = 16  # 每个GPU的batch更大（有4个GPU帮忙了）
gradient_accumulation_steps = 2  # 减少累积步数

"""
等效batch计算:
  单GPU配置:
    batch_size = 8
    gradient_accumulation_steps = 4
    等效batch = 8 × 4 = 32
    
  4×GPU配置:
    batch_size = 16
    gradient_accumulation_steps = 2  
    num_gpus = 4
    等效batch = 16 × 2 × 4 = 128
    
  结果: 4倍吞吐量！
"""
```

### 🚀 3.4 启动分布式训练

```bash
# 使用torchrun启动（PyTorch推荐）
torchrun --standalone --nproc_per_node=4 \
  train.py config/train_code_assistant_ddp.py

# 参数说明:
#   --standalone: 单机多卡模式
#   --nproc_per_node=4: 使用4个GPU

# 预期输出:
"""
[GPU 0] Initializing DDP...
[GPU 1] Initializing DDP...
[GPU 2] Initializing DDP...
[GPU 3] Initializing DDP...
✅ All processes initialized

[GPU 0] iter 0: loss 3.2145, time 456.78ms
[GPU 0] iter 10: loss 2.9876, time 123.45ms
...
[GPU 0] iter 500: train loss 2.1234, val loss 2.3456
📊 加速比: 3.8x (理想是4x)
⏱️ 预计剩余时间: 8分钟
...
"""
```

### 📈 3.5 加速效果对比

```python
实际加速比（经验值）:

┌──────────┬──────────┬──────────┬──────────┐
│ GPU数量  │ 训练时间 │ 加速比   │ 效率     │
├──────────┼──────────┼──────────┼──────────┤
│ 1        │ 40分钟   │ 1.0x     │ 100%     │
│ 2        │ 22分钟   │ 1.8x     │ 90%      │
│ 4        │ 12分钟   │ 3.3x     │ 82%      │
│ 8        │ 7分钟    │ 5.7x     │ 71%      │
└──────────┴──────────┴──────────┴──────────┘

为什么不是完美线性加速？
  1. 梯度同步开销（All-Reduce通信）
  2. 负载不均衡
  3. 内存带宽瓶颈
  
  但80%+的效率已经很好了！
```

### ⚠️ 3.6 常见问题

```python
Q1: "NCCL error: unhandled system error"
A: GPU之间通信问题
   └─ 检查: nvidia-smi topo -m
   └─ 解决: 使用相同型号的GPU

Q2: 显存溢出（OOM）
A: 多GPU不会减少单GPU显存使用
   └─ 解决: 每个GPU都要能装下模型

Q3: 速度没有提升
A: 可能是数据加载瓶颈
   └─ 增加num_workers
   └─ 使用更快的存储（SSD）

Q4: Loss不同步
A: 正常！每个GPU看到不同数据
   └─ 只有GPU 0打印loss
   └─ 最终模型是同步的
```

---

## 📚 第四部分：模型优化（让它又快又小）

### 💡 4.1 为什么要优化模型？

**生活比喻：给汽车减重**

```
原始模型（FP32）:
  就像一辆装满行李的SUV
  ✅ 功能完整
  ❌ 耗油（显存）
  ❌ 跑得慢
  
优化后模型（INT8）:
  就像精简后的轿车
  ✅ 功能基本保持
  ✅ 省油75%
  ✅ 跑得快2-3倍
```

### 📊 4.2 优化方法对比

```python
优化技术对比:

┌──────────┬──────────┬──────────┬──────────┬──────────┐
│ 方法     │ 压缩比   │ 速度提升 │ 精度损失 │ 难度     │
├──────────┼──────────┼──────────┼──────────┼──────────┤
│ FP16     │ 2x       │ 1.5x     │ 几乎无   │ ⭐ 简单  │
│ INT8     │ 4x       │ 2-3x     │ 1-3%     │ ⭐⭐ 中等│
│ INT4     │ 8x       │ 3-4x     │ 5-10%    │ ⭐⭐⭐ 难│
│ 剪枝     │ 2-3x     │ 1.5-2x   │ 可变     │ ⭐⭐⭐⭐│
└──────────┴──────────┴──────────┴──────────┴──────────┘

推荐顺序：
  1️⃣ 先用INT8量化（最佳性价比）
  2️⃣ 如果还需要更快，考虑INT4
  3️⃣ 剪枝留给研究
```

### 🔧 4.3 INT8量化（逐步理解）

**什么是量化？**

```python
原理：用更少的bit表示数字

FP32（浮点32位）:
  范围: -3.4×10³⁸ ~ 3.4×10³⁸
  精度: 非常高
  存储: 4 bytes
  
INT8（整数8位）:
  范围: -128 ~ 127
  精度: 较低
  存储: 1 byte
  
量化过程:
  1. 统计权重的范围: min=-2.5, max=2.5
  2. 映射到INT8: -2.5→-128, 2.5→127
  3. 推理时反量化回浮点数
  
关键: 大部分权重在[-2, 2]范围内
       所以精度损失很小！
```

### 📝 4.4 量化脚本（详细注释版）

```python
# quantize_model.py
import torch
from model import GPT, GPTConfig
import os

def quantize_model(checkpoint_path, output_path):
    """
    将FP32模型量化到INT8
    
    好处:
      - 模型大小减少75%
      - 推理速度提升2-3倍
      - 精度损失 < 2%
    """
    print("=" * 60)
    print("🔧 开始量化模型")
    print("=" * 60)
    
    # ===== 步骤1: 加载原始模型 =====
    print("\n📥 加载原始模型...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 创建模型
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    model.load_state_dict(checkpoint['model'])
    model.eval()  # 设置为评估模式
    
    # 统计参数
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 模型参数量: {num_params/1e6:.2f}M")
    
    # ===== 步骤2: 动态量化 =====
    print("\n⚙️ 执行INT8量化...")
    print("   量化目标: 所有Linear层")
    
    """
    动态量化（Dynamic Quantization）:
      - 只量化权重（Weights），不量化激活值
      - 推理时动态计算激活值的量化参数
      - 最简单，效果好，推荐使用
    """
    model_quantized = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},  # 只量化Linear层
        dtype=torch.qint8   # 量化到INT8
    )
    
    print("✅ 量化完成")
    
    # ===== 步骤3: 保存量化模型 =====
    print("\n💾 保存量化模型...")
    torch.save({
        'model': model_quantized.state_dict(),
        'model_args': checkpoint['model_args'],
        'quantized': True,
        'quantization_type': 'dynamic_int8'
    }, output_path)
    
    # ===== 步骤4: 对比分析 =====
    print("\n📊 量化效果分析:")
    print("-" * 60)
    
    orig_size = os.path.getsize(checkpoint_path) / 1e6
    quant_size = os.path.getsize(output_path) / 1e6
    compression_ratio = orig_size / quant_size
    
    print(f"原始模型:")
    print(f"  文件大小: {orig_size:.2f} MB")
    print(f"  精度: FP32 (32位浮点)")
    print(f"  推理速度: 基准")
    
    print(f"\n量化模型:")
    print(f"  文件大小: {quant_size:.2f} MB ⬇️")
    print(f"  精度: INT8 (8位整数)")
    print(f"  推理速度: 预计2-3x ⬆️")
    
    print(f"\n压缩效果:")
    print(f"  压缩比: {compression_ratio:.2f}x")
    print(f"  节省空间: {orig_size - quant_size:.2f} MB")
    print(f"  预计精度损失: < 2%")
    
    print("\n" + "=" * 60)
    print("✅ 量化完成！")
    print("=" * 60)
    
    return model_quantized

def test_quantized_model(model_path, prompt="def fibonacci(n):\n    "):
    """
    测试量化模型的生成效果
    """
    print(f"\n🧪 测试量化模型...")
    print(f"输入: {prompt}")
    
    # 加载并生成
    # （这里简化，实际需要加载tokenizer等）
    print("输出: [代码补全结果]")
    print("✅ 生成正常")

if __name__ == '__main__':
    # 配置路径
    original_model = 'out-code-assistant/ckpt.pt'
    quantized_model = 'out-code-assistant/ckpt_int8.pt'
    
    # 执行量化
    model = quantize_model(original_model, quantized_model)
    
    # 可选：测试量化模型
    # test_quantized_model(quantized_model)
```

### 🚀 4.5 运行量化

```bash
# 运行量化脚本
python quantize_model.py

# 预期输出：
"""
============================================================
🔧 开始量化模型
============================================================

📥 加载原始模型...
✅ 模型参数量: 124.44M

⚙️ 执行INT8量化...
   量化目标: 所有Linear层
✅ 量化完成

💾 保存量化模型...

📊 量化效果分析:
------------------------------------------------------------
原始模型:
  文件大小: 497.35 MB
  精度: FP32 (32位浮点)
  推理速度: 基准

量化模型:
  文件大小: 124.67 MB ⬇️
  精度: INT8 (8位整数)
  推理速度: 预计2-3x ⬆️

压缩效果:
  压缩比: 3.99x
  节省空间: 372.68 MB
  预计精度损失: < 2%

============================================================
✅ 量化完成！
============================================================
"""
```

### 📊 4.6 量化前后对比测试

```python
# benchmark_quantization.py
import torch
import time

def benchmark_model(model_path, num_runs=100):
    """
    基准测试：比较量化前后的性能
    """
    # 加载模型
    model = load_model(model_path)
    
    # 预热
    for _ in range(10):
        output = model.generate("def test():\n    ", max_new_tokens=50)
    
    # 测试推理时间
    start_time = time.time()
    for _ in range(num_runs):
        output = model.generate("def test():\n    ", max_new_tokens=50)
    elapsed = time.time() - start_time
    
    avg_time = elapsed / num_runs
    tokens_per_sec = 50 / avg_time
    
    return {
        'avg_time': avg_time,
        'tokens_per_sec': tokens_per_sec
    }

# 对比测试
print("测试FP32模型...")
fp32_results = benchmark_model('out-code-assistant/ckpt.pt')

print("测试INT8模型...")
int8_results = benchmark_model('out-code-assistant/ckpt_int8.pt')

# 打印结果
print("\n性能对比:")
print(f"FP32: {fp32_results['avg_time']*1000:.1f}ms/生成 | {fp32_results['tokens_per_sec']:.1f} tokens/s")
print(f"INT8: {int8_results['avg_time']*1000:.1f}ms/生成 | {int8_results['tokens_per_sec']:.1f} tokens/s")
print(f"加速比: {fp32_results['avg_time']/int8_results['avg_time']:.2f}x")

"""
典型输出:
  FP32: 250.3ms/生成 | 200 tokens/s
  INT8: 105.7ms/生成 | 473 tokens/s
  加速比: 2.37x ✅
"""
```

### ✅ 4.7 量化检查清单

```python
验证量化是否成功:

□ 模型文件大小减少约75%
□ 能够正常加载和推理
□ 生成的代码质量相近
□ 推理速度提升2x以上

常见问题:

Q: 量化后模型无法加载？
A: 确保加载时使用相同的模型架构
   torch.load(..., map_location='cpu')

Q: 生成质量下降明显？
A: 尝试更温和的量化：
   - 先用FP16
   - 或只量化部分层

Q: 速度没有提升？
A: 可能是:
   - CPU上效果不明显（用GPU）
   - batch_size太小
   - 其他瓶颈（I/O等）

进阶优化:
  □ 尝试静态量化（需要校准数据）
  □ 使用GPTQ/AWQ（更好的INT4量化）
  □ 结合剪枝技术
```

---

## 📚 第五部分：API服务（让别人能用你的模型）

### 💡 5.1 为什么需要API服务？

**生活比喻：开外卖店**

```
只有模型（本地运行）:
  就像只能在家做饭
  ✅ 自己吃方便
  ❌ 别人用不了
  ❌ 不能远程访问
  
API服务（Web服务）:
  就像开了外卖店
  ✅ 任何人都能点餐（调用API）
  ✅ 不用关心后厨（模型）怎么工作
  ✅ 可以同时服务多个客户
```

### 📊 5.2 API设计原则

```python
好的API设计:

1. RESTful风格
   POST /complete  # 代码补全
   GET  /health    # 健康检查
   
2. 清晰的输入输出
   输入: { "code": "...", "max_tokens": 50 }
   输出: { "completion": "...", "latency_ms": 156 }
   
3. 错误处理
   ✅ 明确的错误信息
   ✅ 合适的HTTP状态码
   
4. 性能监控
   ✅ 记录延迟
   ✅ 统计请求量
```

### 🔧 5.3 FastAPI服务（逐步构建）

**为什么选择FastAPI？**

```python
FastAPI的优势:
  ✅ 自动生成API文档（访问 /docs）
  ✅ 类型验证（Pydantic）
  ✅ 异步支持（高并发）
  ✅ 性能优秀
  ✅ 易学易用

对比:
  Flask: 简单但功能少
  Django: 功能多但重
  FastAPI: 完美平衡 ⭐⭐⭐⭐⭐
```

### 📝 5.4 完整API实现（详细注释）

```python
# serve_api.py
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import torch
from model import GPT, GPTConfig
import tiktoken
from contextlib import asynccontextmanager
from typing import Optional
import time
import logging

# ===== 配置日志 =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===== 全局变量 =====
model = None
tokenizer = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ===== 生命周期管理 =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用启动和关闭时的处理
    
    启动时:
      - 加载模型到内存
      - 预热模型
    
    关闭时:
      - 释放模型内存
    """
    global model, tokenizer
    
    # 启动
    logger.info("=" * 60)
    logger.info("🚀 启动API服务")
    logger.info("=" * 60)
    
    logger.info("📥 加载模型...")
    checkpoint = torch.load(
        'out-code-assistant/ckpt_int8.pt',
        map_location=device
    )
    
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    model.load_state_dict(checkpoint['model'])
    model.eval()  # 设置为评估模式
    model.to(device)
    
    logger.info("📝 加载tokenizer...")
    tokenizer = tiktoken.get_encoding("gpt2")
    
    logger.info(f"✅ 模型已加载到 {device}")
    logger.info(f"📊 模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 预热模型（第一次推理通常较慢）
    logger.info("🔥 预热模型...")
    with torch.no_grad():
        dummy_input = torch.zeros((1, 10), dtype=torch.long, device=device)
        _ = model.generate(dummy_input, max_new_tokens=10)
    logger.info("✅ 预热完成")
    
    logger.info("=" * 60)
    logger.info("✅ API服务就绪！")
    logger.info("=" * 60)
    
    yield  # 应用运行期间
    
    # 关闭
    logger.info("🔄 关闭API服务...")
    del model
    del tokenizer
    logger.info("✅ 资源已释放")

# ===== 创建FastAPI应用 =====
app = FastAPI(
    title="代码补全API",
    description="基于GPT的Python代码补全服务",
    version="1.0.0",
    lifespan=lifespan
)

# ===== 数据模型 =====
class CompletionRequest(BaseModel):
    """代码补全请求"""
    code: str = Field(
        ...,
        description="输入的代码",
        example="def fibonacci(n):\n    if n <= 1:\n        return n\n    "
    )
    max_tokens: int = Field(
        50,
        description="最大生成token数",
        ge=1,
        le=200
    )
    temperature: float = Field(
        0.8,
        description="生成温度（0-2）",
        ge=0.0,
        le=2.0
    )
    top_k: Optional[int] = Field(
        200,
        description="Top-K采样",
        ge=1,
        le=1000
    )

class CompletionResponse(BaseModel):
    """代码补全响应"""
    completion: str = Field(..., description="生成的代码补全")
    tokens: int = Field(..., description="生成的token数")
    latency_ms: float = Field(..., description="延迟（毫秒）")
    
class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    model_loaded: bool
    device: str
    uptime_seconds: float

# ===== 全局变量 =====
start_time = time.time()
request_count = 0

# ===== API端点 =====
@app.post("/complete", response_model=CompletionResponse)
async def complete_code(request: CompletionRequest):
    """
    代码补全API
    
    功能:
      - 接收部分代码
      - 返回智能补全
      - 记录延迟和统计
    """
    global request_count
    request_count += 1
    
    logger.info(f"📝 收到请求 #{request_count}")
    logger.debug(f"   输入长度: {len(request.code)} 字符")
    
    try:
        start = time.time()
        
        # ===== 步骤1: 输入处理 =====
        input_ids = tokenizer.encode(request.code)
        
        # 截断过长的输入
        max_input_length = 512
        if len(input_ids) > max_input_length:
            logger.warning(f"⚠️ 输入过长，截断到 {max_input_length} tokens")
            input_ids = input_ids[-max_input_length:]
        
        # 转换为tensor
        x = torch.tensor([input_ids], dtype=torch.long, device=device)
        
        # ===== 步骤2: 模型推理 =====
        with torch.no_grad():
            y = model.generate(
                x,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k
            )
        
        # ===== 步骤3: 输出处理 =====
        output_ids = y[0].tolist()
        # 只返回新生成的部分
        generated_ids = output_ids[len(input_ids):]
        completion = tokenizer.decode(generated_ids)
        
        # ===== 步骤4: 统计 =====
        latency = (time.time() - start) * 1000
        
        logger.info(f"✅ 请求完成")
        logger.info(f"   生成tokens: {len(generated_ids)}")
        logger.info(f"   延迟: {latency:.1f}ms")
        logger.info(f"   速度: {len(generated_ids)/(latency/1000):.1f} tokens/s")
        
        return CompletionResponse(
            completion=completion,
            tokens=len(generated_ids),
            latency_ms=latency
        )
    
    except Exception as e:
        logger.error(f"❌ 请求失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"生成失败: {str(e)}"
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    健康检查端点
    
    用于:
      - 监控服务状态
      - 负载均衡器探测
      - 自动重启判断
    """
    uptime = time.time() - start_time
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=device,
        uptime_seconds=uptime
    )

@app.get("/stats")
async def get_stats():
    """
    统计信息端点
    """
    uptime = time.time() - start_time
    
    return {
        "total_requests": request_count,
        "uptime_seconds": uptime,
        "requests_per_minute": request_count / (uptime / 60) if uptime > 0 else 0,
        "device": device,
        "model_parameters": f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M"
    }

# ===== 启动服务 =====
if __name__ == '__main__':
    import uvicorn
    
    # 生产环境配置
    uvicorn.run(
        app,
        host="0.0.0.0",  # 监听所有网卡
        port=8000,
        log_level="info"
    )
```

### 🚀 5.5 启动和测试

**启动服务：**

```bash
# 方式1：直接运行
python serve_api.py

# 方式2：使用uvicorn（推荐生产环境）
uvicorn serve_api:app --host 0.0.0.0 --port 8000 --workers 4

# 预期输出：
"""
============================================================
🚀 启动API服务
============================================================
📥 加载模型...
📝 加载tokenizer...
✅ 模型已加载到 cuda
📊 模型参数量: 124.44M
🔥 预热模型...
✅ 预热完成
============================================================
✅ API服务就绪！
============================================================
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
"""
```

### 🧪 5.6 测试API

**1. 健康检查：**

```bash
curl http://localhost:8000/health

# 输出:
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "uptime_seconds": 123.45
}
```

**2. 代码补全：**

```bash
curl -X POST http://localhost:8000/complete \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    ",
    "max_tokens": 50,
    "temperature": 0.8
  }'

# 输出:
{
  "completion": "return fibonacci(n-1) + fibonacci(n-2)\n\n# Test\nprint(fibonacci(10))",
  "tokens": 23,
  "latency_ms": 156.78
}
```

**3. 查看统计：**

```bash
curl http://localhost:8000/stats

# 输出:
{
  "total_requests": 42,
  "uptime_seconds": 3600.5,
  "requests_per_minute": 0.7,
  "device": "cuda",
  "model_parameters": "124.44M"
}
```

**4. 自动文档（FastAPI自动生成）：**

```bash
# 在浏览器打开
open http://localhost:8000/docs

# 你会看到:
#   - 交互式API文档
#   - 可以直接测试每个端点
#   - 自动生成的请求/响应示例
```

### 📊 5.7 性能测试

```python
# benchmark_api.py
import requests
import time
import concurrent.futures
from statistics import mean, stdev

def test_single_request():
    """测试单次请求"""
    response = requests.post(
        'http://localhost:8000/complete',
        json={
            'code': 'def test():\n    ',
            'max_tokens': 50
        }
    )
    return response.json()['latency_ms']

def benchmark_sequential(num_requests=100):
    """顺序请求测试"""
    print(f"📊 顺序测试 {num_requests} 个请求...")
    
    latencies = []
    start_time = time.time()
    
    for i in range(num_requests):
        latency = test_single_request()
        latencies.append(latency)
        if (i + 1) % 10 == 0:
            print(f"  完成 {i+1}/{num_requests}")
    
    total_time = time.time() - start_time
    
    print(f"\n顺序测试结果:")
    print(f"  总时间: {total_time:.2f}s")
    print(f"  平均延迟: {mean(latencies):.1f}ms")
    print(f"  标准差: {stdev(latencies):.1f}ms")
    print(f"  吞吐量: {num_requests/total_time:.2f} req/s")

def benchmark_concurrent(num_requests=100, workers=10):
    """并发请求测试"""
    print(f"\n📊 并发测试 {num_requests} 个请求 ({workers}个worker)...")
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(test_single_request)
            for _ in range(num_requests)
        ]
        latencies = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    total_time = time.time() - start_time
    
    print(f"\n并发测试结果:")
    print(f"  总时间: {total_time:.2f}s")
    print(f"  平均延迟: {mean(latencies):.1f}ms")
    print(f"  标准差: {stdev(latencies):.1f}ms")
    print(f"  吞吐量: {num_requests/total_time:.2f} req/s")

if __name__ == '__main__':
    # 确保服务正在运行
    try:
        requests.get('http://localhost:8000/health')
    except:
        print("❌ API服务未启动，请先运行 python serve_api.py")
        exit(1)
    
    # 运行测试
    benchmark_sequential(50)
    benchmark_concurrent(50, workers=5)

"""
典型输出:
  📊 顺序测试 50 个请求...
  完成 10/50
  完成 20/50
  ...
  
  顺序测试结果:
    总时间: 8.45s
    平均延迟: 165.3ms
    标准差: 12.4ms
    吞吐量: 5.92 req/s
  
  📊 并发测试 50 个请求 (5个worker)...
  
  并发测试结果:
    总时间: 2.15s
    平均延迟: 189.7ms
    标准差: 45.2ms
    吞吐量: 23.26 req/s ✅ 提升4x！
"""
```

### ✅ 5.8 API服务检查清单

```python
功能验证:
  □ 服务能正常启动
  □ 健康检查返回正常
  □ 能够正确补全代码
  □ 延迟在可接受范围（<200ms）
  □ 错误处理正确

性能验证:
  □ 并发支持正常
  □ 内存不泄漏
  □ GPU利用率高
  □ 吞吐量满足需求

监控验证:
  □ 日志输出清晰
  □ 统计信息准确
  □ 能追踪请求

下一步:
  □ 添加认证（API Key）
  □ 添加限流（Rate Limiting）
  □ 添加缓存（Redis）
  □ 监控告警（Prometheus）
```

---

## 📚 第六部分：容器化部署（打包成可移植的应用）

### 💡 6.1 为什么要用Docker？

**生活比喻：外卖打包**

```
不用Docker（直接部署）:
  就像让顾客自己来厨房做饭
  ❌ 环境不一致（Python版本、依赖）
  ❌ 部署麻烦（要手动装各种软件）
  ❌ 难以扩展（每台服务器都要配置）
  
用Docker（容器化）:
  就像外卖打包好送到家
  ✅ 环境一致（容器自带所有依赖）
  ✅ 一键部署（docker run就行）
  ✅ 易于扩展（复制容器即可）
```

### 📊 6.2 Docker核心概念

```python
Docker三要素:

1. Dockerfile（食谱）
   定义如何构建环境
   包含所有安装步骤
   
2. Image（镜像/半成品）
   根据Dockerfile构建出来的
   包含代码、依赖、系统
   可以分享和复用
   
3. Container（容器/成品）
   Image的运行实例
   独立的运行环境
   可以启动、停止、删除

工作流程:
  Dockerfile → docker build → Image → docker run → Container
```

### 🔧 6.3 创建Dockerfile（逐步理解）

```dockerfile
# Dockerfile - 定义容器环境

# ===== 第1步：选择基础镜像 =====
# 使用NVIDIA官方的CUDA镜像（包含GPU支持）
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 为什么用这个？
# - Ubuntu 22.04：稳定的Linux系统
# - CUDA 11.8：支持PyTorch的GPU计算
# - cudnn8：深度学习加速库
# - runtime：只包含运行时，不包含开发工具（更小）

# ===== 第2步：安装Python =====
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 解释:
# apt-get update: 更新软件包列表
# apt-get install: 安装Python 3.10和pip
# rm -rf /var/lib/apt/lists/*: 清理缓存，减小镜像大小

# ===== 第3步：设置工作目录 =====
WORKDIR /app

# 所有后续命令都在/app目录下执行
# 就像cd /app

# ===== 第4步：安装Python依赖 =====
# 先复制requirements.txt（利用Docker缓存）
COPY requirements.txt .

# 安装依赖
RUN pip3 install --no-cache-dir -r requirements.txt

# 为什么分两步？
# Docker会缓存每一层，requirements.txt不常变
# 这样代码改了，依赖不用重新安装

# ===== 第5步：复制应用代码 =====
COPY model.py .
COPY serve_api.py .
COPY out-code-assistant/ckpt_int8.pt ./out-code-assistant/

# 把模型和代码复制到容器里

# ===== 第6步：暴露端口 =====
EXPOSE 8000

# 告诉Docker这个容器会使用8000端口
# 注意：这只是声明，实际映射在docker run时指定

# ===== 第7步：定义启动命令 =====
CMD ["python3", "serve_api.py"]

# 容器启动时执行这个命令
# 相当于在容器里运行: python3 serve_api.py
```

### 📦 6.4 准备依赖文件

```txt
# requirements.txt - Python依赖列表

# PyTorch（深度学习框架）
torch==2.0.1

# FastAPI（Web框架）
fastapi==0.104.1

# Uvicorn（ASGI服务器）
uvicorn==0.24.0

# Pydantic（数据验证）
pydantic==2.5.0

# Tiktoken（OpenAI的tokenizer）
tiktoken==0.5.1

# NumPy（数值计算）
numpy==1.24.3

# 提示：
# - 固定版本号确保环境一致
# - 只包含必需的包
# - 定期更新以获得bug修复
```

### 🚀 6.5 构建和运行Docker容器

**第1步：构建镜像**

```bash
# 在项目根目录执行
docker build -t code-assistant:v1 .

# 参数说明:
#   -t code-assistant:v1  给镜像打标签（名称:版本）
#   .                     Dockerfile所在目录

# 预期输出:
"""
[+] Building 125.3s (12/12) FINISHED
 => [1/7] FROM nvidia/cuda:11.8.0...           15.2s
 => [2/7] RUN apt-get update...                 8.5s
 => [3/7] WORKDIR /app                          0.1s
 => [4/7] COPY requirements.txt .               0.1s
 => [5/7] RUN pip3 install...                  85.3s
 => [6/7] COPY model.py .                       0.1s
 => [7/7] COPY serve_api.py .                   0.1s
 => exporting to image                         15.9s
 => => naming to docker.io/library/code-assistant:v1

✅ Successfully tagged code-assistant:v1
"""

# 查看镜像
docker images code-assistant

# 输出:
# REPOSITORY         TAG    IMAGE ID       CREATED        SIZE
# code-assistant     v1     a1b2c3d4e5f6   1 minute ago   4.2GB
```

**第2步：运行容器**

```bash
# 运行容器（GPU版本）
docker run -d \
  --gpus all \
  -p 8000:8000 \
  --name code-assistant \
  --restart unless-stopped \
  code-assistant:v1

# 参数详解:
#   -d                    后台运行（detached）
#   --gpus all            使用所有GPU
#   -p 8000:8000          端口映射（主机:容器）
#   --name code-assistant 容器名称
#   --restart unless-stopped  自动重启（除非手动停止）
#   code-assistant:v1     使用的镜像

# 预期输出:
# a7f8e9d0c1b2...（容器ID）

# 查看运行状态
docker ps

# 输出:
"""
CONTAINER ID   IMAGE                 STATUS         PORTS
a7f8e9d0c1b2   code-assistant:v1     Up 10 seconds  0.0.0.0:8000->8000/tcp
"""
```

**第3步：查看日志**

```bash
# 实时查看日志
docker logs -f code-assistant

# 预期输出（和直接运行一样）:
"""
============================================================
🚀 启动API服务
============================================================
📥 加载模型...
✅ 模型已加载到 cuda
...
✅ API服务就绪！
============================================================
"""

# 查看最近100行日志
docker logs --tail 100 code-assistant

# 如果有问题，加上时间戳
docker logs -f --timestamps code-assistant
```

**第4步：测试容器化服务**

```bash
# 健康检查
curl http://localhost:8000/health

# 代码补全测试
curl -X POST http://localhost:8000/complete \
  -H "Content-Type: application/json" \
  -d '{"code": "def hello():\n    ", "max_tokens": 20}'

# 如果都正常，容器化成功！✅
```

### 🛠️ 6.6 Docker常用命令

```bash
# ===== 容器管理 =====

# 查看运行中的容器
docker ps

# 查看所有容器（包括停止的）
docker ps -a

# 停止容器
docker stop code-assistant

# 启动容器
docker start code-assistant

# 重启容器
docker restart code-assistant

# 删除容器
docker rm code-assistant

# 强制删除运行中的容器
docker rm -f code-assistant

# ===== 镜像管理 =====

# 查看镜像
docker images

# 删除镜像
docker rmi code-assistant:v1

# 清理未使用的镜像
docker image prune

# ===== 调试 =====

# 进入容器（查看内部）
docker exec -it code-assistant bash

# 查看容器详情
docker inspect code-assistant

# 查看容器资源使用
docker stats code-assistant

# 复制文件到容器
docker cp local_file.txt code-assistant:/app/

# 从容器复制文件
docker cp code-assistant:/app/output.txt ./
```

### 📊 6.7 镜像优化（减小体积）

```dockerfile
# Dockerfile.optimized - 优化版本

# ===== 优化1：使用多阶段构建 =====
# 构建阶段（包含编译工具）
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as builder

WORKDIR /build
COPY requirements.txt .
RUN pip3 install --user --no-cache-dir -r requirements.txt

# 运行阶段（只包含运行时）
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# ===== 优化2：只安装必需软件 =====
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ===== 优化3：从builder复制已安装的包 =====
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# 复制应用
COPY model.py serve_api.py ./
COPY out-code-assistant/ckpt_int8.pt ./out-code-assistant/

EXPOSE 8000
CMD ["python3", "serve_api.py"]

# 效果对比:
# 原始: 4.2GB
# 优化: 2.8GB （节省33%）
```

### ✅ 6.8 容器化检查清单

```python
功能验证:
  □ 镜像构建成功
  □ 容器能够启动
  □ API服务可访问
  □ GPU在容器内可用
  □ 模型加载正常

性能验证:
  □ 启动时间合理（<1分钟）
  □ 推理性能不降低
  □ 内存使用正常
  □ 日志输出正常

安全验证:
  □ 不使用root用户运行
  □ 最小化安装包
  □ 固定版本号
  □ 扫描安全漏洞

下一步:
  □ 推送到镜像仓库（Docker Hub/私有仓库）
  □ 使用Docker Compose管理多容器
  □ 部署到Kubernetes
```

### 🚢 6.9 推送镜像到仓库

```bash
# ===== 推送到Docker Hub =====

# 1. 登录Docker Hub
docker login

# 2. 给镜像打标签（加上用户名）
docker tag code-assistant:v1 yourusername/code-assistant:v1

# 3. 推送
docker push yourusername/code-assistant:v1

# 4. 在其他机器上拉取
docker pull yourusername/code-assistant:v1
docker run -d --gpus all -p 8000:8000 yourusername/code-assistant:v1

# ===== 推送到私有仓库 =====

# 1. 标签格式：registry.example.com/code-assistant:v1
docker tag code-assistant:v1 registry.example.com/code-assistant:v1

# 2. 推送到私有仓库
docker push registry.example.com/code-assistant:v1
```

---

## 📚 第七部分：Kubernetes编排（管理多个容器）

### 💡 7.1 为什么需要Kubernetes？

**生活比喻：连锁餐厅管理**

```
单个Docker容器:
  就像开一家小餐馆
  ✅ 自己管理方便
  ❌ 倒了就没了（单点故障）
  ❌ 忙不过来（不能自动扩容）
  ❌ 手动管理麻烦
  
Kubernetes（K8s）:
  就像连锁餐厅的管理系统
  ✅ 自动开关分店（自动扩缩容）
  ✅ 某家店倒了自动补（自愈能力）
  ✅ 统一管理调度（负载均衡）
  ✅ 监控所有分店（健康检查）
```

### 📊 7.2 Kubernetes核心概念

```python
K8s四大组件（由简单到复杂）:

1. Pod（容器组）
   最小部署单元
   包含1个或多个容器
   共享网络和存储
   
   比喻：一个餐厅的厨房团队

2. Deployment（部署）
   管理Pod的副本数
   支持滚动更新
   自动重启失败的Pod
   
   比喻：餐厅扩张计划（要开几家分店）

3. Service（服务）
   为Pod提供稳定的访问入口
   负载均衡
   服务发现
   
   比喻：餐厅的总机号（打这个号会分配到空闲的店）

4. HPA（水平自动扩缩容）
   根据CPU/内存自动调整Pod数量
   高峰期增加，低峰期减少
   
   比喻：根据客流量自动开关分店

工作流程:
  Deployment → 创建多个Pod → Service暴露访问 → HPA自动调整
```

### 🔧 7.3 创建Kubernetes配置（逐步理解）

```yaml
# k8s/deployment.yaml - Kubernetes部署配置

# ===== 第1部分：Deployment（部署管理）=====
apiVersion: apps/v1
kind: Deployment
metadata:
  name: code-assistant
  labels:
    app: code-assistant
spec:
  # 副本数：初始运行3个Pod
  replicas: 3
  
  # 选择器：管理哪些Pod
  selector:
    matchLabels:
      app: code-assistant
  
  # Pod模板：定义Pod的样子
  template:
    metadata:
      labels:
        app: code-assistant
    
    spec:
      # 容器配置
      containers:
      - name: code-assistant
        image: code-assistant:v1  # 使用我们构建的镜像
        
        # 端口配置
        ports:
        - containerPort: 8000
          name: http
        
        # 资源配置（重要！）
        resources:
          # 资源限制（不能超过）
          limits:
            nvidia.com/gpu: 1   # 每个Pod一个GPU
            memory: "8Gi"        # 最多用8GB内存
            cpu: "4"             # 最多用4核CPU
          
          # 资源请求（保证分配）
          requests:
            nvidia.com/gpu: 1   # 必须有1个GPU
            memory: "4Gi"        # 至少4GB内存
            cpu: "2"             # 至少2核CPU
        
        # 环境变量（可选）
        env:
        - name: MODEL_PATH
          value: "/app/out-code-assistant/ckpt_int8.pt"
        - name: LOG_LEVEL
          value: "INFO"
        
        # 存活探针（判断容器是否还活着）
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30  # 启动后30秒才开始检查
          periodSeconds: 10        # 每10秒检查一次
          timeoutSeconds: 5        # 5秒无响应算失败
          failureThreshold: 3      # 连续3次失败就重启
        
        # 就绪探针（判断容器是否准备好接收流量）
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10  # 启动后10秒开始检查
          periodSeconds: 5         # 每5秒检查一次
          timeoutSeconds: 3        # 3秒无响应算失败
          failureThreshold: 2      # 连续2次失败就标记为未就绪

---

# ===== 第2部分：Service（服务暴露）=====
apiVersion: v1
kind: Service
metadata:
  name: code-assistant-service
  labels:
    app: code-assistant
spec:
  # 选择要暴露的Pod
  selector:
    app: code-assistant
  
  # 端口配置
  ports:
  - name: http
    port: 80           # Service对外端口
    targetPort: 8000   # Pod内部端口
    protocol: TCP
  
  # 服务类型
  type: LoadBalancer   # 云环境会自动创建负载均衡器
  
  # 其他类型:
  # ClusterIP: 只能集群内访问（默认）
  # NodePort: 通过节点IP+端口访问
  # LoadBalancer: 云提供商的负载均衡器（推荐）

---

# ===== 第3部分：HPA（自动扩缩容）=====
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: code-assistant-hpa
spec:
  # 要管理的Deployment
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: code-assistant
  
  # Pod数量范围
  minReplicas: 3      # 最少3个Pod（保证高可用）
  maxReplicas: 10     # 最多10个Pod（控制成本）
  
  # 扩缩容指标
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70  # CPU使用率超过70%就扩容
  
  # 扩缩容行为（可选，更精细控制）
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60   # 扩容前观察60秒
      policies:
      - type: Percent
        value: 50          # 每次最多增加50%的Pod
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300  # 缩容前观察5分钟
      policies:
      - type: Pods
        value: 1           # 每次最多减少1个Pod
        periodSeconds: 60
```

### 🚀 7.4 部署到Kubernetes

**第1步：准备K8s集群**

```bash
# 如果是学习/测试，可以用minikube
minikube start --gpus all

# 如果是云环境（推荐）
# AWS EKS, Google GKE, Azure AKS等
# 或者自建Kubernetes集群

# 验证集群
kubectl cluster-info
kubectl get nodes

# 确保节点有GPU
kubectl describe nodes | grep nvidia.com/gpu
```

**第2步：部署应用**

```bash
# 应用配置
kubectl apply -f k8s/deployment.yaml

# 预期输出:
"""
deployment.apps/code-assistant created
service/code-assistant-service created
horizontalpodautoscaler.autoscaling/code-assistant-hpa created
"""

# 查看创建的资源
kubectl get all

# 输出:
"""
NAME                                  READY   STATUS    RESTARTS   AGE
pod/code-assistant-7d8f9c5b6d-abc12   1/1     Running   0          30s
pod/code-assistant-7d8f9c5b6d-def34   1/1     Running   0          30s
pod/code-assistant-7d8f9c5b6d-ghi56   1/1     Running   0          30s

NAME                             TYPE           CLUSTER-IP      EXTERNAL-IP   PORT(S)
service/code-assistant-service   LoadBalancer   10.100.200.50   <pending>     80:31234/TCP

NAME                             READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/code-assistant   3/3     3            3           30s

NAME                                        DESIRED   CURRENT   READY   AGE
replicaset.apps/code-assistant-7d8f9c5b6d   3         3         3       30s

NAME                                                 REFERENCE                   TARGETS   MINPODS   MAXPODS
horizontalpodautoscaler.autoscaling/code-assistant   Deployment/code-assistant   45%/70%   3         10
"""
```

**第3步：查看Pod状态**

```bash
# 查看Pod列表
kubectl get pods

# 查看Pod详情
kubectl describe pod code-assistant-7d8f9c5b6d-abc12

# 查看Pod日志（实时）
kubectl logs -f code-assistant-7d8f9c5b6d-abc12

# 查看所有Pod的日志
kubectl logs -f deployment/code-assistant
```

**第4步：测试服务**

```bash
# 方式1：端口转发（本地测试）
kubectl port-forward service/code-assistant-service 8000:80

# 在另一个终端测试
curl http://localhost:8000/health

# 方式2：通过LoadBalancer IP（生产环境）
# 等待外部IP分配
kubectl get service code-assistant-service

# 输出:
# NAME                       TYPE           EXTERNAL-IP      PORT(S)
# code-assistant-service     LoadBalancer   34.123.45.67     80:31234/TCP

# 测试
curl http://34.123.45.67/health
```

### 📊 7.5 K8s常用管理命令

```bash
# ===== Pod管理 =====

# 查看Pod
kubectl get pods
kubectl get pods -o wide  # 显示更多信息（节点、IP）

# 查看Pod详情
kubectl describe pod <pod-name>

# 进入Pod（调试）
kubectl exec -it <pod-name> -- bash

# 删除Pod（会自动重建）
kubectl delete pod <pod-name>

# ===== Deployment管理 =====

# 查看Deployment
kubectl get deployment

# 修改副本数
kubectl scale deployment code-assistant --replicas=5

# 更新镜像（滚动更新）
kubectl set image deployment/code-assistant \
  code-assistant=code-assistant:v2

# 查看更新状态
kubectl rollout status deployment/code-assistant

# 回滚到上一个版本
kubectl rollout undo deployment/code-assistant

# ===== Service管理 =====

# 查看Service
kubectl get service
kubectl get svc  # 简写

# 查看端点（实际的Pod IP）
kubectl get endpoints

# ===== HPA管理 =====

# 查看HPA状态
kubectl get hpa

# 查看HPA详情
kubectl describe hpa code-assistant-hpa

# ===== 日志和监控 =====

# 查看Pod日志
kubectl logs <pod-name>
kubectl logs -f <pod-name>  # 实时
kubectl logs --tail=100 <pod-name>  # 最近100行

# 查看事件
kubectl get events --sort-by='.lastTimestamp'

# 查看资源使用
kubectl top nodes  # 节点资源
kubectl top pods   # Pod资源

# ===== 清理资源 =====

# 删除所有资源
kubectl delete -f k8s/deployment.yaml

# 或者单独删除
kubectl delete deployment code-assistant
kubectl delete service code-assistant-service
kubectl delete hpa code-assistant-hpa
```

### 🔄 7.6 滚动更新（零停机部署）

```bash
# 场景：发布新版本v2

# 1. 构建新镜像
docker build -t code-assistant:v2 .
docker push yourusername/code-assistant:v2

# 2. 更新Deployment
kubectl set image deployment/code-assistant \
  code-assistant=yourusername/code-assistant:v2

# 3. 观察滚动更新过程
kubectl rollout status deployment/code-assistant

# 输出（实时更新）:
"""
Waiting for deployment "code-assistant" rollout to finish:
1 out of 3 new replicas have been updated...
2 out of 3 new replicas have been updated...
3 out of 3 new replicas have been updated...
Waiting for 1 old replicas to be terminated...
deployment "code-assistant" successfully rolled out
"""

# 4. 验证新版本
kubectl get pods  # 应该看到新的Pod

# 5. 如果有问题，立即回滚
kubectl rollout undo deployment/code-assistant

# K8s的滚动更新策略:
# - 逐个替换Pod（默认一次替换25%）
# - 新Pod就绪后才删除旧Pod
# - 确保始终有足够的Pod在运行
# - 零停机时间！
```

### ✅ 7.7 Kubernetes检查清单

```python
部署验证:
  □ 所有Pod都在Running状态
  □ Service有外部IP（LoadBalancer）
  □ 健康检查通过（/health返回200）
  □ 能够正常处理请求
  □ HPA正常工作

高可用验证:
  □ 至少3个副本
  □ Pod分布在不同节点
  □ 手动删除Pod会自动重建
  □ 滚动更新零停机
  
性能验证:
  □ 负载均衡正常
  □ 扩缩容符合预期
  □ 资源使用合理
  □ 延迟满足要求

下一步:
  □ 配置监控（Prometheus）
  □ 配置日志（ELK/Loki）
  □ 配置告警
  □ 备份和灾难恢复
```

---

## 📚 第八部分：监控与运维（让服务稳定运行）

### 💡 8.1 为什么需要监控？

**生活比喻：开餐厅需要仪表盘**

```
没有监控的服务:
  就像蒙着眼开餐厅
  ❓ 今天来了多少客人？不知道
  ❓ 厨房效率如何？不知道
  ❓ 是不是要挂了？不知道
  💥 等发现问题时，已经出大事了
  
有监控的服务:
  就像装满仪表盘的餐厅
  ✅ 实时客流量（QPS）
  ✅ 平均等待时间（延迟）
  ✅ 错误率（失败的订单）
  ✅ 资源使用（厨房负载）
  🚨 问题还没发生就预警
```

### 📊 8.2 监控的四个黄金指标

```python
生产环境必须监控的指标（RED方法）:

1. Rate（请求速率）
   每秒处理多少请求？
   指标：QPS（Queries Per Second）
   正常：10-100 QPS
   异常：突然暴涨或归零
   
2. Errors（错误率）
   多少请求失败了？
   指标：错误率 = 失败请求/总请求
   正常：< 1%
   异常：> 5%（严重）
   
3. Duration（延迟）
   请求需要多久响应？
   指标：P50, P95, P99延迟
   正常：P95 < 200ms
   异常：P95 > 1s
   
4. Saturation（资源饱和度）
   系统资源还剩多少？
   指标：CPU、内存、GPU使用率
   正常：< 70%
   异常：> 90%（快挂了）

比喻：
  Rate = 客流量
  Errors = 投诉率
  Duration = 等餐时间
  Saturation = 厨房忙碌度
```

### 🔧 8.3 添加Prometheus监控（完整实现）

**第1步：安装依赖**

```bash
pip install prometheus-client
```

**第2步：完整的监控API实现**

```python
# serve_api_with_metrics.py - 带完整监控的API服务

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import torch
import tiktoken
import time
import logging

# ===== Prometheus监控 =====
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app

# 定义四大黄金指标
# 1. Rate: 请求计数器（按状态分类）
request_count = Counter(
    'code_completion_requests_total',
    'Total number of code completion requests',
    ['status']  # 标签：success, error
)

# 2. Duration: 请求延迟直方图
request_duration = Histogram(
    'code_completion_duration_seconds',
    'Duration of code completion requests in seconds',
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]  # 延迟分桶
)

# 3. Errors: 在request_count中已包含（status='error'）

# 4. Saturation: 活跃请求数（并发）
active_requests = Gauge(
    'active_requests',
    'Number of requests currently being processed'
)

# 额外指标：生成的token数量
tokens_generated = Counter(
    'tokens_generated_total',
    'Total number of tokens generated'
)

# GPU内存使用（如果有GPU）
gpu_memory_usage = Gauge(
    'gpu_memory_used_bytes',
    'GPU memory used in bytes'
)

# 模型加载状态
model_loaded = Gauge(
    'model_loaded',
    'Whether model is loaded (1=loaded, 0=not loaded)'
)

# ===== 配置日志 =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===== 全局变量 =====
model = None
tokenizer = None
device = None

# ===== 生命周期管理 =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动和关闭时的操作"""
    global model, tokenizer, device
    
    logger.info("Loading model...")
    model_loaded.set(0)
    
    try:
        # 加载模型
        checkpoint = torch.load('out-code-assistant/ckpt_int8.pt', map_location='cpu')
        model = checkpoint['model']
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()
        
        # 加载tokenizer
        tokenizer = tiktoken.get_encoding("gpt2")
        
        logger.info(f"Model loaded on {device}")
        model_loaded.set(1)
        
        yield  # 应用运行中
        
    finally:
        # 清理资源
        logger.info("Shutting down...")
        model_loaded.set(0)

# ===== 创建FastAPI应用 =====
app = FastAPI(
    title="Code Assistant API with Monitoring",
    version="1.0.0",
    lifespan=lifespan
)

# 挂载Prometheus metrics端点
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# ===== 数据模型 =====
class CompletionRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50
    temperature: float = 0.8
    top_k: int = 200

class CompletionResponse(BaseModel):
    completion: str
    latency_ms: float
    tokens_generated: int

# ===== API端点 =====

@app.get("/health")
async def health_check():
    """健康检查"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(device)
    }

@app.post("/complete", response_model=CompletionResponse)
async def complete_code(request: CompletionRequest):
    """代码补全（带完整监控）"""
    
    # 增加活跃请求计数
    active_requests.inc()
    start_time = time.time()
    
    try:
        # 1. Tokenize输入
        input_ids = tokenizer.encode(request.prompt)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
        
        # 2. 截断过长输入
        max_input_length = 512 - request.max_new_tokens
        if input_ids.size(1) > max_input_length:
            input_ids = input_ids[:, -max_input_length:]
        
        # 3. 生成
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k
            )
        
        # 4. Decode
        completion = tokenizer.decode(output_ids[0].tolist())
        num_tokens = output_ids.size(1) - input_ids.size(1)
        
        # 5. 计算延迟
        latency = time.time() - start_time
        
        # ===== 记录监控指标 =====
        request_duration.observe(latency)
        request_count.labels(status='success').inc()
        tokens_generated.inc(num_tokens)
        
        # 记录GPU内存（如果有）
        if torch.cuda.is_available():
            gpu_memory_usage.set(torch.cuda.memory_allocated(device))
        
        logger.info(
            f"Request completed: latency={latency:.3f}s, "
            f"tokens={num_tokens}, prompt_length={len(request.prompt)}"
        )
        
        return CompletionResponse(
            completion=completion,
            latency_ms=latency * 1000,
            tokens_generated=num_tokens
        )
    
    except Exception as e:
        # 记录错误
        request_count.labels(status='error').inc()
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # 减少活跃请求计数
        active_requests.dec()

@app.get("/stats")
async def get_stats():
    """获取统计信息"""
    stats = {
        "model_loaded": model is not None,
        "device": str(device) if device else None,
    }
    
    if torch.cuda.is_available():
        stats["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated(device) / 1024 / 1024
        stats["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved(device) / 1024 / 1024
    
    return stats

# ===== 启动服务 =====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**第3步：启动服务并访问指标**

```bash
# 启动服务
python serve_api_with_metrics.py

# 访问Prometheus指标端点
curl http://localhost:8000/metrics

# 输出示例:
"""
# HELP code_completion_requests_total Total number of code completion requests
# TYPE code_completion_requests_total counter
code_completion_requests_total{status="success"} 123.0
code_completion_requests_total{status="error"} 5.0

# HELP code_completion_duration_seconds Duration of code completion requests
# TYPE code_completion_duration_seconds histogram
code_completion_duration_seconds_bucket{le="0.1"} 45.0
code_completion_duration_seconds_bucket{le="0.2"} 89.0
code_completion_duration_seconds_bucket{le="0.5"} 118.0
code_completion_duration_seconds_bucket{le="1.0"} 123.0
code_completion_duration_seconds_sum 78.5
code_completion_duration_seconds_count 123.0

# HELP active_requests Number of requests currently being processed
# TYPE active_requests gauge
active_requests 2.0

# HELP tokens_generated_total Total number of tokens generated
# TYPE tokens_generated_total counter
tokens_generated_total 6150.0
"""
```

### 📈 8.4 配置Prometheus抓取

```yaml
# prometheus.yml - Prometheus配置

global:
  scrape_interval: 15s      # 每15秒抓取一次
  evaluation_interval: 15s  # 每15秒评估告警规则

# 抓取配置
scrape_configs:
  - job_name: 'code-assistant'
    
    # 抓取目标（多个Pod）
    kubernetes_sd_configs:
    - role: pod
      namespaces:
        names:
        - default
    
    # 过滤：只抓取我们的服务
    relabel_configs:
    - source_labels: [__meta_kubernetes_pod_label_app]
      regex: code-assistant
      action: keep
    
    # 指标路径
    metrics_path: /metrics
    
    # 静态配置（测试用）
    # static_configs:
    # - targets: ['localhost:8000']

# 告警规则
rule_files:
  - 'alerts.yml'

# Alertmanager配置（发送告警）
alerting:
  alertmanagers:
  - static_configs:
    - targets: ['localhost:9093']
```

### 🚨 8.5 配置告警规则

```yaml
# alerts.yml - 告警规则

groups:
- name: code_assistant_alerts
  interval: 30s
  rules:
  
  # 告警1：高错误率
  - alert: HighErrorRate
    expr: |
      (
        rate(code_completion_requests_total{status="error"}[5m]) 
        / 
        rate(code_completion_requests_total[5m])
      ) > 0.05
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value | humanizePercentage }} (threshold: 5%)"
  
  # 告警2：高延迟
  - alert: HighLatency
    expr: |
      histogram_quantile(0.95, 
        rate(code_completion_duration_seconds_bucket[5m])
      ) > 1.0
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High P95 latency detected"
      description: "P95 latency is {{ $value }}s (threshold: 1s)"
  
  # 告警3：服务不可用
  - alert: ServiceDown
    expr: up{job="code-assistant"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Service is down"
      description: "{{ $labels.instance }} is not responding"
  
  # 告警4：GPU内存不足
  - alert: GPUMemoryHigh
    expr: |
      gpu_memory_used_bytes / (16 * 1024 * 1024 * 1024) > 0.9
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "GPU memory usage high"
      description: "GPU memory usage is {{ $value | humanizePercentage }}"
  
  # 告警5：请求堆积
  - alert: RequestQueueing
    expr: active_requests > 10
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Too many concurrent requests"
      description: "{{ $value }} active requests (threshold: 10)"
```

### 📊 8.6 配置Grafana仪表板

```json
{
  "dashboard": {
    "title": "Code Assistant Production Metrics",
    "tags": ["production", "ai", "code-assistant"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "请求速率（QPS）",
        "type": "graph",
        "targets": [{
          "expr": "sum(rate(code_completion_requests_total[1m]))",
          "legendFormat": "总QPS"
        }, {
          "expr": "sum(rate(code_completion_requests_total{status='success'}[1m]))",
          "legendFormat": "成功"
        }, {
          "expr": "sum(rate(code_completion_requests_total{status='error'}[1m]))",
          "legendFormat": "失败"
        }]
      },
      {
        "id": 2,
        "title": "延迟分布（P50/P95/P99）",
        "type": "graph",
        "targets": [{
          "expr": "histogram_quantile(0.50, sum(rate(code_completion_duration_seconds_bucket[5m])) by (le))",
          "legendFormat": "P50"
        }, {
          "expr": "histogram_quantile(0.95, sum(rate(code_completion_duration_seconds_bucket[5m])) by (le))",
          "legendFormat": "P95"
        }, {
          "expr": "histogram_quantile(0.99, sum(rate(code_completion_duration_seconds_bucket[5m])) by (le))",
          "legendFormat": "P99"
        }]
      },
      {
        "id": 3,
        "title": "错误率",
        "type": "singlestat",
        "targets": [{
          "expr": "sum(rate(code_completion_requests_total{status='error'}[5m])) / sum(rate(code_completion_requests_total[5m]))",
          "format": "percent"
        }],
        "thresholds": "0.01,0.05",
        "colors": ["green", "yellow", "red"]
      },
      {
        "id": 4,
        "title": "活跃请求数",
        "type": "graph",
        "targets": [{
          "expr": "sum(active_requests)",
          "legendFormat": "并发请求"
        }]
      },
      {
        "id": 5,
        "title": "Token生成速率",
        "type": "graph",
        "targets": [{
          "expr": "sum(rate(tokens_generated_total[1m]))",
          "legendFormat": "tokens/s"
        }]
      },
      {
        "id": 6,
        "title": "GPU内存使用",
        "type": "graph",
        "targets": [{
          "expr": "gpu_memory_used_bytes / (1024*1024*1024)",
          "legendFormat": "GPU内存 (GB)"
        }]
      }
    ],
    "refresh": "10s"
  }
}
```

### 🔍 8.7 日志管理最佳实践

```python
# 结构化日志（方便查询）

import logging
import json
from datetime import datetime

class StructuredLogger:
    """结构化日志记录器"""
    
    def __init__(self, name):
        self.logger = logging.getLogger(name)
    
    def log_request(self, request_id, prompt, latency, tokens, status):
        """记录请求日志"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "type": "request",
            "request_id": request_id,
            "prompt_length": len(prompt),
            "latency_ms": latency * 1000,
            "tokens_generated": tokens,
            "status": status
        }
        self.logger.info(json.dumps(log_data))
    
    def log_error(self, request_id, error, stack_trace):
        """记录错误日志"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "type": "error",
            "request_id": request_id,
            "error": str(error),
            "stack_trace": stack_trace
        }
        self.logger.error(json.dumps(log_data))

# 使用方式:
logger = StructuredLogger("code-assistant")

@app.post("/complete")
async def complete_code(request: CompletionRequest):
    request_id = str(uuid.uuid4())
    
    try:
        # ... 处理请求 ...
        
        logger.log_request(
            request_id=request_id,
            prompt=request.prompt,
            latency=latency,
            tokens=num_tokens,
            status="success"
        )
    
    except Exception as e:
        logger.log_error(
            request_id=request_id,
            error=e,
            stack_trace=traceback.format_exc()
        )
```

### ✅ 8.8 监控运维检查清单

```python
基础监控:
  □ Prometheus正常抓取指标
  □ Grafana仪表板显示正常
  □ 四大黄金指标都有
  □ /metrics端点正常工作

告警配置:
  □ 高错误率告警
  □ 高延迟告警
  □ 服务宕机告警
  □ 资源耗尽告警
  □ 告警能正常发送（邮件/钉钉/Slack）

日志管理:
  □ 结构化日志
  □ 日志聚合（ELK/Loki）
  □ 日志查询方便
  □ 日志保留策略（7天/30天）

健康检查:
  □ /health端点正常
  □ Liveness探针配置
  □ Readiness探针配置
  □ 自动重启失败Pod

性能指标:
  □ QPS: 监控请求速率
  □ 延迟: P95 < 200ms
  □ 错误率: < 1%
  □ 资源: CPU/GPU < 70%

下一步:
  □ 配置SLO（服务等级目标）
  □ 配置SLA（服务等级协议）
  □ 制定应急预案
  □ 定期演练故障恢复
```

---

## 📚 第九部分：性能优化（让服务更快更省钱）

### 💡 9.1 为什么需要性能优化？

**生活比喻：优化就是降本增效**

```
没优化的服务:
  就像开高油耗的卡车送外卖
  🐌 跑得慢（客户等太久）
  💸 油耗高（成本高）
  😓 累得快（资源不够用）
  💔 赚不到钱（入不敷出）
  
优化后的服务:
  就像换成电动摩托
  🚀 跑得快（延迟低）
  💰 省油钱（成本低）
  💪 能跑更久（资源充足）
  😊 利润高（有钱赚）

核心目标：
  更快 + 更便宜 + 更稳定
```

### 📊 9.2 性能优化全景图

```python
优化的四个维度（从易到难）:

1. 模型优化（减肥瘦身）
   目标：模型更小更快
   ├── ✅ INT8量化 (4x压缩, 我们已做)
   ├── 🔧 INT4量化 (8x压缩, 更激进)
   ├── 🔧 模型剪枝 (删除不重要的参数)
   └── 🔧 知识蒸馏 (训练小模型模仿大模型)
   
   收益：💰💰💰💰💰 (成本减少50-75%)
   难度：⭐⭐ (中等)

2. 推理优化（提速加快）
   目标：生成速度更快
├── ✅ KV Cache (已在model.py中)
   ├── 🔧 vLLM (专业推理引擎, 10-20x加速)
   ├── 🔧 TensorRT (NVIDIA优化, 极致性能)
   └── 🔧 Continuous Batching (批处理优化)
   
   收益：🚀🚀🚀🚀 (速度提升5-20x)
   难度：⭐⭐⭐ (较难)

3. 服务优化（架构改进）
   目标：系统更高效
   ├── ✅ 异步API (FastAPI, 已实现)
   ├── 🔧 请求缓存 (Redis, 缓存常见请求)
   ├── 🔧 负载均衡 (Nginx, 分散流量)
   └── 🔧 API Gateway (统一入口)
   
   收益：💡💡💡 (吞吐量提升2-3x)
   难度：⭐⭐ (中等)

4. 基础设施优化（省钱省心）
   目标：成本更低
   ├── ✅ Docker容器化 (已完成)
   ├── ✅ K8s编排 (已完成)
   ├── ✅ 自动扩缩容 HPA (已完成)
   └── 🔧 Spot实例 (节省70%成本)
   
   收益：💰💰💰 (成本降低30-70%)
   难度：⭐ (简单)

优化顺序建议:
  1. 先做低垂的果实（Spot实例、缓存）
  2. 再做模型优化（量化、剪枝）
  3. 最后做推理优化（vLLM、TensorRT）
```

### 💰 9.3 成本分析与优化策略

**第1步：计算当前成本**

```python
# cost_calculator.py - 成本计算器

def calculate_monthly_cost(
    gpu_type="A10G",
    num_gpus=3,
    requests_per_day=10000,
    avg_tokens_per_request=50,
):
    """计算每月运行成本"""
    
    # 1. GPU小时成本（云服务商价格）
    gpu_hourly_cost = {
        "T4": 0.35,    # 入门级
        "A10G": 1.00,  # 中端（推荐）
        "A100": 3.67,  # 高端
    }
    
    cost_per_hour = gpu_hourly_cost[gpu_type]
    hours_per_month = 24 * 30
    
    # 2. 基础成本（GPU一直开着）
    base_cost = cost_per_hour * num_gpus * hours_per_month
    
    # 3. 请求量统计
    total_requests = requests_per_day * 30
    total_tokens = total_requests * avg_tokens_per_request
    
    # 4. 每千次请求成本
    cost_per_1k_requests = (base_cost / total_requests) * 1000
    
    # 5. 每千个token成本
    cost_per_1k_tokens = (base_cost / total_tokens) * 1000
    
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║               💰 当前成本分析                              ║
╠═══════════════════════════════════════════════════════════╣
║ GPU配置:     {num_gpus}x {gpu_type:<10}                       ║
║ 单价:        ${cost_per_hour:.2f}/小时                        ║
║ 月度成本:    ${base_cost:,.2f}/月                          ║
║ ─────────────────────────────────────────────────────────║
║ 月度请求:    {total_requests:,} 次                         ║
║ 月度Token:   {total_tokens:,} 个                           ║
║ ─────────────────────────────────────────────────────────║
║ 每1K请求:    ${cost_per_1k_requests:.4f}                    ║
║ 每1K Token:  ${cost_per_1k_tokens:.6f}                      ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    return base_cost, cost_per_1k_requests

# 示例：计算当前配置成本
current_cost, _ = calculate_monthly_cost(
    gpu_type="A10G",
    num_gpus=3,
    requests_per_day=10000
)

# 输出:
"""
╔═══════════════════════════════════════════════════════════╗
║               💰 当前成本分析                              ║
╠═══════════════════════════════════════════════════════════╣
║ GPU配置:     3x A10G                                       ║
║ 单价:        $1.00/小时                                    ║
║ 月度成本:    $2,190.00/月                                  ║
║ ─────────────────────────────────────────────────────────║
║ 月度请求:    300,000 次                                    ║
║ 月度Token:   15,000,000 个                                 ║
║ ─────────────────────────────────────────────────────────║
║ 每1K请求:    $7.3000                                       ║
║ 每1K Token:  $0.146000                                     ║
╚═══════════════════════════════════════════════════════════╝
"""
```

**第2步：优化方案对比**

```python
# 优化策略对比

def compare_optimization_strategies():
    """对比不同优化策略的效果"""
    
    # 基准配置
    baseline = {
        "name": "当前配置（未优化）",
        "gpu_type": "A10G",
        "num_gpus": 3,
        "monthly_cost": 2190,
        "latency_ms": 200,
        "throughput_qps": 100,
    }
    
    # 优化方案
    strategies = [
        {
            "name": "方案1：使用Spot实例",
            "description": "使用云提供商的闲置GPU（可能被中断）",
            "gpu_type": "A10G (Spot)",
            "num_gpus": 3,
            "monthly_cost": 657,      # 节省70%
            "latency_ms": 200,         # 延迟不变
            "throughput_qps": 100,     # 吞吐不变
            "difficulty": "⭐ 简单",
            "risk": "⚠️ 可能被中断（需要容错）",
        },
        {
            "name": "方案2：INT8量化（已完成）",
            "description": "模型压缩4倍，速度提升2倍",
            "gpu_type": "A10G",
            "num_gpus": 2,             # GPU减少1个
            "monthly_cost": 1460,      # 节省33%
            "latency_ms": 150,         # 延迟降低
            "throughput_qps": 150,     # 吞吐提升
            "difficulty": "⭐⭐ 中等",
            "risk": "✅ 精度略降（可接受）",
        },
        {
            "name": "方案3：vLLM推理引擎",
            "description": "专业推理优化，10-20x加速",
            "gpu_type": "A10G",
            "num_gpus": 1,             # 只需1个GPU！
            "monthly_cost": 730,       # 节省67%
            "latency_ms": 100,         # 延迟大幅降低
            "throughput_qps": 500,     # 吞吐大幅提升
            "difficulty": "⭐⭐⭐ 较难",
            "risk": "✅ 需要适配代码",
        },
        {
            "name": "方案4：综合优化（最佳）",
            "description": "Spot实例 + INT8量化 + vLLM",
            "gpu_type": "A10G (Spot)",
            "num_gpus": 1,
            "monthly_cost": 219,       # 节省90%！
            "latency_ms": 100,
            "throughput_qps": 500,
            "difficulty": "⭐⭐⭐ 较难",
            "risk": "⚠️ 需要容错机制",
        },
    ]
    
    # 打印对比表
    print("\n╔════════════════════════════════════════════════════════════════════════════════╗")
    print("║                        🚀 优化方案对比                                          ║")
    print("╠════════════════════════════════════════════════════════════════════════════════╣")
    print(f"║ {'方案':<20} │ {'成本/月':<12} │ {'延迟':<10} │ {'吞吐':<10} │ {'难度':<10} ║")
    print("╠════════════════════════════════════════════════════════════════════════════════╣")
    
    # 基准
    print(f"║ {baseline['name']:<20} │ ${baseline['monthly_cost']:>6,.0f} (基准) │ "
          f"{baseline['latency_ms']:>4}ms     │ {baseline['throughput_qps']:>4} QPS   │ {'N/A':<10} ║")
    
    # 优化方案
    for s in strategies:
        savings = (baseline['monthly_cost'] - s['monthly_cost']) / baseline['monthly_cost'] * 100
        print(f"║ {s['name']:<20} │ ${s['monthly_cost']:>6,.0f} (↓{savings:>3.0f}%) │ "
              f"{s['latency_ms']:>4}ms     │ {s['throughput_qps']:>4} QPS   │ {s['difficulty']:<10} ║")
    
    print("╚════════════════════════════════════════════════════════════════════════════════╝\n")
    
    # 详细说明
    for i, s in enumerate(strategies, 1):
        print(f"\n{s['name']}:")
        print(f"  📝 {s['description']}")
        print(f"  💰 成本: ${s['monthly_cost']}/月")
        print(f"  ⚡ 性能: {s['latency_ms']}ms延迟, {s['throughput_qps']} QPS")
        print(f"  🔧 难度: {s['difficulty']}")
        print(f"  ⚠️  风险: {s['risk']}")

# 运行对比
compare_optimization_strategies()
```

**第3步：实施Spot实例（低垂的果实）**

```yaml
# k8s/deployment-spot.yaml - 使用Spot实例的K8s配置

apiVersion: apps/v1
kind: Deployment
metadata:
  name: code-assistant-spot
spec:
  replicas: 3
  selector:
    matchLabels:
      app: code-assistant
  template:
    metadata:
      labels:
        app: code-assistant
    spec:
      # 关键配置：使用Spot实例
      nodeSelector:
        # AWS
        eks.amazonaws.com/capacityType: SPOT
        # 或 GCP
        # cloud.google.com/gke-preemptible: "true"
        # 或 Azure
        # kubernetes.azure.com/scalesetpriority: spot
      
      # 容错配置（Spot可能被中断）
      tolerations:
      - key: "spot"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"
      
      # 优雅关闭（给30秒保存状态）
      terminationGracePeriodSeconds: 30
      
      containers:
      - name: code-assistant
        image: code-assistant:v1
        # ... 其他配置同前 ...
        
        # 生命周期钩子（被中断前保存状态）
        lifecycle:
          preStop:
            exec:
              command: ["/bin/sh", "-c", "sleep 15"]  # 等待请求处理完

---

# 配置PodDisruptionBudget（确保始终有Pod在运行）
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: code-assistant-pdb
spec:
  minAvailable: 1  # 至少1个Pod可用
  selector:
    matchLabels:
      app: code-assistant
```

**部署Spot实例**

```bash
# 部署到Spot实例
kubectl apply -f k8s/deployment-spot.yaml

# 验证节点类型
kubectl get nodes -L eks.amazonaws.com/capacityType

# 输出:
"""
NAME                STATUS   CAPACITYTYPE
ip-10-0-1-10.ec2    Ready    SPOT
ip-10-0-1-11.ec2    Ready    SPOT
ip-10-0-1-12.ec2    Ready    ON_DEMAND  # 保留1个按需实例做备份
"""

# 监控Spot中断事件
kubectl get events | grep -i spot

# 成本节省效果:
# 原价: $2,190/月
# 现在: $657/月
# 节省: $1,533/月 (70%)
```

### 🚀 9.4 推理优化：vLLM集成（进阶）

**为什么vLLM这么快？**

```python
原生PyTorch推理的问题:
  1. 每次只处理1个请求（串行）
  2. KV Cache管理低效（内存碎片）
  3. 没有充分利用GPU（算力浪费）
  
  结果：QPS低、延迟高、成本高

vLLM的黑科技:
  1. PagedAttention（分页注意力机制）
     → 高效管理KV Cache，减少内存浪费
  
  2. Continuous Batching（连续批处理）
     → 同时处理多个请求，提升GPU利用率
  
  3. 内核优化（CUDA Kernels）
     → 专门为Transformer优化的底层实现
  
  结果：QPS提升10-20x，延迟降低50%+

比喻：
  PyTorch = 单线程餐厅（一次做一个菜）
  vLLM = 流水线餐厅（同时做多个菜，效率爆炸）
```

**集成vLLM（完整实现）**

```python
# serve_vllm.py - 使用vLLM的高性能API服务

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from contextlib import asynccontextmanager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== 全局变量 =====
llm_engine = None

# ===== 生命周期管理 =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载vLLM引擎"""
    global llm_engine
    
    logger.info("Initializing vLLM engine...")
    
    # 创建vLLM引擎
    llm_engine = LLM(
        model="out-code-assistant/",  # 模型路径
        tensor_parallel_size=1,        # 单GPU
        dtype="float16",               # 使用FP16
        max_model_len=512,             # 最大序列长度
        gpu_memory_utilization=0.9,    # GPU显存使用率
        trust_remote_code=True,        # 信任自定义代码
    )
    
    logger.info("vLLM engine loaded successfully")
    yield
    
    logger.info("Shutting down vLLM engine...")

# ===== FastAPI应用 =====
app = FastAPI(
    title="Code Assistant with vLLM",
    version="2.0.0",
    lifespan=lifespan
)

# ===== 数据模型 =====
class CompletionRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50
    temperature: float = 0.8
    top_k: int = 200

class CompletionResponse(BaseModel):
    completion: str
    latency_ms: float

# ===== API端点 =====

@app.get("/health")
async def health_check():
    """健康检查"""
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="vLLM engine not loaded")
    return {"status": "healthy", "engine": "vLLM"}

@app.post("/complete", response_model=CompletionResponse)
async def complete_code(request: CompletionRequest):
    """代码补全（vLLM加速）"""
    import time
    start_time = time.time()
    
    try:
        # 创建采样参数
        sampling_params = SamplingParams(
            max_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
        )
        
        # vLLM生成（自动批处理！）
        outputs = llm_engine.generate(
            prompts=[request.prompt],
            sampling_params=sampling_params
        )
        
        # 提取结果
        completion = outputs[0].outputs[0].text
        latency = time.time() - start_time
        
        logger.info(f"Completed in {latency:.3f}s")
        
        return CompletionResponse(
            completion=completion,
            latency_ms=latency * 1000
        )
    
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== 启动服务 =====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**性能对比测试**

```python
# benchmark_vllm.py - 对比vLLM和原生PyTorch

import requests
import time
import concurrent.futures

def test_api(url, num_requests=100, concurrency=10):
    """压测API"""
    
    def single_request():
        start = time.time()
        response = requests.post(
            f"{url}/complete",
            json={
                "prompt": "def fibonacci(n):",
                "max_new_tokens": 50
            }
        )
        latency = time.time() - start
        return latency
    
    # 并发测试
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        start_time = time.time()
        latencies = list(executor.map(lambda _: single_request(), range(num_requests)))
        total_time = time.time() - start_time
    
    # 统计
    avg_latency = sum(latencies) / len(latencies)
    qps = num_requests / total_time
    
    return {
        "avg_latency_ms": avg_latency * 1000,
        "qps": qps,
        "total_time": total_time,
    }

# 测试原生PyTorch
print("测试原生PyTorch...")
pytorch_results = test_api("http://localhost:8000", num_requests=100, concurrency=10)

# 测试vLLM
print("测试vLLM...")
vllm_results = test_api("http://localhost:8001", num_requests=100, concurrency=10)

# 对比
print(f"""
╔════════════════════════════════════════════════════════╗
║            🚀 vLLM vs PyTorch 性能对比                  ║
╠════════════════════════════════════════════════════════╣
║ 指标              │ PyTorch      │ vLLM        │ 提升   ║
╠════════════════════════════════════════════════════════╣
║ 平均延迟          │ {pytorch_results['avg_latency_ms']:>6.1f}ms    │ {vllm_results['avg_latency_ms']:>6.1f}ms  │ {pytorch_results['avg_latency_ms']/vllm_results['avg_latency_ms']:.1f}x   ║
║ 吞吐量(QPS)       │ {pytorch_results['qps']:>6.1f}      │ {vllm_results['qps']:>6.1f}    │ {vllm_results['qps']/pytorch_results['qps']:.1f}x   ║
║ 总耗时            │ {pytorch_results['total_time']:>6.1f}s     │ {vllm_results['total_time']:>6.1f}s   │ {pytorch_results['total_time']/vllm_results['total_time']:.1f}x   ║
╚════════════════════════════════════════════════════════╝

结论:
  ✅ vLLM延迟降低 {(1 - vllm_results['avg_latency_ms']/pytorch_results['avg_latency_ms'])*100:.0f}%
  ✅ vLLM吞吐提升 {(vllm_results['qps']/pytorch_results['qps'] - 1)*100:.0f}%
  ✅ 可以减少 {int(1 - pytorch_results['qps']/vllm_results['qps'])*100}% 的GPU！
""")
```

### ✅ 9.5 性能优化检查清单

```python
立即可做（低垂的果实）:
  □ 使用Spot/Preemptible实例（节省70%成本）
  □ 调整HPA的min/max副本数
  □ 添加请求缓存（Redis）
  □ 启用CDN（静态资源加速）
  
模型优化（中等难度）:
  □ INT8量化（我们已完成）
  □ INT4量化（更激进，需要GPTQ/AWQ）
  □ 动态量化（根据输入自适应）
  □ 模型剪枝（减少参数量）
  
推理优化（较难，收益大）:
  □ 集成vLLM（10-20x加速）
  □ 使用TensorRT（NVIDIA专用）
  □ Flash Attention（注意力加速）
  □ Continuous Batching（批处理优化）
  
成本优化:
  □ 监控GPU利用率（应>70%）
  □ 使用Reserved Instances（长期更便宜）
  □ 设置闲时自动缩容
  □ 定期审计资源使用

最终目标:
  ✅ P95延迟 < 200ms
  ✅ GPU利用率 > 70%
  ✅ 成本降低 > 50%
  ✅ QPS提升 > 5x
```

---

## 🎯 本章总结

恭喜你完成了从零到生产的完整旅程！让我们回顾核心内容。

### 💎 九步完整流程

```python
从零到生产，我们经历了：

第1步：数据准备 → 收集10万行代码，分词处理
第2步：模型训练 → 基于GPT-2训练基础模型
第3步：分布式加速 → 4卡DDP，训练时间缩短4x
第4步：模型优化 → INT8量化，模型体积减少4x
第5步：API服务 → FastAPI构建RESTful接口
第6步：容器化 → Docker打包，一键部署
第7步：K8s部署 → 3副本+自动扩缩容，生产级可用
第8步：监控运维 → Prometheus+Grafana，可观测
第9步：性能优化 → Spot+vLLM，成本降低90%

最终成果：
  ✅ P95延迟<200ms
  ✅ 吞吐量100+ QPS
  ✅ 月成本$200-500
  ✅ 可用性99.9%
```

### 🎨 部署方案速查表

| 你的情况 | 推荐方案 | 硬件 | 成本/月 | 时间 |
|---------|---------|-----|---------|------|
| 🎓 **学习实验** | 本地+Docker | 自己电脑 | $0 | 2-3天 |
| 💡 **个人项目** | 单机云服务器 | 1x T4 GPU | $10-20 | 3天 |
| 🚀 **创业MVP** | K8s托管 | 2-3x A10 | $200-500 | 1-2周 |
| 🏢 **企业级** | 自建K8s | 10+ GPU | $2000+ | 1-2月 |

### ✨ 三大核心能力

**1. 端到端部署**
```
你现在能够：
  ✅ 从数据收集到模型训练
  ✅ 从模型到API服务
  ✅ 从API到生产环境
  
完整掌握AI工程化流程
```

**2. 生产问题解决**
```
遇到问题知道怎么办：
  • 服务挂了？→ 查日志、看监控、快速回滚
  • 延迟太高？→ 量化、vLLM、优化推理
  • 成本太高？→ Spot实例、资源调度
  
5分钟内定位90%的问题
```

**3. 性能成本优化**
```
会权衡和优化：
  • 延迟 vs 成本：选对GPU类型
  • 质量 vs 速度：量化精度权衡
  • 稳定 vs 成本：Spot实例策略
  
节省50-90%成本的方法
```

---

## ✅ 知识自检

### 📝 必须掌握（基础理解）

回答这5个问题，确保你理解了核心概念：

1. **为什么要做模型量化？**
   <details>
   <summary>点击查看答案</summary>
   减少模型大小和显存占用。INT8量化可以将模型从FP32（4字节/参数）降到INT8（1字节/参数），体积减少4x，推理速度提升2x，但精度几乎不变。
   </details>

2. **K8s的Pod、Deployment、Service分别是什么？**
   <details>
   <summary>点击查看答案</summary>
   - Pod：最小部署单元，运行容器
   - Deployment：管理Pod副本数量和更新策略
   - Service：负载均衡，提供稳定的访问入口
   </details>

3. **如何判断部署是否成功？**
   <details>
   <summary>点击查看答案</summary>
   检查3点：1) 健康检查返回200  2) curl测试API正常响应  3) 日志无错误信息。K8s中用kubectl get pods看Pod状态是Running。
   </details>

4. **Spot实例为什么便宜？有什么风险？**
   <details>
   <summary>点击查看答案</summary>
   便宜70%是因为用的是云服务商的空闲资源。风险是随时可能被中断（2分钟通知）。适合训练任务（可断点续训），不适合实时服务。
   </details>

5. **Prometheus的四大黄金指标是什么？**
   <details>
   <summary>点击查看答案</summary>
   1) Latency（延迟）  2) Traffic（流量/吞吐）  3) Errors（错误率）  4) Saturation（饱和度/资源利用率）。监控这4个指标就能掌握系统健康度。
   </details>

### 🎯 建议掌握（深入理解）

如果能回答这些，说明你真正理解了原理：

- **为什么分布式训练4卡不是4倍加速？**
  - 提示：通信开销

- **vLLM如何实现10x吞吐提升？**
  - 提示：Continuous batching + PagedAttention

- **如何实现零停机更新？**
  - 提示：滚动更新 + 健康检查

### 🛠️ 实战验证（动手检验）

完成这3个任务，确保你能应用所学：

```bash
# 任务1: 30分钟快速部署（新手）
wget pretrained_model.bin
python serve_api.py
curl http://localhost:8000/complete

# 任务2: 4小时端到端（有基础）
# 数据→训练→量化→API→Docker

# 任务3: 1周完整项目（求职用）
# 构建代码补全服务，开源到GitHub
```

---

## 🐛 常见问题 FAQ

### Q1: 我是新手，应该选什么部署方式？

**快速答案**：从本地Docker开始，再考虑云服务器。

| 阶段 | 方案 | 成本 | 适合场景 |
|-----|------|------|---------|
| 🎓 **学习实验** | 本地+Docker | $0 | 跑通流程，理解概念 |
| 💡 **原型Demo** | 单机云服务器 | $10-20/月 | 给别人看，有公网IP |
| 🚀 **正式产品** | K8s托管 | $200-500/月 | 高可用，自动扩缩容 |
| 🏢 **企业级** | 自建集群 | $2000+/月 | 大规模，定制需求 |

**建议路径**：
```python
Week 1: 本地Docker（理解原理）
Week 2: 云服务器（学会上云）
Week 3-4: K8s托管（生产级部署）
```

---

### Q2: 我的API太慢，延迟多少算正常？

**快速答案**：T4 GPU上500-1000ms算正常，A100上100-200ms。

```python
延迟基准（生成50 tokens）：
  CPU:         5-10秒    ❌ 太慢，只能测试
  T4 GPU:      500-1000ms ✅ 可用
  A10 GPU:     200-400ms  ✅ 很好
  A100 GPU:    100-200ms  🚀 极快

目标（根据应用）：
  交互式聊天：<500ms
  代码补全：  <200ms
  批量处理：  <5秒
```

**如何优化？**
```bash
1. 确保用GPU（立即10x加速）
   model = model.to('cuda')

2. 模型量化（2x加速）
   python quantize_model.py

3. 集成vLLM（5-10x加速）
   pip install vllm
```

---

### Q3: 部署AI服务要花多少钱？

**快速答案**：$0-$500/月，看你的规模和优化程度。

```python
实际成本估算：

个人学习（$0-10/月）：
  • 本地运行：$0
  • 偶尔用云GPU：每天用2小时 = $0.35×2×30 ≈ $20/月

小团队MVP（$50-200/月）：
  • 1x T4 GPU 24小时
  • 用Spot实例（省70%）
  • 实际：$250/月 × 30% = $75/月

正式产品（$200-500/月）：
  • 2-3x A10 GPU
  • 量化+vLLM优化
  • K8s自动扩缩容

省钱技巧（省50-90%）：
  ✅ Spot实例 → 省70%
  ✅ 模型量化 → 省50%
  ✅ vLLM优化 → 省80%
  ✅ 按需使用 → 省67%
```

---

### Q4: 服务突然挂了，怎么快速定位？

**5步排查法**（5分钟定位90%问题）：

```bash
Step 1: 服务还活着吗？
  curl http://your-api/health
  
Step 2: 查看日志（最重要！）
  # Docker:
  docker logs <container-id>
  
  # K8s:
  kubectl logs <pod-name>
  
  常见错误：
    "CUDA out of memory" → 减小batch_size
    "ModuleNotFoundError" → pip install缺失的包
    "Port already in use" → 换端口或kill进程

Step 3: 检查资源
  nvidia-smi  # GPU使用情况
  htop        # CPU/内存
  
Step 4: 网络问题？
  ping your-api
  curl -v http://your-api/health
  
Step 5: 还没解决？重启试试
  docker restart <container-id>
  # 或
  kubectl delete pod <pod-name>  # K8s会自动重启
```

---

### Q5: 如何判断部署是否成功？

**3个检查点**：

```bash
✅ 1. 健康检查通过
curl http://localhost:8000/health
# 返回 {"status": "ok"}

✅ 2. API正常响应
curl -X POST http://localhost:8000/complete \
  -d '{"prompt":"def hello():","max_new_tokens":20}'
# 返回补全结果

✅ 3. 性能指标正常
• 延迟<1秒（T4 GPU）
• GPU利用率>70%
• 日志无错误
```

---

### Q6: 量化会损失多少精度？

**快速答案**：INT8量化几乎不损失（<2%），但体积减少4x。

```python
量化对比（GPT-2 124M）：

精度        模型大小   延迟    Perplexity  质量评价
FP32(原始)  400MB     100ms   15.2       ⭐⭐⭐⭐⭐
INT8        100MB     50ms    15.5 (+2%) ⭐⭐⭐⭐⭐
INT4        50MB      25ms    16.8 (+10%)⭐⭐⭐⭐

建议：
  • 生产环境：用INT8（最佳平衡）
  • 极致速度：用INT4（可接受轻微质量损失）
  • 研究实验：用FP32（无损精度）
```

---

### Q7: K8s和Docker有什么区别？

**一句话**：Docker管单个容器，K8s管一群容器。

```python
Docker（单机容器）：
  ✅ 打包应用+环境
  ✅ 本地开发、测试
  ❌ 不管副本、扩缩容、负载均衡
  
  适合：
    • 本地开发
    • 单机部署
    • 小项目

Kubernetes（容器编排）：
  ✅ 管理多个容器（跨多台机器）
  ✅ 自动扩缩容
  ✅ 负载均衡
  ✅ 滚动更新
  ✅ 自动重启
  
  适合：
    • 生产环境
    • 高可用服务
    • 需要扩展的产品

关系：K8s运行Docker容器
```

---

### Q8: 这章内容能帮我找工作吗？

**答案**：能！做一个端到端项目，写进简历。

**✅ 如何转化为面试优势：**

```python
1. 做一个Demo项目（最重要）
   项目：代码补全AI服务
   
   亮点：
     ✅ 端到端：数据→训练→部署
     ✅ 生产级：Docker+K8s
     ✅ 可观测：监控+日志
     ✅ 有数据：延迟、吞吐、成本
   
   开源到GitHub，面试时演示！

2. 简历怎么写？
   
   ❌ 错误："了解Docker和K8s"
   
   ✅ 正确："端到端部署GPT模型到生产环境：
     • Docker容器化，减少部署时间80%
     • K8s实现自动扩缩容，支持100+并发
     • INT8量化+vLLM，成本降低70%
     • Prometheus+Grafana监控
     [项目链接：github.com/你的用户名/项目]"

3. 面试常问问题
   Q: 如何部署AI模型？
   A: 我做过一个项目...（讲你的Demo）
   
   Q: 如何优化推理性能？
   A: 我用过量化、vLLM、KV Cache...
   
   Q: 生产环境遇到过什么问题？
   A: 遇到过OOM，通过量化解决...

目标岗位：
  • AI工程师（MLOps）
  • 后端工程师（AI方向）
  • DevOps工程师
  • 全栈工程师（AI产品）
```

---

## 📚 推荐资源

### 必读文档（精选3个）

```
1. FastAPI官方文档 ⭐⭐⭐⭐⭐
   https://fastapi.tiangolo.com/
   最好的Python API框架

2. Docker官方教程 ⭐⭐⭐⭐⭐
   https://docs.docker.com/get-started/
   容器化入门必读

3. Kubernetes基础 ⭐⭐⭐⭐
   https://kubernetes.io/docs/tutorials/
   K8s核心概念
```

### 重要文章（精选3篇）

```
1. Building LLM applications for production ⭐⭐⭐⭐⭐
   by Chip Huyen
   https://huyenchip.com/2023/04/11/llm-engineering.html
   LLM工程化完整指南

2. Patterns for Building LLM-based Systems ⭐⭐⭐⭐
   by Eugene Yan
   https://eugeneyan.com/writing/llm-patterns/
   7种常见架构模式

3. How to Deploy Large Language Models ⭐⭐⭐⭐
   by Hugging Face
   https://huggingface.co/blog/deploy-llms
   实战指南，代码示例丰富
```

### 实用工具

```bash
# 开发工具
pip install fastapi uvicorn[standard] torch tiktoken

# 测试工具
pip install locust pytest httpx

# 监控工具
pip install prometheus-client

# Docker（已安装就跳过）
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# kubectl
brew install kubectl  # macOS
# 或 https://kubernetes.io/docs/tasks/tools/
```

---

## 🚀 下一步行动

### 💡 立即实践（今天就可以做）

```bash
# 🎓 30分钟快速验证（完全新手）
wget https://huggingface.co/gpt2/pytorch_model.bin
python serve_api.py --model gpt2
curl http://localhost:8000/complete

# 🔬 4小时端到端（有基础）
python collect_code.py          # 准备数据
python train.py config/tiny.py  # 训练小模型
python quantize_model.py        # 量化优化
python serve_api.py             # 启动API
docker build -t my-service .    # Docker打包

# 🚀 1周完整项目（求职用）
# 构建代码补全服务，开源到GitHub
# 写README，画架构图，记录性能数据
```

### 📅 本周计划（巩固所学）

```python
Day 1-2: 复现本章所有示例代码
  ✅ 数据准备
  ✅ 模型训练
  ✅ API服务

Day 3-4: 尝试不同配置
  ✅ 改变模型大小
  ✅ 测试量化效果
  ✅ 对比不同GPU

Day 5-7: 端到端项目
  ✅ 自己找数据
  ✅ 训练自己的模型
  ✅ 部署上线
  ✅ 开源到GitHub

🎯 目标：1周内做出可演示的项目
```

### 🎯 继续学习

**按章节顺序（推荐）**：
```
✅ 第10章：生产部署（已完成）

📖 第11章：多模态模型
   学习图文模型、CLIP等
   
📖 第12章：专家混合模型（MoE）
   学习稀疏激活、Switch Transformer
   
📖 第13章：RLHF与对齐
   学习人类反馈强化学习
```

**按需求选择**：
```python
if 你想做应用层开发:
    ✅ 学完本章就够了！
    下一步: 直接做项目，边做边学

elif 你想深入工程化:
    继续学习:
      • 分布式系统设计
      • 微服务架构
      • 云原生技术栈

elif 你想做算法研究:
    继续学习第11-13章:
      • 多模态模型
      • MoE架构
      • RLHF对齐

elif 你想创业/做产品:
    重点补充:
      • 产品设计
      • 用户增长
      • 商业化策略
```

---

**恭喜你完成第10章！** 🎉

你现在已经掌握了从零到生产的完整AI部署技能。从数据准备到模型训练，从API服务到K8s部署，从监控运维到性能优化，你已经具备了部署生产级AI服务的能力。

**准备好了吗？让我们继续前进！** → [11_multimodal_models.md](11_multimodal_models.md)
