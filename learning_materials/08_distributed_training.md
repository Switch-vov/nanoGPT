# 第08章：分布式训练完全指南

> **学习目标**: 掌握使用多GPU/多机训练大规模模型的技术  
> **难度等级**: 🌿🌿🌿🌿 高级  
> **预计时间**: 5-6小时  
> **前置知识**: 01-05章基础知识

## 🎯 你将学到什么

学完本章，你将能够：
- ✅ 理解分布式训练的基本原理
- ✅ 掌握DDP（数据并行）的使用
- ✅ 理解FSDP（模型并行）的原理
- ✅ 掌握DeepSpeed ZeRO的配置
- ✅ 能够诊断和解决分布式训练问题
- ✅ 理解通信开销和优化策略

## 💭 开始之前：为什么要学这个？

**场景**：单GPU训练太慢，大模型装不下。

**比喻**：就像团队协作：
- 👤 一个人：慢，能力有限
- 👥 团队：分工合作，效率倍增
- 🏢 大公司：处理超大项目

**学完之后**：
- ✅ 能用多GPU加速训练
- ✅ 理解不同并行策略
- ✅ 会配置DeepSpeed
- ✅ 能训练更大的模型

---

## 🎯 核心问题

**单GPU的限制：**
- GPT-2 (124M) 在单个A100上训练需要 **2周**
- GPT-3 (175B) 单GPU根本**装不下**
- 大batch size需要**更多显存**

**解决方案：分布式训练**
- 8×GPU → 训练时间缩短到 **4天**
- 数百GPU → 训练超大模型
- 接近**线性加速**

---

## 📚 第一部分：分布式训练基础

### 🔍 什么是分布式训练？

```python
单GPU训练:
  GPU 0: 处理batch 0
  等待...
  GPU 0: 处理batch 1
  等待...
  总时间: N个batch × 单batch时间

分布式训练:
  GPU 0: 处理batch 0  ┐
  GPU 1: 处理batch 1  ├─ 同时进行！
  GPU 2: 处理batch 2  │
  GPU 3: 处理batch 3  ┘
  
  同步梯度（平均所有GPU）
  所有GPU更新相同的参数
  
  总时间: N个batch × 单batch时间 / GPU数量
```

### 📊 三种主要策略

```python
1️⃣ 数据并行 (Data Parallelism) - 最常用
   每个GPU: 完整模型 + 不同数据
   
   ┌─────────────┬─────────────┬─────────────┐
   │   GPU 0     │   GPU 1     │   GPU 2     │
   ├─────────────┼─────────────┼─────────────┤
   │ 模型副本    │ 模型副本    │ 模型副本    │
   │ batch 0     │ batch 1     │ batch 2     │
   └─────────────┴─────────────┴─────────────┘
   
   优势: 实现简单，最常用
   用途: 中小模型（<7B参数）

2️⃣ 模型并行 (Model Parallelism)
   每个GPU: 部分模型 + 相同数据
   
   ┌─────────────┬─────────────┬─────────────┐
   │   GPU 0     │   GPU 1     │   GPU 2     │
   ├─────────────┼─────────────┼─────────────┤
   │ Layer 0-3   │ Layer 4-7   │ Layer 8-11  │
   │ batch 0     │ batch 0     │ batch 0     │
   └─────────────┴─────────────┴─────────────┘
   
   优势: 可训练超大模型
   用途: 大模型（>30B参数）

3️⃣ 流水线并行 (Pipeline Parallelism)
   结合模型并行 + 批次流水
   
   时间线:
   t0: GPU0处理batch0
   t1: GPU0处理batch1, GPU1处理batch0
   t2: GPU0处理batch2, GPU1处理batch1, GPU2处理batch0
   
   优势: 提高GPU利用率
   用途: 配合模型并行
```

---

## ⚙️ 第二部分：PyTorch DDP详解

### 🚀 DDP (DistributedDataParallel)

**核心概念：**

```python
关键术语:

World Size: 总进程数（通常 = GPU数量）
  例: 8个GPU → world_size = 8

Rank: 进程ID（0到world_size-1）
  GPU 0 → rank 0
  GPU 1 → rank 1
  ...

Local Rank: 当前节点内的进程ID
  单机: local_rank = rank
  多机: 节点0的GPU1 → rank=1, local_rank=1
       节点1的GPU1 → rank=9, local_rank=1

Master进程: rank 0
  负责: 日志、保存模型、协调
```

### 🔧 NanoGPT中的DDP实现

**初始化代码分析：**

```python
# train.py 第94-105行

# 检测是否使用DDP
ddp = int(os.environ.get('RANK', -1)) != -1

if ddp:
    # 初始化进程组
    init_process_group(backend=backend)
    
    # 获取进程信息
    ddp_rank = int(os.environ['RANK'])              # 全局rank
    ddp_local_rank = int(os.environ['LOCAL_RANK'])  # 本地rank
    ddp_world_size = int(os.environ['WORLD_SIZE'])  # 总进程数
    
    # 设置设备
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    
    # 判断是否是master进程
    master_process = ddp_rank == 0
else:
    # 单GPU训练
    master_process = True
    ddp_world_size = 1
    ddp_rank = 0
```

**模型包装：**

```python
# train.py 第245-253行

model = GPT(gptconf)
model.to(device)

if ddp:
    # 包装成DDP模型
    model = DDP(model, device_ids=[ddp_local_rank])
```

**梯度同步：**

```python
# DDP的工作原理

# 前向传播（每个GPU独立）
for micro_step in range(gradient_accumulation_steps):
    X, Y = get_batch('train')  # 每个GPU获取不同的batch
    logits, loss = model(X, Y)  # 独立计算
    loss.backward()             # 独立反向传播

# DDP自动同步梯度
# 1. 收集所有GPU的梯度
# 2. 求平均
# 3. 广播回所有GPU

# 参数更新（所有GPU执行相同更新）
optimizer.step()
optimizer.zero_grad()
```

---

## 🚀 第三部分：实战：单机多卡训练

### 📝 步骤1：准备代码

NanoGPT已经支持DDP，无需修改！✅

### 📝 步骤2：启动训练

**单机8卡训练：**

```bash
# 使用torchrun（推荐）
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

# 参数说明:
# --standalone: 单机模式
# --nproc_per_node=8: 每个节点8个进程（8个GPU）

# torchrun会自动设置环境变量:
# RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
```

**单机4卡训练（如果只有4个GPU）：**

```bash
torchrun --standalone --nproc_per_node=4 train.py config/train_shakespeare_char.py
```

### 📝 步骤3：查看输出

```bash
# 输出示例（rank 0的日志）
Initializing process group...
{
    'master_process': True,
    'seed_offset': 0,
    'ddp_world_size': 8,
    'tokens_per_iter': 524288,  # 增加了8倍！
}

step 0: train loss 4.2434, val loss 4.2438
step 10: train loss 3.8765, val loss 3.8821
...

# 其他rank（1-7）的输出被抑制
# 只有master进程（rank 0）打印日志
```

### 📝 步骤4：理解加速比

```python
理论加速:

单GPU:
  batch_size = 12
  时间/iter = 500ms
  tokens/sec = (12 × 1024) / 0.5 = 24,576

8×GPU:
  batch_size = 12 × 8 = 96
  时间/iter = 550ms  # 略增加（通信开销）
  tokens/sec = (96 × 1024) / 0.55 = 178,913
  
加速比 = 178,913 / 24,576 = 7.3x

实际加速比:
  2 GPU: 1.9x
  4 GPU: 3.7x
  8 GPU: 7.3x
  
效率 = 实际加速比 / GPU数量
  8 GPU: 7.3 / 8 = 91% ✅ 很好！
```

---

## 🌐 第四部分：多机多卡训练

### 🔧 网络配置

**前提条件：**
```bash
# 1. 所有节点可以互相访问
ping node1
ping node2

# 2. 相同的代码和数据
rsync -av /workspace/ node2:/workspace/

# 3. 测试网络带宽
iperf3 -s  # 在node1运行服务器
iperf3 -c node1  # 在node2测试
# 期望: >10 Gbps (如果有Infiniband: >100 Gbps)
```

### 🚀 启动多机训练

**节点0（Master）：**

```bash
# 在节点0（IP: 192.168.1.100）
torchrun \
  --nproc_per_node=8 \    # 每个节点8个GPU
  --nnodes=2 \            # 总共2个节点
  --node_rank=0 \         # 当前节点rank=0 (master)
  --master_addr=192.168.1.100 \  # master地址
  --master_port=29500 \   # 通信端口
  train.py config/train_gpt2.py
```

**节点1（Worker）：**

```bash
# 在节点1（IP: 192.168.1.101）
torchrun \
  --nproc_per_node=8 \
  --nnodes=2 \
  --node_rank=1 \         # 当前节点rank=1 (worker)
  --master_addr=192.168.1.100 \  # master地址
  --master_port=29500 \
  train.py config/train_gpt2.py
```

**进程分布：**

```
节点0 (192.168.1.100):
  GPU 0 → rank 0 (master进程)
  GPU 1 → rank 1
  ...
  GPU 7 → rank 7

节点1 (192.168.1.101):
  GPU 0 → rank 8
  GPU 1 → rank 9
  ...
  GPU 7 → rank 15

总共: 16个进程，world_size=16
```

### ⚠️ 常见问题

```python
问题1: "Address already in use"
解决: 更换端口
  --master_port=29501

问题2: 连接超时
检查:
  1. 防火墙是否开放端口
  2. IP地址是否正确
  3. 网络是否连通

问题3: "NCCL timeout"
解决:
  export NCCL_TIMEOUT=3600  # 增加超时时间
  export NCCL_DEBUG=INFO    # 开启调试信息

问题4: 没有Infiniband
添加:
  export NCCL_IB_DISABLE=1  # 禁用Infiniband
  (会变慢，但能工作)
```

---

## 📊 第五部分：性能优化

### ⚡ 1. 梯度累积 + DDP

```python
# NanoGPT的实现（train.py 第292-305行）

gradient_accumulation_steps = 5  # 累积5次
batch_size = 12  # 每个GPU的batch

# 有效batch size
effective_batch = batch_size × gradient_accumulation_steps × world_size
                = 12 × 5 × 8
                = 480

# 这样可以模拟更大的batch，但显存占用少
```

**为什么这样做？**

```
场景: 想要batch_size=480，但显存只够batch_size=12

方法1: 直接用480 ❌
  显存: 爆炸

方法2: 8×GPU，每个用60 ❌
  显存: 还是不够

方法3: 8×GPU，每个12，梯度累积5次 ✅
  显存: OK
  效果: 等同于batch_size=480
```

### ⚡ 2. Zero Redundancy Optimizer (ZeRO)

**标准DDP的问题：**

```python
模型参数: 7B × 4 bytes = 28GB

在8个GPU上:
  每个GPU存储: 
  - 模型参数: 28GB
  - 梯度: 28GB  
  - 优化器状态: 56GB (AdamW有2个buffer)
  总计: 112GB ❌ 单GPU装不下！

问题: 每个GPU都存储完整的副本（冗余）
```

**ZeRO的解决方案：**

```python
ZeRO-1: 分片优化器状态
  每个GPU: 模型(28GB) + 梯度(28GB) + 优化器状态(7GB)
  总计: 63GB ✅

ZeRO-2: 分片优化器状态 + 梯度
  每个GPU: 模型(28GB) + 梯度(3.5GB) + 优化器状态(7GB)
  总计: 38.5GB ✅

ZeRO-3: 分片所有参数
  每个GPU: 模型(3.5GB) + 梯度(3.5GB) + 优化器状态(7GB)
  总计: 14GB ✅✅

可以训练更大的模型！
```

**使用ZeRO（需要DeepSpeed）：**

```bash
# 安装DeepSpeed
pip install deepspeed

# 创建配置文件 ds_config.json
{
  "train_batch_size": 96,
  "gradient_accumulation_steps": 1,
  "zero_optimization": {
    "stage": 2,  # ZeRO-2
    "offload_optimizer": {
      "device": "cpu"  # 可选：将优化器状态卸载到CPU
    }
  }
}

# 启动训练
deepspeed --num_gpus=8 train.py config/train_gpt2.py \
  --deepspeed --deepspeed_config=ds_config.json
```

### ⚡ 3. Gradient Checkpointing

**问题：激活值占用大量内存**

```python
前向传播需要保存中间结果用于反向传播:

Layer 1 → activation₁ (保存)
Layer 2 → activation₂ (保存)
...
Layer 12 → activation₁₂ (保存)

对于大模型: 激活值 > 参数量
  GPT-3: 激活值可达200GB+
```

**Gradient Checkpointing：**

```python
策略: 不保存所有激活值，只保存部分

前向传播:
  保存: Layer 0, 4, 8, 12 的激活值
  丢弃: Layer 1, 2, 3, 5, 6, 7...

反向传播:
  需要Layer 7的激活值？
  → 从Layer 4重新前向计算到Layer 7
  
权衡:
  内存: 减少75%
  时间: 增加20-30%
```

**实现：**

```python
# 在model.py中修改

import torch.utils.checkpoint as checkpoint

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_checkpoint = config.gradient_checkpointing
        # ...
    
    def forward(self, idx, targets=None):
        # ...
        for block in self.transformer.h:
            if self.use_checkpoint and self.training:
                # 使用checkpointing
                x = checkpoint.checkpoint(block, x)
            else:
                x = block(x)
        # ...
```

---

## 🔬 第六部分：监控和调试

### 📊 监控GPU利用率

```bash
# 实时监控
watch -n 1 nvidia-smi

# 输出示例（8 GPU训练）
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx       Driver Version: 525.xx       CUDA Version: 12.0   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  A100-SXM...  On   | 00000000:00:04.0 Off |                    0 |
| N/A   45C    P0   280W / 400W |  38421MiB / 40960MiB |    100%      Default |
|-------------------------------+----------------------+----------------------+
|   1  A100-SXM...  On   | 00000000:00:05.0 Off |                    0 |
| N/A   46C    P0   285W / 400W |  38421MiB / 40960MiB |    100%      Default |
|-------------------------------+----------------------+----------------------+
...

期望:
  GPU-Util: 95-100% (充分利用)
  Memory-Usage: 80-95% (不要太满，避免OOM)
  Power: 接近上限 (说明在认真工作)
```

### 🐛 调试技巧

```python
# 1. 设置环境变量查看详细信息
export NCCL_DEBUG=INFO      # NCCL通信调试
export TORCH_DISTRIBUTED_DEBUG=INFO  # PyTorch DDP调试

# 2. 检查梯度同步
# 在train.py中添加：
if iter_num % 100 == 0:
    # 检查所有GPU的参数是否一致
    if ddp:
        for name, param in model.named_parameters():
            # 收集所有GPU的参数
            tensor_list = [torch.zeros_like(param) for _ in range(ddp_world_size)]
            dist.all_gather(tensor_list, param.data)
            
            # 检查是否一致
            for i, t in enumerate(tensor_list[1:], 1):
                if not torch.allclose(tensor_list[0], t):
                    print(f"WARNING: rank 0 and rank {i} param {name} differ!")

# 3. 测试通信带宽
python -m torch.distributed.run \
  --nproc_per_node=8 \
  -m torch.distributed.launch \
  --test_distributed_communication
```

---

## 💡 第七部分：最佳实践

### ✅ Do's (推荐做法)

```python
1. 使用torchrun启动
   ✅ torchrun --nproc_per_node=8 train.py
   ❌ python -m torch.distributed.launch (旧方法)

2. 合理设置batch size
   每个GPU的batch_size = 能装下的最大值
   总batch_size = per_gpu_batch × world_size × grad_accum

3. 只在master进程做IO
   if master_process:
       print(...)
       save_checkpoint(...)
       log_to_wandb(...)

4. 同步随机数种子（但加上offset）
   torch.manual_seed(seed + ddp_rank)

5. 使用pin_memory加速数据传输
   X, Y = X.pin_memory().to(device, non_blocking=True)

6. 定期保存checkpoint
   每N步保存一次，防止训练中断
```

### ❌ Don'ts (避免做法)

```python
1. 不要在非master进程打印
   ❌ print(f"rank {rank}: loss = {loss}")
   ✅ if master_process: print(f"loss = {loss}")

2. 不要假设所有GPU同步
   ❌ 直接访问model.module.某参数
   ✅ 使用DDP包装后的接口

3. 不要使用全局变量
   ❌ global_counter += 1  # 每个进程独立，不同步
   ✅ 在master进程维护，或用分布式存储

4. 不要忘记设置随机种子
   否则每个GPU生成相同的数据增强

5. 不要在训练循环中频繁同步
   ❌ dist.barrier() 在每个iter
   ✅ 只在必要时同步（如evaluation）
```

### 🎯 性能调优清单

```python
□ 使用mixed precision (bfloat16/float16)
□ 启用torch.compile()
□ 使用flash attention
□ 合理的batch size (填满显存的80-90%)
□ 梯度累积（模拟更大batch）
□ pin_memory + non_blocking传输
□ 数据预加载（避免CPU成为瓶颈）
□ 使用fast tokenizer
□ 减少Python开销（减少print、logging）
□ 使用profiler找瓶颈
```

---

## 📊 第八部分：实战案例

### 🎯 案例1：训练GPT-2 (124M)

```bash
# 配置
硬件: 8×A100 40GB
模型: GPT-2 (124M参数)
数据: OpenWebText (9B tokens)
目标: 4天完成训练

# 启动命令
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

# config/train_gpt2.py
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 40  # 5×8 (已包含8 GPU)

# 有效batch size
effective_batch = 12 × 40 × 8 = 3,840 tokens per update

# 预期结果
训练时间: ~4天
最终loss: ~2.85
tokens/sec: ~200K
GPU利用率: ~95%
```

### 🎯 案例2：训练小模型快速实验

```bash
# 配置
硬件: 4×RTX 3090
模型: 10M参数
数据: Shakespeare (1M tokens)
目标: 30分钟内完成

# 启动命令
torchrun --standalone --nproc_per_node=4 \
  train.py config/train_shakespeare_char.py \
  --n_layer=4 --n_embd=256 \
  --max_iters=5000

# 预期结果
训练时间: ~25分钟
最终loss: ~1.4
加速比: 3.7x (相比单GPU)
```

---

## 🔧 第九部分：故障排查

### 🐛 常见错误及解决

```python
错误1: RuntimeError: NCCL error in: ...

原因: 
  - 通信问题
  - 网络不稳定
  - 某个GPU崩溃

解决:
  1. 检查所有GPU: nvidia-smi
  2. 测试GPU: nvidia-smi -L
  3. 重启训练
  4. 增加超时: export NCCL_TIMEOUT=7200

错误2: RuntimeError: Address already in use

原因: 端口被占用

解决:
  1. 更换端口: --master_port=29501
  2. 杀死旧进程: pkill -f train.py
  3. 查看端口: lsof -i :29500

错误3: CUDA out of memory

原因: 显存不够

解决:
  1. 减小batch_size
  2. 减小block_size
  3. 增加gradient_accumulation_steps
  4. 启用gradient checkpointing
  5. 使用ZeRO

错误4: Loss diverges (变成NaN)

原因: 
  - 学习率太大
  - 梯度爆炸
  - 数据有问题

解决:
  1. 减小learning_rate
  2. 启用gradient clipping
  3. 检查数据质量
  4. 增加warmup步数

错误5: 速度很慢，GPU利用率低

原因:
  - 数据加载慢
  - batch太小
  - Python开销大

解决:
  1. 增大batch_size
  2. 使用更多dataloader workers
  3. 预处理数据
  4. 减少print/logging
  5. 启用torch.compile()
```

---

## 📈 第十部分：高级主题

### 🚀 1. FSDP (Fully Sharded Data Parallel)

PyTorch内置的ZeRO-3替代品：

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# 包装模型
model = FSDP(
    model,
    sharding_strategy="FULL_SHARD",  # 等同于ZeRO-3
    mixed_precision=...,
    device_id=torch.cuda.current_device(),
)

# 优势
- PyTorch原生支持
- 不需要DeepSpeed
- 与torch.compile兼容

# 适用场景
- 大模型（>7B参数）
- 显存不够的情况
```

### 🌐 2. 3D并行

组合三种并行策略：

```python
场景: 训练100B参数模型，64个GPU

策略:
  数据并行: 8份
  流水线并行: 4层
  张量并行: 2份
  
  总GPU数 = 8 × 4 × 2 = 64 ✅

每个GPU:
  模型: 100B / (4×2) = 12.5B 参数
  batch: 总batch / 8
  
可行！
```

### 📊 3. Activation Checkpointing的高级用法

```python
# 选择性checkpointing
# 只对大的层使用

class GPT(nn.Module):
    def forward(self, x):
        for i, block in enumerate(self.transformer.h):
            # 只对偶数层使用checkpointing
            if i % 2 == 0 and self.training:
                x = checkpoint.checkpoint(block, x)
            else:
                x = block(x)
        return x
```

---

## 🎓 总结

### ✨ 核心要点

```python
1. DDP = 最常用的分布式策略
   - 每个GPU完整模型
   - 自动梯度同步
   - 近线性加速

2. 启动命令
   torchrun --standalone --nproc_per_node=N train.py

3. 关键概念
   - world_size: 总进程数
   - rank: 进程ID
   - master进程: 负责日志和保存

4. 优化技巧
   - 梯度累积
   - Mixed precision
   - Gradient checkpointing
   - ZeRO/FSDP (大模型)

5. 最佳实践
   - 只在master进程做IO
   - 使用pin_memory
   - 定期保存checkpoint
   - 监控GPU利用率
```

### 🎯 实用建议

```python
你的情况 → 推荐方案

1-2个GPU:
  → 标准DDP即可

4-8个GPU:
  → DDP + 梯度累积

大模型（显存不够）:
  → FSDP/ZeRO-2/3

多机训练:
  → 检查网络带宽
  → 使用Infiniband（如果有）
  → NCCL_IB_DISABLE=1（如果没有）

超大模型（>30B）:
  → 3D并行
  → 使用Megatron-LM或DeepSpeed
```

### 🚀 下一步

```python
立即可做:

1. 测试DDP
   torchrun --standalone --nproc_per_node=2 train.py ...
   
2. 监控性能
   nvidia-smi
   计算加速比
   
3. 优化配置
   调整batch_size
   启用mixed precision
   
进阶:
   
4. 多机训练
   准备多个节点
   测试网络
   
5. 大模型训练
   尝试FSDP
   使用DeepSpeed
```

---

**记住：**

> 分布式训练不是简单的"加更多GPU"，
> 而是理解通信模式、优化瓶颈、
> 权衡计算和通信开销的艺术。
>
> 从单GPU到8GPU，你能获得7x加速。
> 但从8GPU到64GPU，需要更多的工程技巧。

---

## 🎓 总结与检查

### ✅ 知识检查清单

完成学习后，你应该能够：

**基础概念（必须掌握）**
- [ ] 理解什么是数据并行（DDP）
- [ ] 知道AllReduce的作用
- [ ] 理解为什么需要同步梯度
- [ ] 能够启动多GPU训练
- [ ] 知道如何查看GPU使用情况
- [ ] 理解加速比的计算方法

**进阶理解（建议掌握）**
- [ ] 理解FSDP的分片策略
- [ ] 知道DeepSpeed ZeRO的三个阶段
- [ ] 理解通信开销的来源
- [ ] 能够诊断分布式训练问题
- [ ] 知道如何优化通信效率
- [ ] 理解混合精度训练的原理

**实战能力（最终目标）**
- [ ] 能够配置多GPU训练
- [ ] 会选择合适的并行策略
- [ ] 能够优化训练性能
- [ ] 会解决常见的分布式问题
- [ ] 能够进行多机训练
- [ ] 理解如何训练超大模型

### 📊 并行策略速查表

| 策略 | 适用场景 | 显存占用 | 通信开销 | 实现难度 | 推荐指数 |
|------|---------|---------|---------|---------|---------|
| **DDP** | 模型能放进单GPU | 每GPU都有完整模型 | 中等 | ⭐ 简单 | ⭐⭐⭐⭐⭐ |
| **FSDP** | 模型太大放不下 | 分片，显存节省 | 较高 | ⭐⭐ 中等 | ⭐⭐⭐⭐ |
| **DeepSpeed ZeRO-1** | 优化器状态大 | 节省4x | 低 | ⭐⭐ 中等 | ⭐⭐⭐⭐ |
| **DeepSpeed ZeRO-2** | 梯度也很大 | 节省8x | 中等 | ⭐⭐ 中等 | ⭐⭐⭐⭐⭐ |
| **DeepSpeed ZeRO-3** | 超大模型 | 节省N倍 | 高 | ⭐⭐⭐ 复杂 | ⭐⭐⭐ |

### 🎯 如何选择并行策略？

```python
# 决策树
if 模型 < 1B参数 and 能放进单GPU:
    使用 DDP  # 最简单，性能最好
    
elif 模型 < 10B参数:
    使用 FSDP 或 DeepSpeed ZeRO-2  # 平衡性能和显存
    
elif 模型 > 10B参数:
    使用 DeepSpeed ZeRO-3  # 必须分片
    
# 具体例子
GPT-2 (124M):   DDP ✅
GPT-2 (1.5B):   FSDP 或 ZeRO-2 ✅
GPT-3 (175B):   ZeRO-3 ✅
```

### 🚀 下一步学习

现在你已经掌握了分布式训练，接下来应该学习：

1. **09_model_optimization.md** - 学习模型量化和推理优化
2. **10_production_deployment.md** - 学习如何部署到生产环境
3. **实践项目** - 用多GPU训练一个实际模型

### 💡 实践建议

**立即可做**：
```bash
# 1. 测试2个GPU的DDP
torchrun --standalone --nproc_per_node=2 train.py

# 2. 监控GPU使用
watch -n 1 nvidia-smi

# 3. 计算加速比
# 单GPU时间 / 多GPU时间 = 加速比
# 理想：2GPU应该接近2x
```

**系统实验**：
```bash
# 实验1：测试不同GPU数量
for ngpu in 1 2 4 8; do
    echo "Testing $ngpu GPUs"
    torchrun --nproc_per_node=$ngpu train.py --max_iters=100
done

# 实验2：对比DDP和FSDP
python train_ddp.py    # DDP
python train_fsdp.py   # FSDP
# 对比：速度、显存、最终loss

# 实验3：优化通信
# 调整gradient_accumulation_steps
# 测试混合精度训练
```

**进阶研究**：
1. 阅读PyTorch DDP文档
2. 研究DeepSpeed的实现
3. 尝试多机训练
4. 优化通信瓶颈

---

## 📚 推荐资源

### 📖 必读文档
- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)
- [DeepSpeed官方文档](https://www.deepspeed.ai/getting-started/)

### 📄 重要论文
- **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models** (Rajbhandari et al., 2020)
  - https://arxiv.org/abs/1910.02054
  - DeepSpeed ZeRO的原始论文

- **PyTorch Distributed: Experiences on Accelerating Data Parallel Training** (Li et al., 2020)
  - https://arxiv.org/abs/2006.15704
  - PyTorch DDP的设计

### 🎥 视频教程
- [Andrej Karpathy: Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- [PyTorch DDP Tutorial](https://www.youtube.com/watch?v=Cvdhwx-OBBo)

### 🔧 实用工具
```bash
# 监控工具
nvidia-smi          # GPU使用情况
nvtop               # 更友好的GPU监控
htop                # CPU和内存监控

# 性能分析
torch.profiler      # PyTorch性能分析
nsys                # NVIDIA Nsight Systems
```

---

## 🐛 常见问题 FAQ

### Q1: 为什么2个GPU没有2倍加速？
**A**: 因为有通信开销。
```
理想情况：
  1 GPU: 100秒
  2 GPU: 50秒（2x加速）

实际情况：
  1 GPU: 100秒
  2 GPU: 60秒（1.67x加速）
  
原因：
  - 梯度同步需要时间（AllReduce）
  - 数据加载可能成为瓶颈
  - 小模型通信占比更大

如何优化：
  ✅ 增大batch_size（减少通信频率）
  ✅ 使用gradient_accumulation
  ✅ 优化数据加载（num_workers）
  ✅ 使用混合精度（减少通信量）
```

### Q2: DDP和DataParallel有什么区别？
**A**: DDP更快更好，DataParallel已过时。
```
DataParallel（旧，不推荐）:
  ❌ 单进程多线程（Python GIL限制）
  ❌ 主GPU负载不均衡
  ❌ 速度慢
  ❌ 不支持多机

DDP（新，推荐）:
  ✅ 多进程（无GIL限制）
  ✅ 负载均衡
  ✅ 速度快（2-3x）
  ✅ 支持多机

结论：永远使用DDP！
```

### Q3: 如何解决"NCCL timeout"错误？
**A**: 这是通信超时，常见原因和解决方法：
```bash
# 原因1：网络慢
export NCCL_TIMEOUT=1800  # 增加超时时间（秒）

# 原因2：GPU之间通信问题
export NCCL_DEBUG=INFO    # 查看详细日志
export NCCL_IB_DISABLE=1  # 禁用InfiniBand（如果有问题）

# 原因3：代码有bug导致hang
# 检查：
# - 是否所有进程都执行相同的操作？
# - 是否有条件分支导致某些进程卡住？
# - 是否正确使用barrier？

# 调试技巧
NCCL_DEBUG=INFO torchrun --nproc_per_node=2 train.py
# 查看哪个操作卡住了
```

### Q4: 显存不够怎么办？
**A**: 多种解决方案：
```python
# 方案1：使用FSDP（推荐）
# 自动分片模型，节省显存

# 方案2：使用DeepSpeed ZeRO
# ZeRO-2: 节省8x显存
# ZeRO-3: 节省Nx显存（N=GPU数量）

# 方案3：减小batch_size
batch_size = 8  # 从32减到8

# 方案4：使用gradient_accumulation
gradient_accumulation_steps = 4
# 效果batch_size = 8 * 4 = 32

# 方案5：使用gradient checkpointing
# 用计算换显存
model.gradient_checkpointing_enable()

# 方案6：混合精度训练
# FP16比FP32节省50%显存
```

### Q5: 如何验证分布式训练正确？
**A**: 检查这些指标：
```python
# 1. 加速比
单GPU时间 = 100秒
2GPU时间 = 55秒
加速比 = 100/55 = 1.82x
# 应该接近2x（考虑通信开销）

# 2. 最终loss
单GPU loss = 2.50
2GPU loss = 2.48
# 应该非常接近（误差<1%）

# 3. GPU利用率
nvidia-smi
# 所有GPU应该接近100%

# 4. 梯度一致性
# 打印梯度范数，应该相同
for p in model.parameters():
    print(p.grad.norm())
```

### Q6: 多机训练如何配置？
**A**: 需要配置节点通信：
```bash
# 节点1（主节点）
torchrun \
  --nnodes=2 \           # 总共2个节点
  --nproc_per_node=8 \   # 每个节点8个GPU
  --node_rank=0 \        # 主节点rank=0
  --master_addr="192.168.1.1" \  # 主节点IP
  --master_port=29500 \  # 通信端口
  train.py

# 节点2（工作节点）
torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --node_rank=1 \        # 工作节点rank=1
  --master_addr="192.168.1.1" \  # 主节点IP
  --master_port=29500 \
  train.py

# 注意事项：
# 1. 所有节点能互相ping通
# 2. 防火墙开放端口
# 3. 代码和数据在所有节点都有
# 4. 环境完全一致
```

### Q7: 如何选择GPU数量？
**A**: 根据模型大小和预算：
```
小模型（<500M参数）:
  1-2 GPU: 够用
  4 GPU: 如果追求速度
  8+ GPU: 浪费（通信开销大）

中模型（500M-5B）:
  2-4 GPU: 最佳平衡
  8 GPU: 大规模训练
  
大模型（>5B）:
  8+ GPU: 必须
  16-64 GPU: 工业级
  
实际建议：
  - 开发调试：1 GPU
  - 小规模训练：2-4 GPU
  - 生产训练：8 GPU
  - 超大模型：16+ GPU
```

### Q8: gradient_accumulation在分布式训练中如何工作？
**A**: 每个GPU独立累积，最后同步：
```python
# 例子：4个GPU，gradient_accumulation_steps=2

# Step 1: 前向+反向（不更新）
GPU0: batch_0 → grad_0（累积）
GPU1: batch_1 → grad_1（累积）
GPU2: batch_2 → grad_2（累积）
GPU3: batch_3 → grad_3（累积）

# Step 2: 前向+反向（不更新）
GPU0: batch_4 → grad_0 += grad_4
GPU1: batch_5 → grad_1 += grad_5
GPU2: batch_6 → grad_2 += grad_6
GPU3: batch_7 → grad_3 += grad_7

# Step 3: 同步梯度+更新
AllReduce(grad_0, grad_1, grad_2, grad_3)
optimizer.step()  # 所有GPU同时更新

# 效果：
# 相当于batch_size = 4 GPU × 2 steps × per_gpu_batch
# 但显存只需要per_gpu_batch
```

### Q9: 如何调试分布式训练代码？
**A**: 使用这些技巧：
```python
# 技巧1：先在单GPU测试
python train.py  # 确保代码没问题

# 技巧2：使用2个GPU测试
torchrun --nproc_per_node=2 train.py
# 比8个GPU更容易调试

# 技巧3：添加日志
import torch.distributed as dist
if dist.get_rank() == 0:
    print("Only rank 0 prints this")

# 技巧4：检查同步
dist.barrier()  # 确保所有进程到达这里
print(f"Rank {dist.get_rank()} passed barrier")

# 技巧5：使用环境变量
export NCCL_DEBUG=INFO      # 查看通信细节
export TORCH_DISTRIBUTED_DEBUG=INFO  # 查看分布式细节

# 技巧6：捕获异常
try:
    train()
except Exception as e:
    print(f"Rank {dist.get_rank()} error: {e}")
    dist.destroy_process_group()
```

### Q10: 混合精度训练在分布式中如何使用？
**A**: 与单GPU类似，但要注意梯度缩放：
```python
from torch.cuda.amp import autocast, GradScaler

# 初始化
scaler = GradScaler()

# 训练循环
for batch in dataloader:
    optimizer.zero_grad()
    
    # 混合精度前向
    with autocast():
        loss = model(batch)
        loss = loss / gradient_accumulation_steps
    
    # 缩放反向传播
    scaler.scale(loss).backward()
    
    if (step + 1) % gradient_accumulation_steps == 0:
        # 梯度裁剪（需要unscale）
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # 更新参数
        scaler.step(optimizer)
        scaler.update()

# 好处：
# - 速度快2x
# - 显存节省50%
# - 精度几乎无损
```

---

**恭喜你完成第08章！** 🎉

你现在已经掌握了分布式训练的核心技术。从单GPU到多GPU，从DDP到DeepSpeed，你已经具备了训练大规模模型的能力。

**准备好了吗？让我们继续前进！** → [09_model_optimization.md](09_model_optimization.md)
