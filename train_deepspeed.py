"""
使用 DeepSpeed 训练 GPT-2 XL 的脚本
支持 ZeRO-3 模型分片和 CPU offloading
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import deepspeed
from deepspeed import comm as dist

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# 配置来自 finetune_shakespeare.py
out_dir = 'out-shakespeare-xl-deepspeed'
eval_interval = 5  # 每5步评估一次
eval_iters = 20  # 评估20个batch（平衡速度和准确性）
wandb_log = False
wandb_project = 'shakespeare'
wandb_run_name = 'ft-xl-ds-' + str(time.time())

dataset = 'shakespeare'
init_from = 'gpt2-xl'  # 1.5B 参数模型

always_save_checkpoint = False  # 只保存最佳模型，不是每次都保存

# DeepSpeed 会自动处理这些
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 50

learning_rate = 3e-5
decay_lr = False
block_size = 1024

# 系统
device = 'cuda'
dtype = 'float16'
compile = False  # DeepSpeed 不兼容 PyTorch compile

# DeepSpeed 配置文件 (使用 ZeRO-2，更稳定)
deepspeed_config = 'ds_config_zero2.json'

# -----------------------------------------------------------------------------
# 解析命令行参数
import sys
# 过滤掉 DeepSpeed 添加的参数
sys.argv = [arg for arg in sys.argv if not arg.startswith('--local_rank')]

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) if os.path.exists('configurator.py') else None
config = {k: globals()[k] for k in config_keys}

# -----------------------------------------------------------------------------
# DeepSpeed 初始化
local_rank = int(os.environ.get('LOCAL_RANK', -1))
world_size = int(os.environ.get('WORLD_SIZE', 1))
rank = int(os.environ.get('RANK', -1))

# 初始化分布式环境（必须在 DeepSpeed 之前）
if local_rank != -1:
    torch.cuda.set_device(local_rank)
    device = f'cuda:{local_rank}'
    master_process = (rank == 0)
    
    # 初始化进程组（DeepSpeed会处理，不需要手动init）
    # 设置device后DeepSpeed会自动正确初始化
else:
    master_process = True
    rank = 0
    local_rank = 0
    world_size = 1
    device = 'cuda:0'

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + rank)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# 数据加载
data_dir = os.path.join('data', dataset)

def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# -----------------------------------------------------------------------------
# 模型初始化
iter_num = 0
best_val_loss = 1e9

meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    if master_process:
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# 初始化 model_args 字典
model_args = dict(n_layer=None, n_head=None, n_embd=None, block_size=block_size,
                  bias=False, vocab_size=meta_vocab_size, dropout=0.0)

if init_from == 'scratch':
    if master_process:
        print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args.update(dict(n_layer=12, n_head=12, n_embd=768))
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from.startswith('gpt2'):
    if master_process:
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=0.0)
    model = GPT.from_pretrained(init_from, override_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

if master_process:
    print(f"number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

model.to(device)

# -----------------------------------------------------------------------------
# 创建优化器（PyTorch原生AdamW，避免DeepSpeed的fused版本编译问题）
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.1
)

# -----------------------------------------------------------------------------
# DeepSpeed 引擎初始化（传入optimizer）
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    optimizer=optimizer,  # 传入我们创建的优化器
    config=deepspeed_config,
)

# -----------------------------------------------------------------------------
# 评估函数（支持分布式 - 完整修复版）
@torch.no_grad()
def estimate_loss():
    """
    在所有进程上执行评估，并通过 all_reduce 同步结果
    
    关键点：
    1. 所有进程都执行此函数
    2. 设置相同的随机种子确保所有进程评估相同的数据
    3. 使用 all_reduce 同步结果
    4. 只在 master 进程打印
    """
    out = {}
    model_engine.eval()
    
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        
        # 为评估设置固定的种子，确保所有进程采样相同的数据
        eval_seed = 1337 + rank  # 每个rank使用不同的种子
        torch.manual_seed(eval_seed)
        np.random.seed(eval_seed)
        
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model_engine(X, Y)
            losses[k] = loss
        
        # 恢复训练时的随机种子
        torch.manual_seed(1337 + rank)
        np.random.seed(1337 + rank)
        
        # 计算本地平均值（单个GPU上的平均）
        local_mean = losses.mean()
        
        # 多GPU环境：通过all_reduce同步各GPU的loss
        # AVG操作会自动计算所有GPU的平均值
        if world_size > 1:
            torch.distributed.all_reduce(
                local_mean, 
                op=torch.distributed.ReduceOp.AVG
            )
        
        out[split] = local_mean.item()
    
    model_engine.train()
    return out

# -----------------------------------------------------------------------------
# 训练循环
if master_process:
    print(f"\n{'='*60}")
    print(f"开始训练 GPT-2 XL (1.5B 参数) with DeepSpeed ZeRO-3")
    print(f"使用 {world_size} 个 GPU")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Max iterations: {max_iters}")
    print(f"{'='*60}\n")

X, Y = get_batch('train')
t0 = time.time()

for iter_num in range(max_iters):
    # 评估（所有进程都需要参与以保持同步）
    if iter_num % eval_interval == 0:
        losses = estimate_loss()  # 所有进程都执行评估，DeepSpeed会自动同步
        if master_process:
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # 检查是否需要保存检查点（所有进程都会进行此判断）
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                # DeepSpeed 保存检查点（所有进程都需要调用）
                model_engine.save_checkpoint(out_dir, tag=f"iter_{iter_num}")
                if master_process:
                    # 只有主进程打印保存信息
                    print(f"saving checkpoint to {out_dir}")
    
    # 前向和反向传播（DeepSpeed集成梯度累积）
    X, Y = get_batch('train')
    
    # DeepSpeed会自动处理梯度累积和缩放
    logits, loss = model_engine(X, Y)
    
    # 反向传播 - DeepSpeed的正确API
    model_engine.backward(loss)
    
    # 参数更新 - DeepSpeed会检查是否达到梯度累积步数
    model_engine.step()
    
    # 计时
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    
    if iter_num % 1 == 0 and master_process:
        lossf = loss.item()
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")

if master_process:
    print(f"\n{'='*60}")
    print("训练完成！")
    print(f"最终模型保存在: {out_dir}")
    print(f"{'='*60}")

# 保存最终模型（所有进程都需要参与）
if master_process:
    print(f"\n保存最终检查点...")
model_engine.save_checkpoint(out_dir, tag="final")
if master_process:
    print(f"✅ 最终模型已保存到: {out_dir}/final/")

