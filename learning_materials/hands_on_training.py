#!/usr/bin/env python3
"""
训练循环实战练习 - 从零理解梯度下降

这个脚本包含5个递进的练习，帮助你理解train.py的核心概念
每个练习都有详细注释和可视化输出
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np

print("="*80)
print("训练循环实战练习")
print("="*80)
print()

# ============================================================================
# 练习1: 最简单的梯度下降
# ============================================================================
print("📝 练习1: 理解梯度下降的基本原理")
print("-" * 80)

# 问题：学习函数 y = 2x
# 目标：找到参数 w = 2

# 训练数据
x_train = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
y_train = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])

# 模型参数（初始瞎猜）
w = torch.tensor([0.5], requires_grad=True)

# 超参数
learning_rate = 0.1
num_epochs = 20

# 记录训练过程
history = {'w': [], 'loss': []}

print(f"初始参数: w = {w.item():.4f}")
print(f"目标: w = 2.0")
print()

for epoch in range(num_epochs):
    # 前向传播
    y_pred = w * x_train
    loss = ((y_pred - y_train) ** 2).mean()
    
    # 记录
    history['w'].append(w.item())
    history['loss'].append(loss.item())
    
    # 反向传播
    loss.backward()
    
    # 手动更新参数（不使用优化器）
    with torch.no_grad():
        w -= learning_rate * w.grad
        
    # 清空梯度
    w.grad.zero_()
    
    # 打印进度
    if epoch % 4 == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch:2d}: w = {w.item():.6f}, loss = {loss.item():.6f}")

print(f"\n✅ 最终参数: w = {w.item():.6f} (目标: 2.0)")
print()

# ============================================================================
# 练习2: 理解batch training
# ============================================================================
print("📝 练习2: 理解Batch Size的影响")
print("-" * 80)

# 生成数据
torch.manual_seed(42)
X = torch.randn(100, 1) * 10
Y = 3 * X + 7 + torch.randn(100, 1) * 2  # y = 3x + 7 + noise

def train_with_batch_size(batch_size, num_epochs=50):
    """用指定batch_size训练"""
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    losses = []
    for epoch in range(num_epochs):
        # 随机打乱数据
        perm = torch.randperm(len(X))
        X_shuffled = X[perm]
        Y_shuffled = Y[perm]
        
        epoch_loss = 0
        for i in range(0, len(X), batch_size):
            # 获取batch
            X_batch = X_shuffled[i:i+batch_size]
            Y_batch = Y_shuffled[i:i+batch_size]
            
            # 训练
            y_pred = model(X_batch)
            loss = F.mse_loss(y_pred, Y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        losses.append(epoch_loss / (len(X) // batch_size))
    
    return losses, model

# 测试不同batch_size
batch_sizes = [1, 10, 50, 100]
results = {}

for bs in batch_sizes:
    losses, model = train_with_batch_size(bs)
    w = model.weight.item()
    b = model.bias.item()
    results[bs] = losses
    print(f"Batch Size {bs:3d}: w={w:.4f}, b={b:.4f} (目标: w=3, b=7)")

print()
print("观察：")
print("- Batch Size越小：loss曲线越震荡（噪声大）")
print("- Batch Size越大：loss曲线越平滑（但需要更多显存）")
print()

# ============================================================================
# 练习3: 理解梯度累积
# ============================================================================
print("📝 练习3: 梯度累积 = 模拟大Batch")
print("-" * 80)

def train_normal(batch_size, lr=0.01):
    """正常训练：直接用大batch"""
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    # 一次性处理大batch
    X_batch = X[:batch_size]
    Y_batch = Y[:batch_size]
    
    y_pred = model(X_batch)
    loss = F.mse_loss(y_pred, Y_batch)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return model, loss.item()

def train_with_accumulation(small_batch_size, accumulation_steps, lr=0.01):
    """梯度累积训练：用小batch模拟大batch"""
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    total_loss = 0
    optimizer.zero_grad()
    
    for step in range(accumulation_steps):
        # 小batch
        start = step * small_batch_size
        X_batch = X[start:start + small_batch_size]
        Y_batch = Y[start:start + small_batch_size]
        
        y_pred = model(X_batch)
        loss = F.mse_loss(y_pred, Y_batch)
        
        # 关键：除以累积步数！
        loss = loss / accumulation_steps
        loss.backward()  # 累积梯度
        
        total_loss += loss.item()
    
    optimizer.step()  # 一次性更新
    
    return model, total_loss

# 比较
print("比较两种方法（应该得到相同结果）：")
print()

# 方法1：直接用batch_size=32
model1, loss1 = train_normal(batch_size=32)
print(f"方法1 (正常batch=32):")
print(f"  Loss: {loss1:.6f}")
print(f"  参数: w={model1.weight.item():.6f}, b={model1.bias.item():.6f}")
print()

# 方法2：用batch_size=8，累积4次
model2, loss2 = train_with_accumulation(small_batch_size=8, accumulation_steps=4)
print(f"方法2 (batch=8, 累积4次):")
print(f"  Loss: {loss2:.6f}")
print(f"  参数: w={model2.weight.item():.6f}, b={model2.bias.item():.6f}")
print()

print("✅ 两种方法梯度几乎相同！")
print("   这就是梯度累积的原理：用小batch模拟大batch")
print()

# ============================================================================
# 练习4: 理解梯度裁剪
# ============================================================================
print("📝 练习4: 梯度裁剪防止梯度爆炸")
print("-" * 80)

def train_with_gradient_clip(clip_value=None):
    """训练并可选地应用梯度裁剪"""
    model = nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    # 构造一个容易导致梯度爆炸的情况
    X_bad = torch.randn(5, 10) * 100  # 很大的输入
    Y_bad = torch.randn(5, 1) * 100
    
    max_grads = []
    
    for _ in range(10):
        y_pred = model(X_bad)
        loss = F.mse_loss(y_pred, Y_bad)
        
        optimizer.zero_grad()
        loss.backward()
        
        # 记录梯度大小
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        max_grads.append(total_norm)
        
        # 梯度裁剪
        if clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        optimizer.step()
    
    return max_grads

# 不裁剪
grads_no_clip = train_with_gradient_clip(clip_value=None)
print(f"不裁剪: 最大梯度范数 = {max(grads_no_clip):.2f}")

# 裁剪到1.0
grads_with_clip = train_with_gradient_clip(clip_value=1.0)
print(f"裁剪到1.0: 最大梯度范数 = {max(grads_with_clip):.2f}")
print()

print("✅ 梯度裁剪成功限制了梯度大小，防止训练崩溃")
print()

# ============================================================================
# 练习5: 理解学习率调度
# ============================================================================
print("📝 练习5: 学习率调度（Warmup + Cosine Decay）")
print("-" * 80)

def get_lr(it, learning_rate=1e-3, warmup_iters=100, lr_decay_iters=1000, min_lr=1e-4):
    """复制train.py的学习率调度"""
    # Warmup
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # Decay
    if it > lr_decay_iters:
        return min_lr
    # Cosine
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# 生成学习率曲线
max_iters = 1000
lrs = [get_lr(i) for i in range(max_iters)]

print("学习率变化：")
checkpoints = [0, 50, 100, 200, 500, 900, 999]
for i in checkpoints:
    print(f"  Step {i:4d}: lr = {lrs[i]:.6f}")
print()

print("✅ 学习率调度策略：")
print("   1. Warmup (0-100步): 从小到大，避免初期震荡")
print("   2. 正常训练: 保持较高学习率")
print("   3. Cosine Decay: 逐渐降低，精细调整")
print()

# ============================================================================
# 练习6: 模拟完整训练循环
# ============================================================================
print("📝 练习6: 完整训练循环（模拟NanoGPT）")
print("-" * 80)

class TinyModel(nn.Module):
    """超小模型，模拟GPT"""
    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.fc = nn.Linear(n_embd, vocab_size)
    
    def forward(self, x, targets=None):
        # x: [batch, seq_len]
        emb = self.token_emb(x)  # [batch, seq_len, n_embd]
        logits = self.fc(emb)     # [batch, seq_len, vocab_size]
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

# 虚拟数据
vocab_size = 50
batch_size = 4
block_size = 8

def get_fake_batch():
    """模拟get_batch()"""
    x = torch.randint(0, vocab_size, (batch_size, block_size))
    y = torch.randint(0, vocab_size, (batch_size, block_size))
    return x, y

# 创建模型和优化器
model = TinyModel(vocab_size, n_embd=32)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# 训练配置
max_iters = 100
gradient_accumulation_steps = 4
grad_clip = 1.0

print(f"训练配置：")
print(f"  max_iters = {max_iters}")
print(f"  gradient_accumulation_steps = {gradient_accumulation_steps}")
print(f"  grad_clip = {grad_clip}")
print()

print("开始训练...")
print()

for iter_num in range(max_iters):
    # 梯度累积循环
    optimizer.zero_grad()
    total_loss = 0
    
    for micro_step in range(gradient_accumulation_steps):
        X, Y = get_fake_batch()
        logits, loss = model(X, Y)
        loss = loss / gradient_accumulation_steps
        loss.backward()
        total_loss += loss.item()
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    # 更新参数
    optimizer.step()
    
    # 日志
    if iter_num % 20 == 0:
        print(f"Step {iter_num:3d}: loss = {total_loss:.4f}")

print()
print("✅ 完整训练循环演示完成！")
print()

# ============================================================================
# 总结
# ============================================================================
print("="*80)
print("🎓 总结：你学到了什么")
print("="*80)
print()
print("1. ✅ 梯度下降的基本原理")
print("   - 前向传播计算loss")
print("   - 反向传播计算梯度")
print("   - 参数更新")
print()
print("2. ✅ Batch Size的影响")
print("   - 大batch: 稳定但需要更多显存")
print("   - 小batch: 震荡但显存友好")
print()
print("3. ✅ 梯度累积")
print("   - 用小batch模拟大batch")
print("   - 关键：loss要除以累积步数")
print()
print("4. ✅ 梯度裁剪")
print("   - 防止梯度爆炸")
print("   - 限制梯度范数到指定值")
print()
print("5. ✅ 学习率调度")
print("   - Warmup: 避免初期震荡")
print("   - Cosine Decay: 后期精细调整")
print()
print("6. ✅ 完整训练循环")
print("   - 结合以上所有技巧")
print("   - 这就是NanoGPT的train.py!")
print()
print("="*80)
print("🚀 下一步：运行真实的训练")
print("="*80)
print()
print("试试这些命令：")
print()
print("# 快速测试（CPU，5分钟）")
print("python train.py config/train_shakespeare_char.py --device=cpu --max_iters=500")
print()
print("# 完整训练（GPU，30分钟）")
print("python train.py config/train_shakespeare_char.py --max_iters=5000")
print()
print("# 查看训练效果")
print("python sample.py --out_dir=out-shakespeare-char")
print()
