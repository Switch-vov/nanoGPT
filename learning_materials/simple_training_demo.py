#!/usr/bin/env python3
"""
训练循环核心概念演示 - 纯Python版本
不需要任何外部库，帮助理解核心原理
"""

import math
import random

print("="*80)
print("🎓 训练循环核心概念演示")
print("="*80)
print()

# ============================================================================
# 演示1: 最简单的梯度下降
# ============================================================================
print("📝 演示1: 理解梯度下降")
print("-" * 80)
print()
print("问题: 学习函数 y = 2x")
print("目标: 找到参数 w = 2")
print()

# 训练数据
x_data = [1.0, 2.0, 3.0, 4.0, 5.0]
y_data = [2.0, 4.0, 6.0, 8.0, 10.0]

# 初始参数（随机猜测）
w = 0.5
learning_rate = 0.1

print(f"初始状态: w = {w:.4f}")
print(f"学习率: {learning_rate}")
print()
print("开始训练...")
print()

for epoch in range(20):
    # 前向传播：计算预测和损失
    total_loss = 0
    total_gradient = 0
    
    for x, y_true in zip(x_data, y_data):
        # 预测
        y_pred = w * x
        
        # 损失（均方误差）
        loss = (y_pred - y_true) ** 2
        total_loss += loss
        
        # 梯度（手动计算）
        # loss = (w*x - y)^2
        # d(loss)/d(w) = 2*(w*x - y)*x
        gradient = 2 * (y_pred - y_true) * x
        total_gradient += gradient
    
    # 平均
    avg_loss = total_loss / len(x_data)
    avg_gradient = total_gradient / len(x_data)
    
    # 更新参数
    w = w - learning_rate * avg_gradient
    
    # 打印进度
    if epoch % 4 == 0 or epoch == 19:
        print(f"  Epoch {epoch:2d}: w = {w:.6f}, loss = {avg_loss:.6f}, gradient = {avg_gradient:+.6f}")

print()
print(f"✅ 最终结果: w = {w:.6f} (目标: 2.0)")
print(f"   误差: {abs(w - 2.0):.6f}")
print()

# ============================================================================
# 演示2: Batch Size的影响
# ============================================================================
print("📝 演示2: Batch Size的影响")
print("-" * 80)
print()

def train_with_batch(batch_size, epochs=10):
    """用指定batch size训练"""
    w = 0.5
    lr = 0.1
    
    for epoch in range(epochs):
        # 随机打乱数据
        indices = list(range(len(x_data)))
        random.shuffle(indices)
        
        # 按batch处理
        for i in range(0, len(x_data), batch_size):
            batch_indices = indices[i:i+batch_size]
            
            # 计算batch的梯度
            gradient = 0
            for idx in batch_indices:
                x = x_data[idx]
                y_true = y_data[idx]
                y_pred = w * x
                gradient += 2 * (y_pred - y_true) * x
            
            gradient /= len(batch_indices)
            
            # 更新
            w -= lr * gradient
    
    return w

print("比较不同batch size的训练结果：")
print()

for bs in [1, 2, 5]:
    random.seed(42)  # 固定随机种子以便比较
    final_w = train_with_batch(bs, epochs=20)
    print(f"  Batch Size = {bs}: w = {final_w:.6f}")

print()
print("观察：batch size影响训练过程，但最终都能收敛")
print()

# ============================================================================
# 演示3: 梯度累积
# ============================================================================
print("📝 演示3: 梯度累积 = 模拟大Batch")
print("-" * 80)
print()

# 方法1: 直接用大batch
def train_large_batch():
    w = 0.5
    lr = 0.1
    
    # 计算所有样本的梯度
    gradient = 0
    for x, y_true in zip(x_data, y_data):
        y_pred = w * x
        gradient += 2 * (y_pred - y_true) * x
    gradient /= len(x_data)
    
    # 更新
    w -= lr * gradient
    return w

# 方法2: 小batch + 梯度累积
def train_with_accumulation():
    w = 0.5
    lr = 0.1
    accumulation_steps = 5
    
    accumulated_gradient = 0
    
    # 累积5次
    for i in range(accumulation_steps):
        x = x_data[i]
        y_true = y_data[i]
        y_pred = w * x
        
        # 计算梯度并累加
        gradient = 2 * (y_pred - y_true) * x
        accumulated_gradient += gradient
    
    # 平均
    accumulated_gradient /= accumulation_steps
    
    # 更新
    w -= lr * accumulated_gradient
    return w

w1 = train_large_batch()
w2 = train_with_accumulation()

print(f"方法1 (大batch):      w = {w1:.6f}")
print(f"方法2 (梯度累积):     w = {w2:.6f}")
print(f"差异:                {abs(w1 - w2):.8f}")
print()
print("✅ 两种方法结果几乎相同！")
print()

# ============================================================================
# 演示4: 学习率调度
# ============================================================================
print("📝 演示4: 学习率调度")
print("-" * 80)
print()

def get_lr_with_warmup(step, max_lr=0.1, warmup_steps=10, total_steps=50):
    """带warmup的学习率调度"""
    if step < warmup_steps:
        # Warmup: 线性增长
        return max_lr * (step + 1) / warmup_steps
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return max_lr * 0.5 * (1 + math.cos(math.pi * progress))

print("学习率变化：")
print()
print("Step |  Learning Rate  | 阶段")
print("-----|-----------------|--------")

steps_to_show = [0, 5, 10, 15, 25, 40, 49]
for step in steps_to_show:
    lr = get_lr_with_warmup(step)
    if step < 10:
        phase = "Warmup"
    elif step < 40:
        phase = "Training"
    else:
        phase = "Decay"
    print(f"{step:4d} |  {lr:.6f}      | {phase}")

print()
print("可视化（ASCII图）：")
print()

# 简单的ASCII图
max_height = 20
for height in range(max_height, 0, -1):
    line = ""
    for step in range(0, 50, 2):
        lr = get_lr_with_warmup(step)
        normalized_height = int(lr / 0.1 * max_height)
        if normalized_height >= height:
            line += "█"
        else:
            line += " "
    print(f"{height:2d} | {line}")

print("    " + "-" * 25)
print("     0    10   20   30   40  (steps)")
print()

# ============================================================================
# 演示5: 梯度裁剪
# ============================================================================
print("📝 演示5: 梯度裁剪")
print("-" * 80)
print()

def clip_gradient(gradient, max_norm=1.0):
    """梯度裁剪"""
    # 计算梯度范数（绝对值）
    norm = abs(gradient)
    
    if norm > max_norm:
        # 缩放
        return gradient * (max_norm / norm)
    else:
        return gradient

print("梯度裁剪示例：")
print()
print("原始梯度 | 裁剪后 (max_norm=1.0)")
print("---------|----------------------")

test_gradients = [-5.0, -2.0, -0.5, 0.8, 2.5, 10.0]
for g in test_gradients:
    clipped = clip_gradient(g, max_norm=1.0)
    print(f"{g:8.2f} | {clipped:8.2f}")

print()
print("✅ 大梯度被缩小，防止参数更新过大")
print()

# ============================================================================
# 演示6: 完整训练循环模拟
# ============================================================================
print("📝 演示6: 完整训练循环（整合所有技巧）")
print("-" * 80)
print()

class SimpleTrainer:
    def __init__(self):
        self.w = 0.5
        self.best_loss = float('inf')
        self.best_w = self.w
    
    def train_step(self, batch_data, lr, use_grad_clip=True):
        """单步训练"""
        # 梯度累积
        accumulated_grad = 0
        total_loss = 0
        
        for x, y_true in batch_data:
            y_pred = self.w * x
            loss = (y_pred - y_true) ** 2
            gradient = 2 * (y_pred - y_true) * x
            
            accumulated_grad += gradient
            total_loss += loss
        
        # 平均
        accumulated_grad /= len(batch_data)
        avg_loss = total_loss / len(batch_data)
        
        # 梯度裁剪
        if use_grad_clip:
            accumulated_grad = clip_gradient(accumulated_grad, max_norm=2.0)
        
        # 更新参数
        self.w -= lr * accumulated_grad
        
        return avg_loss
    
    def train(self, epochs=30, batch_size=2):
        """完整训练循环"""
        print(f"配置: epochs={epochs}, batch_size={batch_size}")
        print()
        
        data = list(zip(x_data, y_data))
        
        for epoch in range(epochs):
            # 学习率调度
            lr = get_lr_with_warmup(epoch, max_lr=0.1, warmup_steps=5, total_steps=epochs)
            
            # 随机打乱
            random.shuffle(data)
            
            # 批次训练
            epoch_loss = 0
            num_batches = 0
            
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]
                loss = self.train_step(batch, lr)
                epoch_loss += loss
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            
            # 保存最佳
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.best_w = self.w
            
            # 打印
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:2d}: lr={lr:.6f}, w={self.w:.6f}, loss={avg_loss:.6f}")
        
        print()
        print(f"✅ 训练完成！")
        print(f"   最佳参数: w = {self.best_w:.6f}")
        print(f"   最佳损失: {self.best_loss:.6f}")

# 运行训练
random.seed(42)
trainer = SimpleTrainer()
trainer.train()

print()

# ============================================================================
# 总结
# ============================================================================
print("="*80)
print("🎓 总结")
print("="*80)
print()
print("你刚刚学到了：")
print()
print("1. ✅ 梯度下降基础")
print("   - 计算梯度（损失对参数的导数）")
print("   - 更新参数（w = w - lr * gradient）")
print()
print("2. ✅ Batch训练")
print("   - 小batch: 更新频繁，梯度噪声大")
print("   - 大batch: 更新稳定，但需要更多显存")
print()
print("3. ✅ 梯度累积")
print("   - 用小batch模拟大batch")
print("   - 关键: 累积多个小batch的梯度，然后一次更新")
print()
print("4. ✅ 学习率调度")
print("   - Warmup: 开始时学习率从小到大")
print("   - Decay: 后期逐渐降低学习率")
print()
print("5. ✅ 梯度裁剪")
print("   - 防止梯度爆炸")
print("   - 限制梯度的最大值")
print()
print("6. ✅ 完整训练循环")
print("   - 整合所有技巧")
print("   - 这就是train.py的核心！")
print()
print("="*80)
print("🚀 现在你完全理解训练循环了！")
print("="*80)
print()
print("下一步：")
print("1. 运行真实训练: python train.py config/train_shakespeare_char.py")
print("2. 尝试不同的超参数")
print("3. 观察训练过程")
print("4. 生成文本看效果: python sample.py --out_dir=out-shakespeare-char")
print()
