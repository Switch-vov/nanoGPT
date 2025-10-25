# Train.py 完全解析 - 总结与实验指南

## 🎯 核心概念速查表

### 1. 数据流动

```
原始文本 → 编码 → .bin文件 → get_batch() → 模型 → loss
  "To be"    [5,10]   train.bin    张量[4,8]    前向   2.45
```

### 2. 训练一次迭代的完整流程

```python
# 伪代码（简化的train.py核心）

for iter_num in range(max_iters):
    
    # ========== 阶段1: 准备 ==========
    lr = get_lr(iter_num)  # 计算当前学习率
    
    # ========== 阶段2: 梯度累积 ==========
    optimizer.zero_grad()
    for micro_step in range(4):  # 累积4次
        X, Y = get_batch('train')  # [4, 8]
        logits, loss = model(X, Y)  # 前向传播
        loss = loss / 4  # 关键：除以累积步数
        loss.backward()  # 反向传播，梯度累加
    
    # ========== 阶段3: 更新参数 ==========
    clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
    optimizer.step()  # 更新参数
    
    # ========== 阶段4: 记录和评估 ==========
    if iter_num % 2000 == 0:
        val_loss = evaluate()
        save_checkpoint()
```

### 3. 重要公式

#### 损失函数（Cross Entropy）
```
loss = -log(P(正确答案))

例子：
P(正确) = 0.9  → loss = -log(0.9) = 0.11  ✅ 好
P(正确) = 0.5  → loss = -log(0.5) = 0.69  🤔 一般
P(正确) = 0.1  → loss = -log(0.1) = 2.30  ❌ 差
```

#### 梯度下降更新
```
参数(新) = 参数(旧) - 学习率 × 梯度

例子：
w_new = 0.5 - 0.001 × (-2.3) = 0.5 + 0.0023 = 0.5023
```

#### 学习率调度
```
Warmup阶段 (step < 2000):
  lr = max_lr × (step / warmup_steps)

正常训练阶段:
  lr = max_lr

Decay阶段:
  lr = min_lr + (max_lr - min_lr) × cos_decay
```

---

## 🧪 实验列表（从易到难）

### 实验1: 理解配置参数的影响 ⭐

**目标**：直观感受不同参数对训练的影响

```bash
# 基线
python train.py config/train_shakespeare_char.py --max_iters=1000 --out_dir=exp1_baseline

# 实验1A: 更大的学习率
python train.py config/train_shakespeare_char.py --max_iters=1000 --learning_rate=5e-3 --out_dir=exp1_large_lr

# 实验1B: 更小的学习率  
python train.py config/train_shakespeare_char.py --max_iters=1000 --learning_rate=1e-4 --out_dir=exp1_small_lr

# 实验1C: 更小的batch
python train.py config/train_shakespeare_char.py --max_iters=1000 --batch_size=16 --out_dir=exp1_small_batch

# 实验1D: 更大的模型
python train.py config/train_shakespeare_char.py --max_iters=1000 --n_layer=8 --n_embd=512 --out_dir=exp1_large_model
```

**观察重点**：
- loss下降速度
- 最终val loss
- 训练时间
- 生成质量

**分析模板**：
```
实验1A (大学习率):
  - 初期loss下降: [快/慢]
  - 是否收敛: [是/否/震荡]
  - 最终val loss: [数值]
  - 结论: [...]
```

---

### 实验2: 梯度累积验证 ⭐⭐

**目标**：验证梯度累积确实等价于大batch

```bash
# 方法A: 直接用大batch
python train.py config/train_shakespeare_char.py \
  --batch_size=64 \
  --gradient_accumulation_steps=1 \
  --max_iters=500 \
  --out_dir=exp2_large_batch

# 方法B: 小batch + 梯度累积
python train.py config/train_shakespeare_char.py \
  --batch_size=16 \
  --gradient_accumulation_steps=4 \
  --max_iters=500 \
  --out_dir=exp2_grad_accum
```

**观察重点**：
- 两者的loss曲线应该非常相似
- 训练时间可能略有差异
- 最终模型性能应该几乎相同

**理论预测**：
```
有效batch_size = batch_size × gradient_accumulation_steps

方法A: 64 × 1 = 64
方法B: 16 × 4 = 64  ✅ 相同！
```

---

### 实验3: 过拟合与正则化 ⭐⭐

**目标**：理解过拟合，学会使用dropout和weight_decay

```bash
# 基线（容易过拟合）
python train.py config/train_shakespeare_char.py \
  --max_iters=5000 \
  --dropout=0.0 \
  --weight_decay=0.0 \
  --out_dir=exp3_overfit

# 加dropout
python train.py config/train_shakespeare_char.py \
  --max_iters=5000 \
  --dropout=0.2 \
  --weight_decay=0.0 \
  --out_dir=exp3_dropout

# 加weight_decay
python train.py config/train_shakespeare_char.py \
  --max_iters=5000 \
  --dropout=0.0 \
  --weight_decay=0.1 \
  --out_dir=exp3_weight_decay

# 两者都加
python train.py config/train_shakespeare_char.py \
  --max_iters=5000 \
  --dropout=0.2 \
  --weight_decay=0.1 \
  --out_dir=exp3_both
```

**观察重点**：
```
关注train_loss和val_loss的差距：

过拟合迹象：
  train_loss = 1.2
  val_loss = 1.8  ← 差距大！
  
泛化良好：
  train_loss = 1.4
  val_loss = 1.5  ← 差距小 ✅
```

---

### 实验4: 学习率调度 ⭐⭐⭐

**目标**：理解warmup和decay的作用

```bash
# 无warmup, 无decay
python train.py config/train_shakespeare_char.py \
  --max_iters=2000 \
  --warmup_iters=0 \
  --decay_lr=False \
  --out_dir=exp4_no_schedule

# 只有warmup
python train.py config/train_shakespeare_char.py \
  --max_iters=2000 \
  --warmup_iters=100 \
  --decay_lr=False \
  --out_dir=exp4_warmup_only

# 只有decay
python train.py config/train_shakespeare_char.py \
  --max_iters=2000 \
  --warmup_iters=0 \
  --decay_lr=True \
  --lr_decay_iters=2000 \
  --out_dir=exp4_decay_only

# 两者都有（推荐）
python train.py config/train_shakespeare_char.py \
  --max_iters=2000 \
  --warmup_iters=100 \
  --decay_lr=True \
  --lr_decay_iters=2000 \
  --out_dir=exp4_full_schedule
```

**观察重点**：
- 前100步的loss变化（warmup影响）
- 最后200步的loss变化（decay影响）
- 整体收敛速度和稳定性

---

### 实验5: 模型规模实验 ⭐⭐⭐

**目标**：理解模型容量与性能的关系

```bash
# 超小模型
python train.py config/train_shakespeare_char.py \
  --n_layer=2 --n_head=2 --n_embd=128 \
  --max_iters=3000 \
  --out_dir=exp5_tiny

# 小模型
python train.py config/train_shakespeare_char.py \
  --n_layer=4 --n_head=4 --n_embd=256 \
  --max_iters=3000 \
  --out_dir=exp5_small

# 中模型（默认）
python train.py config/train_shakespeare_char.py \
  --n_layer=6 --n_head=6 --n_embd=384 \
  --max_iters=3000 \
  --out_dir=exp5_medium

# 大模型
python train.py config/train_shakespeare_char.py \
  --n_layer=8 --n_head=8 --n_embd=512 \
  --max_iters=3000 \
  --out_dir=exp5_large
```

**计算参数量**：
```python
# 粗略估算
params ≈ n_layer × (4 × n_embd² + 12 × n_embd²) + vocab_size × n_embd
       ≈ n_layer × 16 × n_embd² + vocab_size × n_embd

Tiny:   2 × 16 × 128² + 65 × 128 ≈ 532K
Small:  4 × 16 × 256² + 65 × 256 ≈ 4.2M
Medium: 6 × 16 × 384² + 65 × 384 ≈ 14.2M
Large:  8 × 16 × 512² + 65 × 512 ≈ 33.6M
```

**分析要点**：
- 模型越大，val loss越低（但收益递减）
- 训练时间增长
- 显存需求增长
- 生成质量提升

---

## 📊 结果分析工具

### 工具1: 绘制Loss曲线

创建文件 `plot_loss.py`：

```python
import matplotlib.pyplot as plt
import re

def parse_log(log_file):
    """解析训练日志"""
    steps = []
    train_losses = []
    val_losses = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # 解析类似 "step 0: train loss 4.1234, val loss 4.2345"
            match = re.search(r'step (\d+): train loss ([\d.]+), val loss ([\d.]+)', line)
            if match:
                steps.append(int(match.group(1)))
                train_losses.append(float(match.group(2)))
                val_losses.append(float(match.group(3)))
    
    return steps, train_losses, val_losses

# 使用
steps, train_loss, val_loss = parse_log('training.log')

plt.figure(figsize=(10, 6))
plt.plot(steps, train_loss, label='Train Loss')
plt.plot(steps, val_loss, label='Val Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training Progress')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve.png')
plt.show()
```

### 工具2: 比较多个实验

```python
import matplotlib.pyplot as plt

experiments = {
    'Baseline': 'exp1_baseline/training.log',
    'Large LR': 'exp1_large_lr/training.log',
    'Small LR': 'exp1_small_lr/training.log',
}

plt.figure(figsize=(12, 6))

for name, log_file in experiments.items():
    steps, _, val_loss = parse_log(log_file)
    plt.plot(steps, val_loss, label=name)

plt.xlabel('Steps')
plt.ylabel('Validation Loss')
plt.title('Experiment Comparison')
plt.legend()
plt.grid(True)
plt.savefig('comparison.png')
plt.show()
```

---

## 🔍 调试技巧

### 1. 检查数据加载

```python
# 在train.py开头添加
X, Y = get_batch('train')
print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")
print(f"X sample: {X[0]}")
print(f"Y sample: {Y[0]}")

# 解码查看实际文本
if meta_vocab_size:
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    itos = meta['itos']
    x_text = ''.join([itos[int(i)] for i in X[0]])
    y_text = ''.join([itos[int(i)] for i in Y[0]])
    print(f"X text: {x_text}")
    print(f"Y text: {y_text}")
```

### 2. 监控梯度

```python
# 在反向传播后添加
if iter_num % 100 == 0:
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    print(f"Gradient norm: {total_norm:.4f}")
    
    # 如果梯度过大，说明可能有问题
    if total_norm > 10:
        print("⚠️  警告：梯度很大，可能需要降低学习率或启用梯度裁剪")
```

### 3. 检查模型输出

```python
# 训练前测试一下
model.eval()
with torch.no_grad():
    X_test, Y_test = get_batch('val')
    logits, loss = model(X_test, Y_test)
    print(f"Initial loss: {loss.item():.4f}")
    
    # 理论最大loss（随机猜）
    random_guess_loss = -math.log(1.0 / vocab_size)
    print(f"Random guess loss: {random_guess_loss:.4f}")
    
    # 初始loss应该接近random_guess_loss
model.train()
```

---

## 💡 常见问题解答

### Q1: Loss一直不下降怎么办？

**检查清单**：
1. ✅ 学习率是否太小？试试增大10倍
2. ✅ 数据是否正确？检查X和Y是否对应
3. ✅ 模型是否太小？试试增大n_layer或n_embd
4. ✅ 是否有梯度？打印grad_norm检查

### Q2: Loss变成NaN怎么办？

**原因和解决**：
```
原因1: 学习率太大
  解决: 减小learning_rate到1/10

原因2: 梯度爆炸
  解决: 启用grad_clip=1.0

原因3: 数值不稳定
  解决: 使用float32而不是float16
```

### Q3: 过拟合怎么办？

**识别**：train_loss << val_loss

**解决方案**：
1. 增加dropout (0.1 → 0.2)
2. 增加weight_decay (0 → 0.1)
3. 减小模型 (n_layer -= 2)
4. 获取更多数据
5. 早停（val_loss不再下降就停止）

### Q4: 训练太慢怎么办？

**加速技巧**：
1. 启用compile=True（提速2x）
2. 增大batch_size（GPU利用率提高）
3. 减小eval_interval（少做评估）
4. 使用float16（显存减半，速度提升）
5. 多GPU训练（线性加速）

---

## 🎯 自测题

完成以下问题，检验你的理解：

1. **梯度累积**：
   - batch_size=8, gradient_accumulation_steps=4
   - 等效batch_size是多少？ [答案: 32]
   - 为什么loss要除以4？

2. **学习率**：
   - 如果loss震荡，应该增大还是减小lr？ [答案: 减小]
   - warmup的作用是什么？

3. **过拟合**：
   - train_loss=1.2, val_loss=1.8，这是过拟合吗？ [答案: 是]
   - 如何解决？

4. **梯度裁剪**：
   - grad_clip=1.0的含义是什么？
   - 什么时候需要梯度裁剪？

5. **数据加载**：
   - 为什么Y比X向右移一位？
   - 为什么要随机采样而不是顺序读取？

---

## 🚀 进阶挑战

完成这些挑战，成为真正的高手：

1. **实现你自己的get_batch()**
   - 支持序列打包
   - 支持动态padding

2. **实现learning rate finder**
   - 自动寻找最优学习率
   - 参考：Leslie Smith的论文

3. **实现早停（Early Stopping）**
   - 当val_loss连续N步不降就停止
   - 保存最佳模型

4. **实现梯度可视化**
   - 每层的梯度分布
   - 识别梯度消失/爆炸

5. **实现数据增强**
   - 随机mask部分token
   - 参考BERT的MLM

---

## 📚 推荐阅读

1. **优化器**：
   - Adam论文：https://arxiv.org/abs/1412.6980
   - AdamW论文：https://arxiv.org/abs/1711.05101

2. **学习率调度**：
   - Cosine Annealing：https://arxiv.org/abs/1608.03983
   - Learning Rate Finder：https://arxiv.org/abs/1506.01186

3. **正则化**：
   - Dropout：https://jmlr.org/papers/v15/srivastava14a.html
   - Weight Decay vs L2：https://arxiv.org/abs/1711.05101

4. **梯度**：
   - 反向传播详解：http://cs231n.github.io/optimization-2/
   - 梯度裁剪：https://arxiv.org/abs/1211.5063

---

**恭喜你完成train.py的深度学习！🎉**

现在你已经理解了：
- ✅ 完整的训练循环流程
- ✅ 每个超参数的作用
- ✅ 各种训练技巧
- ✅ 如何调试和优化

**准备好学习model.py了吗？** 那里才是GPT真正的"大脑"所在！
