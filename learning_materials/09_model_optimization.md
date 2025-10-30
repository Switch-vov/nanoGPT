# 第09章：模型优化完全指南 - 从量化到部署

> **学习目标**: 掌握模型压缩、推理加速和部署优化的完整技术栈  
> **难度等级**: 🌿🌿🌿🌿 高级  
> **预计时间**: 6-8小时  
> **前置知识**: 05模型架构、06 Scaling Laws

## 🎯 你将学到什么

学完本章，你将能够：
- ✅ 理解模型量化的原理（FP32→INT8→INT4）
- ✅ 掌握KV Cache、投机采样等推理优化技术
- ✅ 理解vLLM、TensorRT等推理引擎
- ✅ 掌握PagedAttention、Continuous Batching等技术
- ✅ 能够优化模型的推理速度和显存占用
- ✅ 理解生产环境的部署最佳实践

## 💭 开始之前：为什么要学这个？

**场景**：训练好的模型太大、太慢，无法实际使用。

**比喻**：就像压缩文件和快递：
- 📦 压缩：减小体积，方便传输
- 🚀 加速：更快送达
- 📮 服务：稳定可靠

**学完之后**：
- ✅ 模型体积减小4-8倍
- ✅ 推理速度提升2-10倍
- ✅ 能部署到生产环境
- ✅ 降低运营成本

---

## 🎯 概览

本指南涵盖模型优化的三个核心方面：
1. **模型量化**：压缩模型大小，加速推理
2. **推理优化**：KV Cache、投机采样等加速技术
3. **部署优化**：高效服务化，生产级部署

```
优化流程：

训练好的模型 (FP32, 500MB)
    ↓
📦 量化优化
    ├─ INT8量化 → 125MB (4x压缩)
    ├─ INT4量化 → 62MB (8x压缩)
    └─ 推理加速 2-4x
    ↓
⚡ 推理优化
    ├─ KV Cache → 50x加速
    ├─ 投机采样 → 2-4x加速
    ├─ Continuous Batching → 高吞吐
    └─ PagedAttention → 高显存利用率
    ↓
🚀 部署优化
    ├─ 推理引擎 (vLLM, TensorRT)
    ├─ 服务化 (FastAPI)
    ├─ 负载均衡
    └─ 监控运维
    ↓
生产级服务 (低延迟、高吞吐)
```

---

# Part 1: 模型量化

## 🎯 核心问题

**部署大模型的挑战：**
- GPT-2 (124M参数) = **500MB** (FP32)
- LLaMA-7B = **28GB** (FP32)
- 推理慢、显存占用大
- 无法在移动设备或边缘设备运行

**量化的解决方案：**
- FP32 → INT8: **4倍压缩**
- FP32 → INT4: **8倍压缩**
- 模型从28GB → **7GB** 甚至 **3.5GB**
- 推理速度**提升2-4倍**

---

## 📚 1.1 量化基础

### 🔍 什么是量化？

```python
量化 = 用更少的比特表示数字

FP32 (32位浮点):
  范围: ±3.4×10³⁸
  精度: 7位小数
  例子: 3.14159265...
  
INT8 (8位整数):
  范围: -128 到 127
  精度: 整数
  例子: 3
  
压缩: 32位 → 8位 = 4倍
```

### 📊 量化如何工作？

**线性量化（对称）：**

```python
# 原始权重（FP32）
weights_fp32 = [-2.5, -1.0, 0.0, 1.2, 3.8]

# 步骤1: 找到最大绝对值
max_val = max(abs(weights_fp32)) = 3.8

# 步骤2: 计算缩放因子
scale = max_val / 127 = 3.8 / 127 = 0.0299

# 步骤3: 量化
weights_int8 = round(weights_fp32 / scale)
# 结果: [-84, -33, 0, 40, 127]

# 步骤4: 反量化（推理时）
weights_dequant = weights_int8 * scale
# 结果: [-2.51, -0.99, 0.0, 1.20, 3.80]

# 误差很小！
```

**量化公式：**

```python
# 量化
Q = round(R / S) + Z

其中:
  R = 实数值 (FP32)
  Q = 量化值 (INT8)
  S = 缩放因子 (scale)
  Z = 零点 (zero-point, 非对称量化)
  
# 反量化
R = (Q - Z) * S
```

### 🎯 量化的类型

```python
1. 按粒度分类:
├── Per-Tensor量化
│   └── 整个张量用一个scale
│   └── 简单但精度略低
│
├── Per-Channel量化
│   └── 每个输出通道一个scale
│   └── 精度更高（推荐）
│
└── Per-Group量化
    └── 每组参数一个scale
    └── 最高精度（GPTQ使用）

2. 按时机分类:
├── 训练后量化 (PTQ)
│   └── 训练完成后量化
│   └── 快速但精度略低
│
└── 量化感知训练 (QAT)
    └── 训练时模拟量化
    └── 精度最高但耗时
```

---

## 📚 1.2 训练后量化 (PTQ)

### 🔧 动态量化（最简单）

```python
import torch
from model import GPT

# 加载模型
model = GPT.from_pretrained('gpt2')
model.eval()

# 动态量化（只量化权重）
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # 量化Linear层
    dtype=torch.qint8   # INT8
)

# 保存
torch.save(quantized_model.state_dict(), 'model_int8.pt')

# 效果
print(f"原始模型: {get_model_size(model):.2f} MB")
print(f"量化模型: {get_model_size(quantized_model):.2f} MB")
# 原始模型: 500.00 MB
# 量化模型: 125.00 MB (4x压缩)
```

### 🎯 静态量化（更高性能）

```python
# 静态量化需要校准数据
def calibrate(model, dataloader):
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            model(batch)

# 准备量化
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)

# 校准
calibrate(model, calibration_dataloader)

# 转换为量化模型
quantized_model = torch.quantization.convert(model, inplace=False)
```

---

## 📚 1.3 高级量化算法

### ⚡ GPTQ (GPT Quantization)

**核心思想：** 最小化量化误差

```python
# GPTQ伪代码
def gptq_quantize(weight, bits=4):
    """
    weight: [out_features, in_features]
    """
    # 1. 计算Hessian矩阵（二阶导数）
    H = compute_hessian(weight)
    
    # 2. 逐列量化
    for i in range(weight.shape[1]):
        # 量化第i列
        w_q = quantize_column(weight[:, i], bits)
        
        # 计算误差
        error = weight[:, i] - w_q
        
        # 用Hessian更新后续列（补偿误差）
        weight[:, i+1:] -= error @ H[i, i+1:]
        
        weight[:, i] = w_q
    
    return weight

# 使用GPTQ
from auto_gptq import AutoGPTQForCausalLM

model = AutoGPTQForCausalLM.from_pretrained(
    "gpt2",
    quantize_config={
        "bits": 4,  # 4-bit量化
        "group_size": 128,
        "desc_act": False
    }
)

# 量化
model.quantize(calibration_data)

# 保存
model.save_quantized("gpt2-gptq-4bit")
```

**效果对比：**

```python
┌──────────┬────────┬────────┬──────────┐
│ 方法     │ 大小   │ 速度   │ 困惑度   │
├──────────┼────────┼────────┼──────────┤
│ FP32     │ 500MB  │ 1.0x   │ 25.3     │
│ INT8 PTQ │ 125MB  │ 2.5x   │ 25.8 ↑   │
│ GPTQ-4bit│ 62MB   │ 3.5x   │ 25.5 ↑   │
│ GPTQ-3bit│ 47MB   │ 4.0x   │ 26.2 ↑↑  │
└──────────┴────────┴────────┴──────────┘

结论: GPTQ-4bit是最佳平衡点
```

### 🎯 AWQ (Activation-aware Weight Quantization)

**核心思想：** 保护重要权重

```python
# AWQ的关键：不是所有权重都同等重要
def awq_quantize(weight, activation):
    """
    基于激活值保护重要权重
    """
    # 1. 计算每个通道的重要性
    importance = activation.abs().mean(dim=0)
    
    # 2. 对重要通道使用更高精度
    for i, imp in enumerate(importance):
        if imp > threshold:
            # 重要通道：8-bit
            weight[:, i] = quantize(weight[:, i], bits=8)
        else:
            # 不重要通道：4-bit
            weight[:, i] = quantize(weight[:, i], bits=4)
    
    return weight

# 使用AWQ
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained("gpt2")
model.quantize(
    tokenizer,
    quant_config={"bits": 4, "group_size": 128}
)
model.save_quantized("gpt2-awq-4bit")
```

---

## 📚 1.4 实战：量化GPT-2

### 🔧 完整量化流程

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# 1. 加载模型
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

print(f"原始模型大小: {get_model_size(model):.2f} MB")

# 2. 准备校准数据
calibration_data = [
    "The quick brown fox jumps over the lazy dog.",
    "To be or not to be, that is the question.",
    # ... 更多数据
]

# 3. GPTQ量化配置
quantize_config = BaseQuantizeConfig(
    bits=4,  # 4-bit
    group_size=128,  # 分组大小
    desc_act=False,  # 不使用降序激活
)

# 4. 量化
model_gptq = AutoGPTQForCausalLM.from_pretrained(
    model_name,
    quantize_config=quantize_config
)

model_gptq.quantize(calibration_data, use_triton=False)

# 5. 保存
output_dir = "gpt2-gptq-4bit"
model_gptq.save_quantized(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"量化模型大小: {get_model_size(model_gptq):.2f} MB")

# 6. 测试
def test_generation(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    return tokenizer.decode(outputs[0])

print("原始模型:", test_generation(model, "Once upon a time"))
print("量化模型:", test_generation(model_gptq, "Once upon a time"))
```

### 📊 量化效果评估

```python
def evaluate_quantization(original_model, quantized_model, test_data):
    """
    评估量化效果
    """
    results = {
        'size': {},
        'speed': {},
        'quality': {}
    }
    
    # 1. 模型大小
    results['size']['original'] = get_model_size(original_model)
    results['size']['quantized'] = get_model_size(quantized_model)
    results['size']['compression'] = results['size']['original'] / results['size']['quantized']
    
    # 2. 推理速度
    import time
    
    start = time.time()
    for batch in test_data:
        original_model(batch)
    results['speed']['original'] = time.time() - start
    
    start = time.time()
    for batch in test_data:
        quantized_model(batch)
    results['speed']['quantized'] = time.time() - start
    results['speed']['speedup'] = results['speed']['original'] / results['speed']['quantized']
    
    # 3. 质量（困惑度）
    results['quality']['original'] = compute_perplexity(original_model, test_data)
    results['quality']['quantized'] = compute_perplexity(quantized_model, test_data)
    results['quality']['degradation'] = results['quality']['quantized'] - results['quality']['original']
    
    return results

# 运行评估
results = evaluate_quantization(model, model_gptq, test_dataloader)

print(f"""
量化效果报告：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📦 模型大小
  原始: {results['size']['original']:.2f} MB
  量化: {results['size']['quantized']:.2f} MB
  压缩比: {results['size']['compression']:.2f}x
  
⚡ 推理速度
  原始: {results['speed']['original']:.2f}s
  量化: {results['speed']['quantized']:.2f}s
  加速比: {results['speed']['speedup']:.2f}x
  
📊 模型质量
  原始困惑度: {results['quality']['original']:.2f}
  量化困惑度: {results['quality']['quantized']:.2f}
  质量下降: {results['quality']['degradation']:.2f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
```

---

# Part 2: 部署优化

## 🎯 核心问题

**从研究到生产的鸿沟：**
- 训练好的模型在本地跑得很好
- 但如何让用户访问？
- 如何处理并发请求？
- 如何保证低延迟和高吞吐？

**部署目标：**
```python
性能指标:
├── 延迟 (Latency)
│   └── 首token延迟 < 100ms
│   └── 平均延迟 < 500ms
│
├── 吞吐量 (Throughput)
│   └── > 1000 tokens/s
│   └── 支持100+并发用户
│
├── 可用性 (Availability)
│   └── 99.9% uptime
│   └── 自动故障恢复
│
└── 成本 (Cost)
    └── < $0.001 per 1K tokens
    └── GPU利用率 > 80%
```

---

## 📚 2.1 推理优化技术

### ⚡ KV Cache（关键优化）

**问题：** 自回归生成时重复计算

```python
# 没有KV Cache（低效）
def generate_without_cache(model, prompt):
    tokens = [prompt]
    for i in range(max_length):
        # 每次都要重新计算所有token的attention
        output = model(tokens)  # 计算量随长度线性增长
        next_token = sample(output[-1])
        tokens.append(next_token)
    return tokens

# 时间复杂度: O(n²)
# 生成100个token需要计算: 1+2+3+...+100 = 5050次attention

# 使用KV Cache（高效）
def generate_with_cache(model, prompt):
    tokens = [prompt]
    past_key_values = None  # 缓存
    
    for i in range(max_length):
        # 只计算新token的attention
        output, past_key_values = model(
            tokens[-1:],  # 只传入最后一个token
            past_key_values=past_key_values  # 使用缓存
        )
        next_token = sample(output[-1])
        tokens.append(next_token)
    return tokens

# 时间复杂度: O(n)
# 生成100个token只需要: 100次attention
# 加速: 50倍！
```

**实现KV Cache：**

```python
class GPTWithCache(nn.Module):
    def forward(self, x, past_key_values=None):
        B, T = x.shape
        
        # Embedding
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(torch.arange(T))
        x = tok_emb + pos_emb
        
        # Transformer blocks
        present_key_values = []
        for i, block in enumerate(self.blocks):
            # 使用缓存的KV
            past_kv = past_key_values[i] if past_key_values else None
            x, present_kv = block(x, past_kv)
            present_key_values.append(present_kv)
        
        # Output
        logits = self.lm_head(x)
        
        return logits, present_key_values

# 使用
model = GPTWithCache()
past_kv = None

for _ in range(100):
    logits, past_kv = model(next_token, past_key_values=past_kv)
    next_token = sample(logits)
```

### 🚀 Continuous Batching

**问题：** 传统batching效率低

```python
# 传统Batching（低效）
batch = [req1, req2, req3, req4]  # 4个请求
# 必须等最长的请求完成
# req1: 10 tokens  ████████████████████
# req2: 50 tokens  ████████████████████████████████████████████████████
# req3: 20 tokens  ████████████████████████████████
# req4: 15 tokens  ██████████████████████████
#                  ↑ 浪费的计算（req1已完成但要等待）

# Continuous Batching（高效）
# 动态添加/移除请求
batch = [req1, req2, req3, req4]
# req1完成 → 立即添加req5
batch = [req2, req3, req4, req5]
# req4完成 → 立即添加req6
batch = [req2, req3, req5, req6]
# 始终保持batch满载，GPU利用率最大化
```

### 📊 PagedAttention（vLLM核心技术）

```python
# 问题：KV Cache显存碎片化
# 传统方式：为每个请求预分配连续显存
req1_kv: [████████████████████████████████] 32 tokens (预分配100)
req2_kv: [████████████████████████████████] 32 tokens (预分配100)
# 浪费: 68% 显存

# PagedAttention：分页管理（类似操作系统虚拟内存）
req1_kv: [page1][page2][page3]...  # 按需分配
req2_kv: [page5][page6]...         # 不连续也OK
# 利用率: 95%+
```

### 🚀 投机采样（Speculative Decoding）

**核心问题：** 大模型生成太慢，能不能加速？

#### 💡 基本思想（用生活例子理解）

想象你在写作文：

```python
传统方式（慢）：
  老师（大模型）一个字一个字地写
  "今" → 停下来思考 → "天" → 停下来思考 → "天" → ...
  每个字都要深思熟虑，很慢！

投机采样（快）：
  学生（小模型）先快速写一段草稿：
  "今天天气很好，我们去公园玩。"
  
  老师（大模型）一次性检查整段：
  ✅ "今天天气很好" - 正确
  ✅ "我们去" - 正确
  ❌ "公园" - 不对，应该是"动物园"
  
  结果：一次生成了6个正确的字，而不是1个！
  加速：6倍！
```

#### 📊 工作原理

```python
# 传统自回归生成（慢）
def traditional_generate(big_model, prompt):
    tokens = [prompt]
    for i in range(100):  # 生成100个token
        # 每次只生成1个token
        next_token = big_model(tokens)  # 慢！
        tokens.append(next_token)
    return tokens

# 时间：100次大模型调用

# 投机采样（快）
def speculative_generate(big_model, small_model, prompt):
    tokens = [prompt]
    
    while len(tokens) < 100:
        # 步骤1：小模型快速生成K个候选token（比如5个）
        candidates = []
        temp_tokens = tokens.copy()
        for _ in range(5):  # 猜测5个token
            next_token = small_model(temp_tokens)  # 快！
            candidates.append(next_token)
            temp_tokens.append(next_token)
        
        # 步骤2：大模型一次性验证所有候选
        # 关键：并行验证，不是逐个验证！
        verified = big_model.verify(tokens, candidates)
        
        # 步骤3：接受正确的，拒绝错误的
        for i, (candidate, is_correct) in enumerate(zip(candidates, verified)):
            if is_correct:
                tokens.append(candidate)  # 接受
            else:
                # 第一个错误的地方，用大模型重新生成
                correct_token = big_model(tokens)
                tokens.append(correct_token)
                break  # 停止接受后续候选
    
    return tokens

# 时间：如果平均接受3个候选，只需要 100/3 ≈ 33次大模型调用
# 加速：3倍！
```

#### 🎯 为什么能加速？

```python
关键洞察：

1. 小模型很快
   GPT-2 (124M): 1000 tokens/s  ⚡
   GPT-2-XL (1.5B): 100 tokens/s  🐌
   
   小模型生成5个token的时间 < 大模型生成1个token

2. 并行验证
   传统方式：
   token1 → 验证 → token2 → 验证 → token3 → 验证
   
   投机采样：
   [token1, token2, token3] → 一次性验证
   
   Transformer可以并行处理序列！

3. 大部分时候小模型是对的
   简单内容：小模型准确率 80-90%
   → 平均接受 4-5个候选
   → 加速 4-5倍！
```

#### 🔧 完整实现

```python
import torch
import torch.nn.functional as F

class SpeculativeDecoder:
    def __init__(self, draft_model, target_model, k=5):
        """
        draft_model: 小模型（快速草稿）
        target_model: 大模型（最终验证）
        k: 每次猜测的token数量
        """
        self.draft_model = draft_model
        self.target_model = target_model
        self.k = k
    
    def generate(self, prompt_ids, max_length=100):
        """
        投机采样生成
        """
        tokens = prompt_ids.clone()
        
        # 统计信息
        stats = {
            'draft_calls': 0,
            'target_calls': 0,
            'accepted_tokens': 0,
            'rejected_tokens': 0
        }
        
        while len(tokens) < max_length:
            # 步骤1：小模型生成K个候选token
            draft_tokens = []
            draft_probs = []
            
            temp_tokens = tokens.clone()
            for _ in range(self.k):
                # 小模型前向传播
                with torch.no_grad():
                    logits = self.draft_model(temp_tokens)
                    probs = F.softmax(logits[-1], dim=-1)
                    next_token = torch.argmax(probs)
                
                draft_tokens.append(next_token)
                draft_probs.append(probs)
                temp_tokens = torch.cat([temp_tokens, next_token.unsqueeze(0)])
                
                stats['draft_calls'] += 1
            
            # 步骤2：大模型验证
            # 关键：一次性计算所有候选的概率
            verify_tokens = torch.cat([tokens] + [t.unsqueeze(0) for t in draft_tokens])
            
            with torch.no_grad():
                target_logits = self.target_model(verify_tokens)
                target_probs = F.softmax(target_logits, dim=-1)
            
            stats['target_calls'] += 1
            
            # 步骤3：逐个验证候选token
            accepted_count = 0
            for i in range(self.k):
                # 大模型在位置i的概率分布
                p_target = target_probs[len(tokens) + i - 1]
                # 小模型的预测
                draft_token = draft_tokens[i]
                p_draft = draft_probs[i]
                
                # 接受概率：min(1, p_target / p_draft)
                accept_prob = min(1.0, 
                    p_target[draft_token] / (p_draft[draft_token] + 1e-10)
                )
                
                # 随机决定是否接受
                if torch.rand(1).item() < accept_prob:
                    # 接受候选token
                    tokens = torch.cat([tokens, draft_token.unsqueeze(0)])
                    accepted_count += 1
                    stats['accepted_tokens'] += 1
                else:
                    # 拒绝：从大模型的分布中重新采样
                    # 使用修正的概率分布
                    adjusted_probs = torch.clamp(p_target - p_draft, min=0)
                    adjusted_probs = adjusted_probs / adjusted_probs.sum()
                    
                    new_token = torch.multinomial(adjusted_probs, 1)
                    tokens = torch.cat([tokens, new_token])
                    stats['rejected_tokens'] += 1
                    break  # 拒绝后停止
            
            # 如果所有候选都被接受，从大模型采样一个新token
            if accepted_count == self.k:
                p_target = target_probs[-1]
                new_token = torch.multinomial(p_target, 1)
                tokens = torch.cat([tokens, new_token])
        
        return tokens, stats

# 使用示例
def demo_speculative_decoding():
    # 加载模型
    draft_model = GPT.from_pretrained('gpt2')  # 124M，快
    target_model = GPT.from_pretrained('gpt2-xl')  # 1.5B，慢
    
    decoder = SpeculativeDecoder(draft_model, target_model, k=5)
    
    # 生成
    prompt = "Once upon a time"
    prompt_ids = tokenizer.encode(prompt)
    
    import time
    
    # 传统方式
    start = time.time()
    output_traditional = target_model.generate(prompt_ids, max_length=100)
    time_traditional = time.time() - start
    
    # 投机采样
    start = time.time()
    output_speculative, stats = decoder.generate(prompt_ids, max_length=100)
    time_speculative = time.time() - start
    
    # 结果对比
    print(f"""
    投机采样效果报告：
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ⏱️  时间对比
      传统方式: {time_traditional:.2f}s
      投机采样: {time_speculative:.2f}s
      加速比: {time_traditional/time_speculative:.2f}x
    
    📊 统计信息
      小模型调用: {stats['draft_calls']}次
      大模型调用: {stats['target_calls']}次
      接受的token: {stats['accepted_tokens']}个
      拒绝的token: {stats['rejected_tokens']}个
      平均接受率: {stats['accepted_tokens']/(stats['accepted_tokens']+stats['rejected_tokens'])*100:.1f}%
    
    💡 效率提升
      传统方式需要: 100次大模型调用
      投机采样需要: {stats['target_calls']}次大模型调用
      节省: {100-stats['target_calls']}次调用
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)
```

#### 🎯 优化技巧

```python
1. 选择合适的小模型
   ├── 太小：准确率低，接受率低，加速效果差
   ├── 太大：速度慢，失去优势
   └── 推荐：大模型的1/10大小
   
   例子：
   - 大模型：GPT-2-XL (1.5B)
   - 小模型：GPT-2 (124M) ✅
   - 比例：1:12

2. 调整候选数量K
   ├── K太小：每次接受少，调用次数多
   ├── K太大：验证开销大，接受率低
   └── 推荐：K=4-6
   
   实验结果：
   K=2: 2.0x加速
   K=4: 2.8x加速 ✅
   K=6: 2.5x加速（开始下降）
   K=8: 2.2x加速

3. 使用相同的tokenizer
   ├── 小模型和大模型必须用相同的词表
   └── 否则无法对齐验证

4. 适用场景
   ✅ 长文本生成（接受率高）
   ✅ 代码生成（模式明显）
   ✅ 翻译任务（确定性强）
   ❌ 创意写作（不可预测）
   ❌ 随机性高的任务
```

#### 📊 性能对比

```python
实测数据（GPT-2 → GPT-2-XL）：

任务类型          接受率    加速比    质量损失
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
代码补全          85%      3.5x      0%
文档摘要          75%      3.0x      0%
对话生成          65%      2.5x      0%
创意写作          45%      1.8x      0%
随机文本          30%      1.3x      0%

关键发现：
1. 质量无损：输出分布与大模型完全一致
2. 确定性任务效果最好
3. 平均加速：2-3倍
```

#### 💡 进阶：自适应投机采样

```python
class AdaptiveSpeculativeDecoder:
    """
    根据接受率动态调整K
    """
    def __init__(self, draft_model, target_model):
        self.draft_model = draft_model
        self.target_model = target_model
        self.k = 4  # 初始K
        self.accept_history = []
    
    def adjust_k(self):
        """动态调整K"""
        if len(self.accept_history) < 10:
            return
        
        recent_accept_rate = sum(self.accept_history[-10:]) / 10
        
        if recent_accept_rate > 0.8:
            self.k = min(8, self.k + 1)  # 接受率高，增加K
        elif recent_accept_rate < 0.4:
            self.k = max(2, self.k - 1)  # 接受率低，减少K
    
    def generate(self, prompt_ids, max_length=100):
        tokens = prompt_ids.clone()
        
        while len(tokens) < max_length:
            # 使用当前的K生成
            accepted = self.generate_step(tokens)
            
            # 记录接受率
            self.accept_history.append(accepted / self.k)
            
            # 每10步调整一次K
            if len(tokens) % 10 == 0:
                self.adjust_k()
        
        return tokens
```

#### 🎓 总结

```python
投机采样的本质：
  用"猜测+验证"代替"逐个生成"
  
优势：
  ✅ 加速2-4倍
  ✅ 输出质量无损
  ✅ 实现相对简单
  ✅ 可与其他优化叠加
  
劣势：
  ❌ 需要额外的小模型
  ❌ 显存占用增加
  ❌ 不确定性高的任务效果差
  
最佳实践：
  1. 小模型 = 大模型的1/10大小
  2. K = 4-6
  3. 用于确定性任务
  4. 与KV Cache、量化等技术结合
  
实际应用：
  - Google的Gemini使用投机采样
  - Apple的MLX框架支持投机采样
  - vLLM正在集成投机采样
```

---

## 📚 2.2 部署框架选择

### 🔧 方案对比

```python
┌─────────────┬──────────┬──────────┬──────────┬──────────┐
│ 框架        │ 易用性   │ 性能     │ 功能     │ 推荐场景 │
├─────────────┼──────────┼──────────┼──────────┼──────────┤
│ FastAPI     │ ⭐⭐⭐⭐⭐│ ⭐⭐     │ ⭐⭐     │ 原型/小规模│
│ vLLM        │ ⭐⭐⭐⭐ │ ⭐⭐⭐⭐⭐│ ⭐⭐⭐⭐ │ 生产推荐  │
│ TensorRT-LLM│ ⭐⭐     │ ⭐⭐⭐⭐⭐│ ⭐⭐⭐   │ 极致性能  │
│ Text Gen UI │ ⭐⭐⭐⭐⭐│ ⭐⭐⭐   │ ⭐⭐⭐⭐⭐│ 开箱即用  │
└─────────────┴──────────┴──────────┴──────────┴──────────┘
```

### ⚡ vLLM部署（推荐）

```python
# 安装
pip install vllm

# 启动服务
from vllm import LLM, SamplingParams

# 加载模型
llm = LLM(
    model="gpt2",
    tensor_parallel_size=1,  # 单GPU
    dtype="float16",
    max_model_len=2048,
)

# 推理
prompts = [
    "Once upon a time",
    "The future of AI is",
]

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)

# 性能对比
"""
HuggingFace:  100 tokens/s
vLLM:         2000+ tokens/s
加速: 20x！
"""
```

### 🌐 API服务化

```python
# 使用FastAPI + vLLM
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams

app = FastAPI()
llm = LLM(model="gpt2")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.8

@app.post("/generate")
async def generate(request: GenerateRequest):
    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )
    
    outputs = llm.generate([request.prompt], sampling_params)
    
    return {
        "text": outputs[0].outputs[0].text,
        "tokens": len(outputs[0].outputs[0].token_ids)
    }

# 启动
# uvicorn app:app --host 0.0.0.0 --port 8000

# 测试
"""
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_tokens": 50}'
"""
```

---

## 📚 2.3 生产级部署

### 🐳 Docker容器化

```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 安装Python
RUN apt-get update && apt-get install -y python3.10 python3-pip

# 安装依赖
COPY requirements.txt .
RUN pip install -r requirements.txt

# 复制代码
COPY . /app
WORKDIR /app

# 下载模型
RUN python download_model.py

# 启动服务
CMD ["python", "-m", "vllm.entrypoints.api_server", \
     "--model", "gpt2", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
```

```bash
# 构建镜像
docker build -t gpt2-service:v1 .

# 运行容器
docker run -d \
  --gpus all \
  -p 8000:8000 \
  --name gpt2-api \
  gpt2-service:v1

# 测试
curl http://localhost:8000/health
```

### ☸️ Kubernetes部署

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gpt2-deployment
spec:
  replicas: 3  # 3个副本
  selector:
    matchLabels:
      app: gpt2
  template:
    metadata:
      labels:
        app: gpt2
    spec:
      containers:
      - name: gpt2
        image: gpt2-service:v1
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1  # 每个pod一个GPU
          requests:
            memory: "8Gi"
            cpu: "4"
---
apiVersion: v1
kind: Service
metadata:
  name: gpt2-service
spec:
  selector:
    app: gpt2
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

```bash
# 部署
kubectl apply -f deployment.yaml

# 查看状态
kubectl get pods
kubectl get services

# 扩容
kubectl scale deployment gpt2-deployment --replicas=10
```

---

## 📚 2.4 监控与运维

### 📊 性能监控

```python
# 使用Prometheus + Grafana
from prometheus_client import Counter, Histogram, Gauge
import time

# 定义指标
request_count = Counter('requests_total', 'Total requests')
request_duration = Histogram('request_duration_seconds', 'Request duration')
gpu_utilization = Gauge('gpu_utilization', 'GPU utilization')
active_requests = Gauge('active_requests', 'Active requests')

@app.post("/generate")
async def generate(request: GenerateRequest):
    request_count.inc()
    active_requests.inc()
    
    start = time.time()
    try:
        # 生成
        output = llm.generate(...)
        
        duration = time.time() - start
        request_duration.observe(duration)
        
        return output
    finally:
        active_requests.dec()

# Grafana仪表板
"""
面板1: 请求QPS（每秒请求数）
面板2: P50/P95/P99延迟
面板3: GPU利用率
面板4: 活跃请求数
面板5: 错误率
"""
```

### 🚨 告警配置

```yaml
# alerting_rules.yaml
groups:
- name: gpt2_alerts
  rules:
  - alert: HighLatency
    expr: histogram_quantile(0.95, request_duration_seconds) > 1.0
    for: 5m
    annotations:
      summary: "P95 latency > 1s"
      
  - alert: HighErrorRate
    expr: rate(errors_total[5m]) > 0.05
    for: 5m
    annotations:
      summary: "Error rate > 5%"
      
  - alert: LowGPUUtilization
    expr: gpu_utilization < 0.5
    for: 10m
    annotations:
      summary: "GPU utilization < 50%"
```

---

## 📚 2.5 成本优化

### 💰 成本分析

```python
# 成本计算
def calculate_cost(
    gpu_type="A100",
    num_gpus=4,
    hours_per_month=730,
    requests_per_second=100
):
    # GPU成本
    gpu_costs = {
        "A100": 3.0,  # $/hour
        "A10G": 1.0,
        "T4": 0.35,
    }
    
    gpu_cost = gpu_costs[gpu_type] * num_gpus * hours_per_month
    
    # 请求量
    total_requests = requests_per_second * 3600 * hours_per_month
    
    # 每千次请求成本
    cost_per_1k = (gpu_cost / total_requests) * 1000
    
    print(f"""
成本分析报告：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GPU配置: {num_gpus}x {gpu_type}
月度成本: ${gpu_cost:,.2f}
月度请求: {total_requests:,.0f}
每1K请求成本: ${cost_per_1k:.4f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)

# 示例
calculate_cost(gpu_type="A10G", num_gpus=2, requests_per_second=50)
```

### 🎯 优化策略

```python
优化策略：

1. 模型优化
├── 量化到INT4 → 显存减少8x → GPU数量减半
├── 使用小模型 → GPT-2 (124M) vs GPT-2-XL (1.5B)
└── 模型蒸馏 → 保持质量，减少参数

2. 推理优化
├── 使用vLLM → 吞吐量提升20x → GPU数量减少
├── Continuous Batching → GPU利用率从50% → 90%
└── KV Cache → 延迟降低50x

3. 基础设施优化
├── Spot实例 → 成本降低70%
├── 自动扩缩容 → 根据负载调整GPU数量
└── 多区域部署 → 降低网络延迟

4. 业务优化
├── 缓存常见请求 → 减少重复计算
├── 限流 → 防止资源浪费
└── 异步处理 → 提高吞吐量

综合优化后：成本可降低 80-90%！
```

---

## 🎯 总结：端到端优化流程

```python
完整优化流程：

Step 1: 模型量化
  ├── 选择量化方法（GPTQ-4bit推荐）
  ├── 准备校准数据
  ├── 执行量化
  └── 验证质量（困惑度 < 5%下降）
  
Step 2: 推理优化
  ├── 实现KV Cache
  ├── 投机采样（加速2-4倍）
  ├── 选择推理框架（vLLM推荐）
  ├── 配置Continuous Batching
  └── 性能测试

Step 3: 服务化
  ├── API封装（FastAPI）
  ├── Docker容器化
  ├── Kubernetes部署
  └── 负载均衡

Step 4: 监控运维
  ├── 添加监控指标
  ├── 配置告警
  ├── 日志收集
  └── 性能调优

Step 5: 成本优化
  ├── 分析成本瓶颈
  ├── 应用优化策略
  ├── 持续监控
  └── 迭代改进

最终效果：
  ✅ 模型大小: 500MB → 62MB (8x压缩)
  ✅ 推理速度: 100 tokens/s → 2000+ tokens/s (20x加速)
  ✅ 投机采样: 额外2-4倍加速（可叠加）
  ✅ 成本: $10/1K requests → $0.001/1K requests (10000x降低)
  ✅ 延迟: 5s → 100ms (50x降低)
```

---

---

## 🎓 总结与检查

### ✅ 知识检查清单

完成学习后，你应该能够：

**基础概念（必须掌握）**
- [ ] 理解什么是模型量化（FP32→INT8→INT4）
- [ ] 知道PTQ和QAT的区别
- [ ] 理解KV Cache的作用
- [ ] 知道投机采样的基本原理
- [ ] 理解PagedAttention如何节省显存
- [ ] 能够选择合适的推理引擎

**进阶理解（建议掌握）**
- [ ] 理解GPTQ、AWQ等量化算法
- [ ] 知道如何实现投机采样
- [ ] 理解Continuous Batching的原理
- [ ] 能够优化推理性能
- [ ] 理解量化对精度的影响
- [ ] 知道如何权衡速度和质量

**实战能力（最终目标）**
- [ ] 能够量化模型并部署
- [ ] 会使用vLLM等推理引擎
- [ ] 能够实现投机采样加速
- [ ] 会监控和优化推理性能
- [ ] 能够解决实际部署问题
- [ ] 理解如何降低推理成本

### 📊 优化技术速查表

| 技术 | 压缩比 | 加速比 | 精度损失 | 实现难度 | 推荐场景 |
|------|--------|--------|---------|---------|---------|
| **INT8量化** | 4x | 2-3x | <1% | ⭐⭐ 中等 | 通用推荐 ⭐⭐⭐⭐⭐ |
| **INT4量化** | 8x | 3-4x | 1-3% | ⭐⭐⭐ 较难 | 显存受限 ⭐⭐⭐⭐ |
| **KV Cache** | - | 50x+ | 无 | ⭐ 简单 | 必备 ⭐⭐⭐⭐⭐ |
| **投机采样** | - | 2-4x | 无 | ⭐⭐⭐ 较难 | 长文本生成 ⭐⭐⭐⭐ |
| **PagedAttention** | 显存2x | - | 无 | ⭐⭐ 中等 | 高并发 ⭐⭐⭐⭐⭐ |
| **Continuous Batching** | - | 吞吐2-3x | 无 | ⭐⭐⭐ 较难 | 生产环境 ⭐⭐⭐⭐⭐ |

### 🎯 如何选择优化策略？

```python
# 决策树
if 目标 == "减小模型大小":
    if 精度要求高:
        使用 INT8量化  # 精度损失<1%
    else:
        使用 INT4量化  # 更小，精度损失1-3%
        
elif 目标 == "加速推理":
    必须使用 KV Cache  # 基础优化
    
    if 生成长文本:
        + 投机采样  # 额外2-4x加速
    
    if 高并发场景:
        + PagedAttention  # 节省显存
        + Continuous Batching  # 提高吞吐
        
elif 目标 == "降低成本":
    量化 + KV Cache + 投机采样  # 组合使用
    
# 推荐组合
生产环境标配:
  ✅ INT8量化（减小4倍）
  ✅ KV Cache（加速50倍）
  ✅ PagedAttention（高并发）
  ✅ Continuous Batching（高吞吐）
  ✅ vLLM推理引擎（集成以上所有）
```

### 🚀 下一步学习

现在你已经掌握了模型优化，接下来应该学习：

1. **10_production_deployment.md** - 学习如何部署到生产环境
2. **实践项目** - 部署一个优化后的模型
3. **性能调优** - 针对实际场景优化性能

### 💡 实践建议

**立即可做**：
```python
# 1. 量化你的模型
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("gpt2")
model = model.to(torch.float16)  # 先试FP16
# 观察：显存减半，速度提升

# 2. 测试KV Cache
# 不使用KV Cache
output = model.generate(input_ids, use_cache=False)
# 使用KV Cache
output = model.generate(input_ids, use_cache=True)
# 对比：速度差异巨大

# 3. 对比推理引擎
# 原生PyTorch vs vLLM
# 测量：吞吐量、延迟、显存
```

**系统实验**：
```bash
# 实验1：量化精度测试
python quantize_test.py \
  --model gpt2 \
  --precision fp32,fp16,int8,int4 \
  --eval_dataset wikitext
# 记录：perplexity变化

# 实验2：推理速度对比
python benchmark_inference.py \
  --model gpt2 \
  --batch_sizes 1,4,16,64 \
  --seq_lengths 128,512,2048
# 记录：tokens/s, latency

# 实验3：投机采样效果
python speculative_decoding_test.py \
  --target_model gpt2-large \
  --draft_model gpt2-small \
  --k_values 3,5,7,10
# 记录：加速比、接受率
```

**进阶研究**：
1. 阅读GPTQ、AWQ论文，理解量化算法
2. 研究vLLM的PagedAttention实现
3. 实现自己的投机采样
4. 优化特定场景的推理性能

---

## 📚 推荐资源

### 📖 必读文档
- [vLLM Documentation](https://docs.vllm.ai/) - 最好的推理引擎
- [TensorRT-LLM Guide](https://github.com/NVIDIA/TensorRT-LLM) - NVIDIA官方
- [Hugging Face Optimum](https://huggingface.co/docs/optimum/) - 优化工具集

### 📄 重要论文

**量化相关**：
1. **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers** (Frantar et al., 2022)
   - https://arxiv.org/abs/2210.17323
   - 4-bit量化，精度损失小

2. **AWQ: Activation-aware Weight Quantization** (Lin et al., 2023)
   - https://arxiv.org/abs/2306.00978
   - 更好的量化方法

3. **SmoothQuant: Accurate and Efficient Post-Training Quantization** (Xiao et al., 2022)
   - https://arxiv.org/abs/2211.10438
   - INT8量化

**推理优化相关**：
4. **Fast Inference from Transformers via Speculative Decoding** (Leviathan et al., 2022)
   - https://arxiv.org/abs/2211.17192
   - 投机采样原始论文

5. **Efficient Memory Management for Large Language Model Serving with PagedAttention** (Kwon et al., 2023)
   - https://arxiv.org/abs/2309.06180
   - vLLM的核心技术

6. **Medusa: Simple LLM Inference Acceleration Framework** (Cai et al., 2024)
   - https://arxiv.org/abs/2401.10774
   - 多头投机采样

### 🎥 视频教程
- [vLLM: Easy, Fast, and Cheap LLM Serving](https://www.youtube.com/watch?v=80bIUggRJf4)
- [Model Quantization Explained](https://www.youtube.com/watch?v=0VdNflU08yA)

### 🔧 实用工具

**量化工具**：
```bash
# AutoGPTQ - 最流行的量化库
pip install auto-gptq
# 使用：一行代码量化模型

# bitsandbytes - 简单易用
pip install bitsandbytes
# 使用：load_in_8bit=True

# llama.cpp - CPU推理
git clone https://github.com/ggerganov/llama.cpp
# 支持：GGUF格式，极致优化
```

**推理引擎**：
```bash
# vLLM - 推荐
pip install vllm
# 特点：PagedAttention, Continuous Batching

# TensorRT-LLM - NVIDIA官方
pip install tensorrt-llm
# 特点：最快，但配置复杂

# Text Generation Inference - HuggingFace
docker pull ghcr.io/huggingface/text-generation-inference
# 特点：开箱即用
```

**性能分析**：
```bash
# PyTorch Profiler
python -m torch.utils.bottleneck script.py

# NVIDIA Nsight
nsys profile python inference.py

# 自定义benchmark
python benchmark.py --model gpt2 --batch_size 32
```

---

## 🐛 常见问题 FAQ

### Q1: 量化会损失多少精度？
**A**: 取决于量化方法和位数。
```
FP32 → FP16:
  精度损失：几乎无（<0.1%）
  速度提升：2x
  显存节省：50%
  建议：总是使用 ✅

FP32 → INT8:
  精度损失：很小（<1%）
  速度提升：2-3x
  显存节省：75%
  建议：通用推荐 ✅

FP32 → INT4:
  精度损失：小（1-3%）
  速度提升：3-4x
  显存节省：87.5%
  建议：显存受限时使用 ⚠️

实测（GPT-2 on WikiText）:
  FP32: perplexity = 29.41
  INT8: perplexity = 29.52 (+0.4%)
  INT4: perplexity = 30.15 (+2.5%)
```

### Q2: KV Cache为什么这么重要？
**A**: 因为它避免了重复计算。
```python
# 不使用KV Cache（每次都重新计算）
生成100个token:
  Token 1: 计算1个token的attention
  Token 2: 计算2个token的attention（重复计算token 1）
  Token 3: 计算3个token的attention（重复计算token 1,2）
  ...
  Token 100: 计算100个token的attention
  
  总计算量: 1+2+3+...+100 = 5050次attention

# 使用KV Cache（缓存之前的K,V）
生成100个token:
  Token 1: 计算1次，缓存K1,V1
  Token 2: 只计算新的，使用缓存的K1,V1
  Token 3: 只计算新的，使用缓存的K1,V1,K2,V2
  ...
  
  总计算量: 100次attention
  
加速比: 5050/100 = 50.5x ！

结论：KV Cache是必须的，没有它推理会慢50倍！
```

### Q3: 投机采样真的不损失质量吗？
**A**: 是的，完全无损！
```python
# 原理：用小模型"猜测"，大模型"验证"

传统生成:
  大模型生成token 1
  大模型生成token 2
  大模型生成token 3
  ...
  
投机采样:
  小模型快速生成: token 1,2,3,4,5
  大模型一次验证: ✅✅✅❌  (前3个对，第4个错)
  保留: token 1,2,3
  大模型生成: token 4（正确的）
  
关键：
  - 最终输出完全由大模型决定
  - 小模型只是"建议"，不影响结果
  - 质量 = 100%大模型质量
  - 速度 = 2-4x（因为小模型很快）

实测：
  原始: "The cat sat on the mat"
  投机: "The cat sat on the mat"
  完全相同！✅
```

### Q4: 如何选择推理引擎？
**A**: 根据需求选择。
```
vLLM（推荐）:
  ✅ 最高吞吐量
  ✅ PagedAttention节省显存
  ✅ Continuous Batching
  ✅ 易于使用
  ❌ 只支持CUDA
  适合：生产环境、高并发

TensorRT-LLM（最快）:
  ✅ 最低延迟
  ✅ NVIDIA官方优化
  ✅ 支持所有NVIDIA GPU
  ❌ 配置复杂
  ❌ 只支持NVIDIA
  适合：追求极致性能

Text Generation Inference（简单）:
  ✅ 开箱即用
  ✅ HuggingFace集成
  ✅ Docker部署
  ❌ 性能不如vLLM
  适合：快速原型

llama.cpp（CPU）:
  ✅ CPU推理
  ✅ 极致优化
  ✅ 跨平台
  ❌ 速度较慢
  适合：没有GPU的场景

推荐：
  - 有GPU：vLLM ⭐⭐⭐⭐⭐
  - 追求极致：TensorRT-LLM ⭐⭐⭐⭐
  - 快速开始：TGI ⭐⭐⭐
  - 只有CPU：llama.cpp ⭐⭐⭐
```

### Q5: PagedAttention如何节省显存？
**A**: 类似操作系统的虚拟内存。
```python
# 传统方法（预分配）
每个请求预留最大长度的显存:
  请求1: 实际50 tokens，预留2048 tokens → 浪费97.5%
  请求2: 实际100 tokens，预留2048 tokens → 浪费95%
  ...
  
  总显存: N个请求 × 2048 × 模型大小
  利用率: 很低（<10%）

# PagedAttention（按需分配）
按实际需要分配，类似操作系统的页表:
  请求1: 实际50 tokens → 只分配50 tokens
  请求2: 实际100 tokens → 只分配100 tokens
  ...
  
  总显存: 实际使用量
  利用率: 很高（>80%）

效果：
  - 相同显存可以处理2-3x请求
  - 或者处理更长的序列
  - 几乎无性能损失

类比：
  传统 = 每人一间大房子（很多空间浪费）
  PagedAttention = 按需分配房间（高效利用）
```

### Q6: Continuous Batching是什么？
**A**: 动态批处理，提高吞吐量。
```python
# 传统Static Batching
等待凑够batch_size才开始:
  请求1到达 → 等待
  请求2到达 → 等待
  请求3到达 → 等待
  请求4到达 → 开始处理（batch_size=4）
  
  问题：
  - 请求1等待时间长
  - GPU可能空闲
  - 吞吐量低

# Continuous Batching
动态加入和移除请求:
  请求1到达 → 立即开始
  请求2到达 → 加入batch
  请求1完成 → 移除，请求3加入
  ...
  
  优点：
  - 延迟低（立即处理）
  - GPU利用率高
  - 吞吐量高2-3x

实测：
  Static: 100 req/s, 平均延迟500ms
  Continuous: 250 req/s, 平均延迟200ms
```

### Q7: 如何验证量化后的模型质量？
**A**: 多维度评估。
```python
# 1. Perplexity测试
from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

fp32_ppl = evaluate_perplexity(fp32_model, dataset)
int8_ppl = evaluate_perplexity(int8_model, dataset)

print(f"FP32: {fp32_ppl:.2f}")
print(f"INT8: {int8_ppl:.2f}")
print(f"差异: {(int8_ppl/fp32_ppl - 1)*100:.1f}%")
# 应该 < 2%

# 2. 下游任务测试
tasks = ["hellaswag", "winogrande", "arc"]
for task in tasks:
    fp32_acc = evaluate(fp32_model, task)
    int8_acc = evaluate(int8_model, task)
    print(f"{task}: {fp32_acc:.1f}% → {int8_acc:.1f}%")

# 3. 生成质量测试
prompts = ["Once upon a time", "The capital of France"]
for prompt in prompts:
    fp32_output = fp32_model.generate(prompt)
    int8_output = int8_model.generate(prompt)
    # 人工对比质量

# 4. 速度和显存测试
benchmark(fp32_model)  # 100 tokens/s, 16GB
benchmark(int8_model)  # 250 tokens/s, 4GB
```

### Q8: 投机采样的draft model如何选择？
**A**: 遵循这些原则。
```
原则1：架构相同
  Target: GPT-2 Large
  Draft: GPT-2 Small ✅
  Draft: BERT ❌（架构不同）

原则2：大小比例
  Target: 1.5B参数
  Draft: 125M-350M参数（1/5 - 1/10）
  Draft: 10M参数 ❌（太小，接受率低）
  Draft: 1B参数 ❌（太大，加速不明显）

原则3：训练数据相似
  Target: 训练在代码上
  Draft: 也训练在代码上 ✅
  Draft: 训练在通用文本 ⚠️（接受率可能低）

实际例子：
  Target: Llama-2-70B
  Draft: Llama-2-7B ✅（10x小）
  
  Target: GPT-3.5
  Draft: GPT-2 ✅（架构相同）

效果：
  - 好的draft: 接受率70-90%，加速3-4x
  - 差的draft: 接受率30-50%，加速1.5-2x
```

### Q9: 如何优化推理成本？
**A**: 多管齐下。
```python
# 成本 = 硬件成本 + 运营成本

# 1. 减小模型（最有效）
量化到INT8: 成本减少75%
量化到INT4: 成本减少87.5%

# 2. 提高吞吐量
使用vLLM: 吞吐量提升2-3x
→ 相同请求量，需要的GPU减少2-3x

# 3. 降低延迟要求
如果可以接受200ms而不是50ms:
  - 可以用更小的GPU
  - 可以增大batch_size
  - 成本降低50%+

# 4. 使用Spot实例
AWS Spot: 成本降低70%
但需要处理中断

# 5. 批处理非实时请求
实时请求: 必须立即处理
离线请求: 可以批处理
→ 离线请求成本降低80%

实际案例：
  原始: A100 × 8, $20/小时, 100 req/s
  优化后: A100 × 2 (INT8+vLLM), $5/小时, 100 req/s
  成本降低: 75% ✅
```

### Q10: 如何调试推理性能问题？
**A**: 系统性分析。
```python
# 1. 测量各部分耗时
import time

# Tokenization
t0 = time.time()
tokens = tokenizer.encode(text)
print(f"Tokenization: {time.time()-t0:.3f}s")

# Model inference
t0 = time.time()
output = model.generate(tokens)
print(f"Generation: {time.time()-t0:.3f}s")

# Decoding
t0 = time.time()
text = tokenizer.decode(output)
print(f"Decoding: {time.time()-t0:.3f}s")

# 2. 分析瓶颈
如果tokenization慢:
  - 使用fast tokenizer
  - 预处理并缓存

如果generation慢:
  - 检查是否使用KV Cache
  - 检查batch_size是否太小
  - 考虑量化

如果显存不够:
  - 使用INT8/INT4
  - 减小batch_size
  - 使用PagedAttention

# 3. 使用profiler
from torch.profiler import profile
with profile() as prof:
    model.generate(tokens)
print(prof.key_averages().table())
# 找出最耗时的操作

# 4. 对比baseline
baseline_speed = 100  # tokens/s
current_speed = measure_speed()
print(f"相对baseline: {current_speed/baseline_speed:.1f}x")
```

---

**恭喜你完成第09章！** 🎉

你现在已经掌握了模型优化的核心技术。从量化到推理加速，从KV Cache到投机采样，你已经具备了部署高性能模型的能力。

**准备好了吗？让我们继续前进！** → [10_production_deployment.md](10_production_deployment.md)

