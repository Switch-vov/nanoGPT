# 模型量化完全指南

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

## 📚 第一部分：量化基础

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
             = [-84, -33, 0, 40, 127]

# 步骤4: 反量化（推理时）
weights_restored = weights_int8 * scale
                 = [-2.51, -0.99, 0.0, 1.20, 3.80]

# 误差
error = weights_restored - weights_fp32
      = [-0.01, 0.01, 0.0, 0.0, 0.0]
```

**可视化：**

```
FP32精度（连续）:
  |-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
  -3  -2  -1   0   1   2   3   4   5

INT8精度（离散）:
  |------|------|------|------|------|
  -3     -2     -1      0      1     2    ...
  
精度下降，但可以接受！
```

---

## 🎨 第二部分：量化方法分类

### 1️⃣ **按量化时机分类**

```python
训练后量化 (Post-Training Quantization, PTQ):
  训练（FP32）→ 模型 → 量化 → 部署
  
  优势:
  ✅ 简单，不需要重新训练
  ✅ 几分钟完成
  ✅ 不需要训练数据
  
  劣势:
  ❌ 精度损失较大（尤其INT4）
  ❌ 对小模型不友好
  
  适用: 大模型（>1B参数），INT8量化

量化感知训练 (Quantization-Aware Training, QAT):
  训练时模拟量化 → 模型 → 部署
  
  优势:
  ✅ 精度损失小
  ✅ 可以用更激进的量化（INT4）
  
  劣势:
  ❌ 需要重新训练
  ❌ 需要训练数据
  ❌ 时间长
  
  适用: 小模型，INT4/INT2量化，精度要求高
```

### 2️⃣ **按量化粒度分类**

```python
Per-Tensor量化:
  整个张量用同一个scale
  
  scale = max(|tensor|) / 127
  
  优势: 简单，快速
  劣势: 精度一般

Per-Channel量化:
  每个输出通道用不同scale
  
  对于权重 W: [out_channels, in_channels]
  scale[i] = max(|W[i, :]|) / 127
  
  优势: 精度更好
  劣势: 稍复杂

Per-Group量化:
  每128个元素一组
  
  优势: 平衡精度和复杂度
  应用: GPTQ, AWQ
```

### 3️⃣ **按量化对象分类**

```python
仅权重量化 (Weight-only):
  权重: INT8/INT4
  激活值: FP16
  
  优势:
  ✅ 减小模型大小
  ✅ 减少内存带宽
  ✅ 精度损失小
  
  适用: 大模型推理（内存带宽瓶颈）

权重+激活量化:
  权重: INT8
  激活值: INT8
  
  优势:
  ✅ 可以用INT8硬件加速
  ✅ 推理更快
  
  劣势:
  ❌ 精度损失较大
  
  适用: 需要极致速度的场景
```

---

## 🔧 第三部分：PyTorch原生量化

### 📝 动态量化（最简单）

**特点：**
- 权重预先量化
- 激活值运行时动态量化
- 适合RNN、Transformer

**实现代码：**

```python
import torch
import torch.quantization as quant
from model import GPT, GPTConfig

# 1. 加载模型
config = GPTConfig()
model = GPT(config)
model.load_state_dict(torch.load('model.pt'))
model.eval()

# 2. 动态量化
model_quantized = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # 量化Linear层
    dtype=torch.qint8    # INT8量化
)

# 3. 保存量化模型
torch.save(model_quantized.state_dict(), 'model_quantized.pt')

# 4. 比较大小
import os
original_size = os.path.getsize('model.pt') / 1e6
quantized_size = os.path.getsize('model_quantized.pt') / 1e6

print(f"原始模型: {original_size:.2f} MB")
print(f"量化模型: {quantized_size:.2f} MB")
print(f"压缩比: {original_size / quantized_size:.2f}x")

# 输出示例:
# 原始模型: 497.35 MB
# 量化模型: 126.54 MB
# 压缩比: 3.93x
```

**性能测试：**

```python
# 测试推理速度和精度

import time
import torch

# 准备输入
input_ids = torch.randint(0, 50257, (1, 256))

# 原始模型
model.eval()
with torch.no_grad():
    start = time.time()
    for _ in range(100):
        output_fp32, _ = model(input_ids)
    time_fp32 = (time.time() - start) / 100

# 量化模型
model_quantized.eval()
with torch.no_grad():
    start = time.time()
    for _ in range(100):
        output_int8, _ = model_quantized(input_ids)
    time_int8 = (time.time() - start) / 100

# 比较
print(f"\n推理时间:")
print(f"FP32: {time_fp32*1000:.2f} ms")
print(f"INT8: {time_int8*1000:.2f} ms")
print(f"加速比: {time_fp32/time_int8:.2f}x")

# 精度对比
diff = torch.abs(output_fp32 - output_int8).mean()
print(f"\n输出差异: {diff:.6f}")

# 典型输出:
# 推理时间:
# FP32: 45.23 ms
# INT8: 28.67 ms
# 加速比: 1.58x
#
# 输出差异: 0.000234
```

---

## 🚀 第四部分：高级量化方法

### 1️⃣ **GPTQ (最流行)**

**核心思想：**
- 逐层量化
- 最小化量化误差
- 支持INT4

**安装：**

```bash
pip install auto-gptq
```

**使用代码：**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# 1. 准备校准数据
# 需要一些代表性文本用于校准
calibration_data = [
    "The quick brown fox jumps over the lazy dog.",
    "To be or not to be, that is the question.",
    # ... 更多样本（通常128-512个）
]

# 2. 量化配置
quantize_config = BaseQuantizeConfig(
    bits=4,              # INT4量化
    group_size=128,      # 每128个元素一组
    desc_act=False,      # 是否量化激活值
)

# 3. 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    torch_dtype=torch.float16,
    device_map="auto"
)

# 4. 量化
model_gptq = AutoGPTQForCausalLM.from_pretrained(
    model,
    quantize_config,
    calibration_data=calibration_data
)

# 5. 保存
model_gptq.save_quantized("./gpt2-gptq-4bit")

# 6. 加载量化模型
model_loaded = AutoGPTQForCausalLM.from_quantized(
    "./gpt2-gptq-4bit",
    device="cuda:0"
)
```

**效果：**

```python
模型大小对比 (LLaMA-7B):
  FP16: 13.5 GB
  INT8: 7 GB (1.93x压缩)
  INT4 (GPTQ): 3.9 GB (3.46x压缩)

精度对比 (Perplexity):
  FP16: 5.68
  INT8: 5.72 (+0.7%)
  INT4 (GPTQ): 5.85 (+3.0%)

推理速度 (tokens/sec):
  FP16: 28
  INT8: 45 (1.6x)
  INT4 (GPTQ): 62 (2.2x)
```

---

### 2️⃣ **AWQ (Activation-aware Weight Quantization)**

**核心思想：**
- 关注重要的权重通道
- 基于激活值大小保护重要通道
- INT4，精度更好

**安装：**

```bash
pip install autoawq
```

**使用代码：**

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# 1. 加载模型和tokenizer
model_path = "meta-llama/Llama-2-7b-hf"
quant_path = "llama-2-7b-awq"

model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 2. 量化配置
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

# 3. 量化（需要校准数据）
model.quantize(tokenizer, quant_config=quant_config)

# 4. 保存
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

# 5. 加载和使用
model_awq = AutoAWQForCausalLM.from_quantized(
    quant_path,
    fuse_layers=True  # 融合层以提速
)

# 推理
inputs = tokenizer("Hello, my name is", return_tensors="pt").to("cuda")
outputs = model_awq.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

**AWQ vs GPTQ：**

```python
对比 (LLaMA-7B, INT4):

指标          | GPTQ    | AWQ     | 胜者
──────────────┼─────────┼─────────┼─────
Perplexity    | 5.85    | 5.75    | AWQ ✅
推理速度      | 62 t/s  | 71 t/s  | AWQ ✅
量化时间      | 4小时   | 30分钟  | AWQ ✅
模型大小      | 3.9 GB  | 4.1 GB  | GPTQ ✅

推荐: 
  精度优先 → AWQ
  速度均衡 → AWQ
  极致压缩 → GPTQ
```

---

### 3️⃣ **bitsandbytes (最简单的INT8/INT4)**

**特点：**
- 一行代码量化
- 支持8bit和4bit
- 与HuggingFace完美集成

**安装：**

```bash
pip install bitsandbytes
```

**使用代码：**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# INT8量化
model_int8 = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    device_map="auto",
    load_in_8bit=True  # 一行搞定！
)

# INT4量化
model_int4 = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map="auto",
    load_in_4bit=True,  # INT4
    bnb_4bit_compute_dtype=torch.float16,  # 计算使用FP16
    bnb_4bit_quant_type="nf4",  # NF4量化类型
    bnb_4bit_use_double_quant=True,  # 双重量化（更小）
)

# 直接使用！
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
inputs = tokenizer("Hello world", return_tensors="pt").to("cuda")
outputs = model_int4.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

**显存占用对比：**

```python
LLaMA-7B:
  FP32: 28 GB  (不可能放入单GPU)
  FP16: 14 GB
  INT8: 7 GB
  INT4 (bitsandbytes): 3.5 GB  ✅ RTX 3090可以跑！
  INT4 + 双重量化: 2.8 GB  ✅✅ 更小！
```

---

## 🎯 第五部分：量化NanoGPT实战

### 📝 完整流程

**步骤1: 训练模型（已完成）**

```bash
python train.py config/train_shakespeare_char.py
# 得到: out/ckpt.pt
```

**步骤2: 创建量化脚本**

```python
# quantize_model.py

import torch
import torch.quantization as quant
from model import GPT, GPTConfig
import os

def quantize_gpt(model_path, output_path, quantization_type='dynamic'):
    """
    量化GPT模型
    
    Args:
        model_path: 原始模型路径
        output_path: 量化后模型保存路径
        quantization_type: 'dynamic' 或 'static'
    """
    print(f"加载模型: {model_path}")
    
    # 加载checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 创建模型
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    
    # 加载权重
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    
    model.eval()
    
    # 量化
    print(f"开始{quantization_type}量化...")
    
    if quantization_type == 'dynamic':
        # 动态量化
        model_quantized = quant.quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.Embedding},  # 量化这些层
            dtype=torch.qint8
        )
    else:
        raise NotImplementedError("目前只支持动态量化")
    
    # 保存量化模型
    print(f"保存量化模型到: {output_path}")
    torch.save({
        'model': model_quantized.state_dict(),
        'model_args': checkpoint['model_args'],
        'quantized': True,
    }, output_path)
    
    # 对比大小
    original_size = os.path.getsize(model_path) / (1024 * 1024)
    quantized_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"\n结果:")
    print(f"原始模型: {original_size:.2f} MB")
    print(f"量化模型: {quantized_size:.2f} MB")
    print(f"压缩比: {original_size / quantized_size:.2f}x")
    
    return model_quantized

def test_quantized_model(model_path, quantized_model_path, meta_path='data/shakespeare_char/meta.pkl'):
    """测试量化模型的性能和精度"""
    import pickle
    import time
    
    # 加载meta
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    # 加载原始模型
    print("\n加载原始模型...")
    checkpoint = torch.load(model_path, map_location='cpu')
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # 加载量化模型
    print("加载量化模型...")
    checkpoint_q = torch.load(quantized_model_path, map_location='cpu')
    model_q = GPT(gptconf)
    model_q.load_state_dict(checkpoint_q['model'])
    model_q.eval()
    
    # 测试输入
    prompt = "ROMEO:"
    input_ids = torch.tensor([encode(prompt)], dtype=torch.long)
    
    # 生成（原始模型）
    print("\n原始模型生成:")
    with torch.no_grad():
        start = time.time()
        y_orig = model.generate(input_ids, max_new_tokens=100, temperature=0.8, top_k=200)
        time_orig = time.time() - start
    text_orig = decode(y_orig[0].tolist())
    print(text_orig[:200])
    print(f"时间: {time_orig:.2f}s")
    
    # 生成（量化模型）
    print("\n量化模型生成:")
    with torch.no_grad():
        start = time.time()
        y_quant = model_q.generate(input_ids, max_new_tokens=100, temperature=0.8, top_k=200)
        time_quant = time.time() - start
    text_quant = decode(y_quant[0].tolist())
    print(text_quant[:200])
    print(f"时间: {time_quant:.2f}s")
    
    print(f"\n加速比: {time_orig / time_quant:.2f}x")

if __name__ == "__main__":
    model_path = "out/ckpt.pt"
    output_path = "out/ckpt_quantized.pt"
    
    # 量化
    quantize_gpt(model_path, output_path)
    
    # 测试
    test_quantized_model(model_path, output_path)
```

**步骤3: 运行量化**

```bash
python quantize_model.py

# 输出示例:
# 加载模型: out/ckpt.pt
# 开始dynamic量化...
# 保存量化模型到: out/ckpt_quantized.pt
#
# 结果:
# 原始模型: 39.47 MB
# 量化模型: 11.23 MB
# 压缩比: 3.52x
#
# 原始模型生成:
# ROMEO:
# Why, then, I thank you all; I thank you, heartily;
# That you have ta'en a tardy sluggard here.
# ...
# 时间: 4.56s
#
# 量化模型生成:
# ROMEO:
# Why, then, I thank you all; I thank you, heartily;
# That you have ta'en a tardy sluggard here.
# ...
# 时间: 3.12s
#
# 加速比: 1.46x
```

---

## 📊 第六部分：量化方法对比

### 🆚 完整对比表

```python
┌─────────────┬────────┬────────┬──────────┬──────────┬──────────┐
│ 方法         │ 压缩比 │ 精度   │ 推理速度 │ 易用性   │ 推荐场景 │
├─────────────┼────────┼────────┼──────────┼──────────┼──────────┤
│ FP16        │ 2x     │ 100%   │ 1.2x     │ ⭐⭐⭐⭐⭐│ 默认选择 │
│ INT8(动态)  │ 4x     │ 99%    │ 1.5x     │ ⭐⭐⭐⭐⭐│ 简单快速 │
│ INT8(静态)  │ 4x     │ 99%    │ 2.0x     │ ⭐⭐⭐   │ 需要校准 │
│ GPTQ(INT4)  │ 8x     │ 95-97% │ 2.5x     │ ⭐⭐⭐   │ 大模型   │
│ AWQ(INT4)   │ 7x     │ 97-98% │ 2.8x     │ ⭐⭐⭐   │ 精度要求高│
│ bitsandbytes│ 8x     │ 96%    │ 2.2x     │ ⭐⭐⭐⭐⭐│ 最简单   │
└─────────────┴────────┴────────┴──────────┴──────────┴──────────┘

选择建议:
  快速开始: FP16 或 bitsandbytes
  生产环境: GPTQ 或 AWQ
  研究实验: PyTorch原生量化
  移动端: ONNX + INT8
```

---

## 💡 第七部分：最佳实践

### ✅ 量化流程

```python
标准流程:

1. 训练模型（FP32/FP16）
   ↓
2. 评估基线性能
   - Perplexity
   - 生成质量
   - 推理速度
   ↓
3. 选择量化方法
   - 模型大小: >7B → GPTQ/AWQ
   - 模型大小: <1B → PyTorch动态量化
   - 易用性优先 → bitsandbytes
   ↓
4. 量化
   - 准备校准数据（PTQ）
   - 运行量化脚本
   ↓
5. 验证
   - 精度损失 < 3% ✅
   - 精度损失 > 5% ❌ 尝试其他方法
   ↓
6. 部署
```

### 🎯 性能调优

```python
技巧1: 混合精度量化
  关键层: FP16 (如第一层和最后一层)
  其他层: INT8/INT4
  
  效果: 精度提升，压缩比略降

技巧2: 敏感度分析
  测试每一层的量化影响
  保护敏感层
  
  工具: torch.quantization.observer

技巧3: 校准数据选择
  代表性强的数据
  覆盖不同领域
  数量: 128-512个样本
  
  质量 > 数量

技巧4: 量化+剪枝
  先剪枝（去除不重要的连接）
  再量化
  
  效果: 更高的压缩比

技巧5: 知识蒸馏+量化
  大模型（教师）→ 小模型（学生）
  小模型 → 量化
  
  效果: 小而精
```

---

## 🐛 第八部分：常见问题

### ❓ 问题1: 量化后精度大幅下降

```python
原因:
  - 模型太小（<100M）
  - 量化方法太激进（INT4）
  - 校准数据不好

解决:
  1. 使用QAT而不是PTQ
  2. 从INT8开始，不要直接INT4
  3. 改进校准数据
  4. 尝试混合精度
  5. 使用AWQ（精度更好）
```

### ❓ 问题2: 量化后反而变慢

```python
原因:
  - CPU上量化INT8可能更慢
  - 没有硬件加速
  - Python开销

解决:
  1. 使用GPU推理
  2. 使用ONNX Runtime
  3. 使用TensorRT
  4. 批量推理而不是单个
```

### ❓ 问题3: 量化模型无法加载

```python
原因:
  - PyTorch版本不匹配
  - 量化后端不一致

解决:
  1. 确保相同PyTorch版本
  2. 保存完整checkpoint而不只是state_dict
  3. 记录量化配置
```

### ❓ 问题4: 显存占用没有减少

```python
原因:
  - 只量化了权重，激活值还是FP32
  - 中间结果未量化

解决:
  1. 使用静态量化（量化激活值）
  2. 减小batch size
  3. 使用gradient checkpointing
```

---

## 🚀 第九部分：高级话题

### 1️⃣ **混合精度量化**

```python
from transformers import AutoModelForCausalLM
import torch

# 自定义量化策略
class MixedPrecisionConfig:
    def __init__(self):
        # 关键层保持FP16
        self.fp16_layers = [
            'transformer.wte',  # Token embedding
            'transformer.h.0',  # 第一层
            'transformer.h.11',  # 最后一层
            'lm_head'           # 输出层
        ]
        
        # 其他层INT8
        self.int8_layers = [
            f'transformer.h.{i}' for i in range(1, 11)
        ]

# 应用混合精度
def apply_mixed_precision(model, config):
    for name, module in model.named_modules():
        if any(name.startswith(fp_layer) for fp_layer in config.fp16_layers):
            # 保持FP16
            module.to(torch.float16)
        elif any(name.startswith(int8_layer) for int8_layer in config.int8_layers):
            # 量化到INT8
            module = torch.quantization.quantize_dynamic(
                module, {torch.nn.Linear}, dtype=torch.qint8
            )
    return model
```

### 2️⃣ **自动搜索最佳量化配置**

```python
import itertools
from tqdm import tqdm

def find_best_quantization(model, test_data, precision_threshold=0.03):
    """
    自动搜索最佳量化配置
    
    Args:
        model: 原始模型
        test_data: 测试数据
        precision_threshold: 可接受的精度损失（3%）
    """
    # 配置空间
    configs = {
        'bits': [8, 4],
        'group_size': [64, 128, 256],
        'symmetric': [True, False],
    }
    
    # 生成所有组合
    config_list = [dict(zip(configs.keys(), v)) 
                   for v in itertools.product(*configs.values())]
    
    best_config = None
    best_compression = 0
    baseline_loss = evaluate(model, test_data)
    
    for config in tqdm(config_list, desc="搜索最佳配置"):
        # 量化
        model_q = quantize_with_config(model, config)
        
        # 评估
        loss_q = evaluate(model_q, test_data)
        loss_increase = (loss_q - baseline_loss) / baseline_loss
        
        # 检查是否满足精度要求
        if loss_increase < precision_threshold:
            compression = calculate_compression(config)
            if compression > best_compression:
                best_compression = compression
                best_config = config
    
    return best_config
```

### 3️⃣ **量化+剪枝组合**

```python
# 先剪枝
import torch.nn.utils.prune as prune

def prune_model(model, amount=0.3):
    """剪枝模型（移除30%的权重）"""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')  # 使剪枝永久
    return model

# 再量化
model = prune_model(model, amount=0.3)
model_quantized = quantize(model)

# 效果
原始: 100 MB, 100% 精度
剪枝: 70 MB, 98% 精度
剪枝+量化: 17.5 MB (5.7x压缩), 96% 精度
```

---

## 🎓 总结

### ✨ 核心要点

```python
1. 量化 = 用更少比特表示权重
   FP32 → INT8: 4x压缩
   FP32 → INT4: 8x压缩

2. 选择量化方法
   简单: bitsandbytes (一行代码)
   精度: AWQ
   压缩: GPTQ
   通用: PyTorch动态量化

3. 量化流程
   训练 → 评估 → 量化 → 验证 → 部署

4. 精度权衡
   INT8: <1% 精度损失 ✅
   INT4: 3-5% 精度损失
   INT4 (AWQ): 2-3% 精度损失 ✅

5. 最佳实践
   - 先FP16，再INT8，最后INT4
   - 使用代表性校准数据
   - 保护关键层
   - 验证精度损失
```

### 🎯 推荐方案

```python
你的场景 → 推荐方案

研究/实验:
  → PyTorch动态量化
  → 简单、灵活

生产部署 (云端):
  → AWQ INT4
  → 精度好、速度快

边缘设备:
  → GPTQ INT4 + ONNX
  → 极致压缩

快速原型:
  → bitsandbytes
  → 一行代码搞定

模型<1B:
  → QAT INT8
  → 精度损失小
```

---

**记住：**

> 量化不是银弹，而是工程权衡。
> 理解精度、速度、模型大小的三角关系，
> 选择最适合你场景的方案。
>
> 从简单开始（FP16），逐步激进（INT4），
> 始终验证精度是否可接受。

🎉 恭喜你掌握了模型量化！接下来学习模型部署。
