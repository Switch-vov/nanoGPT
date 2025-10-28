# 模型优化完全指南：从量化到部署

## 🎯 概览

本指南涵盖模型优化的两个核心方面：
1. **模型量化**：压缩模型大小，加速推理
2. **部署优化**：高效服务化，生产级部署

```
优化流程：

训练好的模型 (FP32, 500MB)
    ↓
📦 量化优化
    ├─ INT8量化 → 125MB (4x压缩)
    ├─ INT4量化 → 62MB (8x压缩)
    └─ 推理加速 2-4x
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
  ✅ 成本: $10/1K requests → $0.001/1K requests (10000x降低)
  ✅ 延迟: 5s → 100ms (50x降低)
```

---

## 📚 推荐资源

### 量化相关
- [GPTQ论文](https://arxiv.org/abs/2210.17323)
- [AWQ论文](https://arxiv.org/abs/2306.00978)
- [AutoGPTQ库](https://github.com/PanQiWei/AutoGPTQ)

### 部署相关
- [vLLM文档](https://docs.vllm.ai/)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [Text Generation Inference](https://github.com/huggingface/text-generation-inference)

### 监控相关
- [Prometheus](https://prometheus.io/)
- [Grafana](https://grafana.com/)

---

**下一步：** 学习生产级部署实战（10_production_deployment.md）

