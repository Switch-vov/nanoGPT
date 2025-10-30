# 第10章：生产级部署实战指南

> **学习目标**: 掌握从训练到部署的完整工程流程  
> **难度等级**: 🌿🌿🌿🌿 高级（工程实战）  
> **预计时间**: 6-8小时  
> **前置知识**: 01-09章全部内容

## 🎯 你将学到什么

学完本章，你将能够：
- ✅ 掌握端到端的部署流程
- ✅ 能够构建高性能API服务
- ✅ 理解Docker和Kubernetes部署
- ✅ 掌握监控和日志系统
- ✅ 能够进行性能优化和故障排查
- ✅ 理解生产环境的最佳实践

## 💭 开始之前：为什么要学这个？

**场景**：训练出模型只是第一步，部署到生产才是真正的挑战。

**比喻**：就像开餐厅：
- 🍳 做出好菜：训练好模型
- 🏪 高效服务：部署和优化
- 💰 赚钱：真正创造价值

**学完之后**：
- ✅ 能独立完成模型部署
- ✅ 理解生产环境的挑战
- ✅ 会优化服务性能
- ✅ 能处理实际问题

---

## 🎯 项目目标

构建一个**生产级的代码补全助手**，从训练到部署的完整流程。

```
项目流程：

数据准备 → 模型训练 → 分布式加速 → 模型优化 → API服务 → 容器化 → 监控运维

预期成果：
  ✅ 训练一个代码补全模型
  ✅ 部署为高性能API服务
  ✅ 支持100+并发用户
  ✅ 延迟 < 200ms
  ✅ 成本 < $0.01/1K tokens
```

---

## 📚 阶段1：数据准备

### 🔧 收集Python代码数据

```python
# collect_code.py
import os
import tiktoken
import numpy as np

def collect_python_files(root_dir):
    """收集所有Python文件"""
    code_files = []
    for root, dirs, files in os.walk(root_dir):
        # 跳过虚拟环境和缓存
        dirs[:] = [d for d in dirs if d not in ['.venv', '__pycache__', 'node_modules']]
        
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        code = f.read()
                        if len(code) > 100:  # 过滤太短的文件
                            code_files.append(code)
                except:
                    continue
    
    return code_files

def prepare_dataset(code_files, output_dir='data/python_code'):
    """准备训练数据"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 合并所有代码
    data = '\n\n# ================\n\n'.join(code_files)
    print(f"总字符数: {len(data):,}")
    
    # 2. 分割训练/验证集
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]
    
    # 3. Tokenize
    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    
    print(f"训练tokens: {len(train_ids):,}")
    print(f"验证tokens: {len(val_ids):,}")
    
    # 4. 保存为二进制文件
    np.array(train_ids, dtype=np.uint16).tofile(f'{output_dir}/train.bin')
    np.array(val_ids, dtype=np.uint16).tofile(f'{output_dir}/val.bin')
    
    print(f"数据已保存到 {output_dir}/")

# 运行
if __name__ == '__main__':
    # 收集代码（替换为你的项目路径）
    code_files = collect_python_files('/path/to/your/python/projects')
    print(f"收集了 {len(code_files)} 个Python文件")
    
    # 准备数据集
    prepare_dataset(code_files)
```

**运行：**
```bash
python collect_code.py

# 输出：
# 收集了 1,234 个Python文件
# 总字符数: 5,678,901
# 训练tokens: 1,234,567
# 验证tokens: 137,174
# 数据已保存到 data/python_code/
```

---

## 📚 阶段2：模型训练

### 🔧 创建训练配置

```python
# config/train_code_assistant.py

import time

# 输出目录
out_dir = 'out-code-assistant'
eval_interval = 500
eval_iters = 100
log_interval = 10

# 数据集
dataset = 'python_code'
gradient_accumulation_steps = 4
batch_size = 8
block_size = 512  # 代码需要长上下文

# 模型（中等大小）
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1

# 微调设置
init_from = 'gpt2'  # 从GPT-2开始
learning_rate = 5e-5  # 小学习率
max_iters = 5000
lr_decay_iters = 5000
min_lr = 5e-6

# 优化
weight_decay = 0.1
grad_clip = 1.0
decay_lr = True
warmup_iters = 100

# 系统
device = 'cuda'
dtype = 'float16'
compile = True
```

### 🚀 开始训练

```bash
# 单GPU训练
python train.py config/train_code_assistant.py

# 预期输出：
# iter 0: loss 3.2145, time 1234.56ms
# iter 500: train loss 2.1234, val loss 2.3456
# iter 1000: train loss 1.8765, val loss 2.1234
# ...
# iter 5000: train loss 1.2345, val loss 1.5678
# 训练完成！最佳val loss: 1.5234
```

---

## 📚 阶段3：分布式加速

### 🔧 多GPU训练配置

```python
# config/train_code_assistant_ddp.py

# 继承单GPU配置
exec(open('config/train_code_assistant.py').read())

# DDP优化
batch_size = 16  # 每个GPU更大batch
gradient_accumulation_steps = 2  # 减少累积步数

# 总batch_size = 16 × 2 × 4 = 128 (和单GPU的8×4×4一样)
```

### 🚀 启动分布式训练

```bash
# 4 GPU训练
torchrun --standalone --nproc_per_node=4 \
  train.py config/train_code_assistant_ddp.py

# 加速效果：
# 单GPU: 2-3小时
# 4 GPU: 30-40分钟 (约4x加速)
```

---

## 📚 阶段4：模型优化

### 🔧 量化模型

```python
# quantize_model.py

import torch
from model import GPT, GPTConfig

def quantize_model(checkpoint_path, output_path):
    """量化模型到INT8"""
    # 1. 加载模型
    checkpoint = torch.load(checkpoint_path)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    print(f"原始模型参数: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 2. 动态量化
    model_quantized = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    
    # 3. 保存
    torch.save({
        'model': model_quantized.state_dict(),
        'model_args': checkpoint['model_args'],
        'quantized': True,
    }, output_path)
    
    # 4. 对比
    import os
    orig_size = os.path.getsize(checkpoint_path) / 1e6
    quant_size = os.path.getsize(output_path) / 1e6
    
    print(f"\n量化结果：")
    print(f"  原始大小: {orig_size:.2f} MB")
    print(f"  量化大小: {quant_size:.2f} MB")
    print(f"  压缩比: {orig_size/quant_size:.2f}x")

if __name__ == '__main__':
    quantize_model(
        'out-code-assistant/ckpt.pt',
        'out-code-assistant/ckpt_int8.pt'
    )
```

**运行：**
```bash
python quantize_model.py

# 输出：
# 原始模型参数: 124.44M
# 
# 量化结果：
#   原始大小: 497.35 MB
#   量化大小: 124.67 MB
#   压缩比: 3.99x
```

---

## 📚 阶段5：API服务

### 🔧 创建FastAPI服务

```python
# serve_api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from model import GPT, GPTConfig
import tiktoken
from contextlib import asynccontextmanager
from typing import Optional
import time

# 全局变量
model = None
tokenizer = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载模型"""
    global model, tokenizer
    
    print("正在加载模型...")
    checkpoint = torch.load('out-code-assistant/ckpt_int8.pt', map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device)
    
    tokenizer = tiktoken.get_encoding("gpt2")
    print(f"模型已加载到 {device}")
    
    yield
    
    # 清理
    del model
    del tokenizer

app = FastAPI(lifespan=lifespan)

class CompletionRequest(BaseModel):
    code: str
    max_tokens: int = 50
    temperature: float = 0.8
    top_k: Optional[int] = 200

class CompletionResponse(BaseModel):
    completion: str
    tokens: int
    latency_ms: float

@app.post("/complete", response_model=CompletionResponse)
async def complete_code(request: CompletionRequest):
    """代码补全API"""
    try:
        start_time = time.time()
        
        # 1. Tokenize输入
        input_ids = tokenizer.encode(request.code)
        if len(input_ids) > 512:
            input_ids = input_ids[-512:]  # 截断
        
        x = torch.tensor([input_ids], dtype=torch.long, device=device)
        
        # 2. 生成
        with torch.no_grad():
            y = model.generate(
                x,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k
            )
        
        # 3. Decode
        output_ids = y[0].tolist()
        completion = tokenizer.decode(output_ids[len(input_ids):])
        
        latency = (time.time() - start_time) * 1000
        
        return CompletionResponse(
            completion=completion,
            tokens=len(output_ids) - len(input_ids),
            latency_ms=latency
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 🚀 启动服务

```bash
# 启动API服务
python serve_api.py

# 输出：
# 正在加载模型...
# 模型已加载到 cuda
# INFO:     Started server process [12345]
# INFO:     Waiting for application startup.
# INFO:     Application startup complete.
# INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 🧪 测试API

```bash
# 测试健康检查
curl http://localhost:8000/health

# 输出：
# {"status":"healthy","model_loaded":true,"device":"cuda"}

# 测试代码补全
curl -X POST http://localhost:8000/complete \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    ",
    "max_tokens": 50,
    "temperature": 0.8
  }'

# 输出：
# {
#   "completion": "return fibonacci(n-1) + fibonacci(n-2)\n\n# Test\nprint(fibonacci(10))",
#   "tokens": 23,
#   "latency_ms": 156.78
# }
```

---

## 📚 阶段6：容器化部署

### 🐳 创建Dockerfile

```dockerfile
# Dockerfile

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 安装Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 复制代码和模型
COPY model.py .
COPY serve_api.py .
COPY out-code-assistant/ckpt_int8.pt ./out-code-assistant/

# 暴露端口
EXPOSE 8000

# 启动服务
CMD ["python3", "serve_api.py"]
```

### 📦 requirements.txt

```txt
torch==2.0.1
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
tiktoken==0.5.1
numpy==1.24.3
```

### 🚀 构建和运行

```bash
# 构建镜像
docker build -t code-assistant:v1 .

# 运行容器
docker run -d \
  --gpus all \
  -p 8000:8000 \
  --name code-assistant \
  code-assistant:v1

# 查看日志
docker logs -f code-assistant

# 测试
curl http://localhost:8000/health
```

---

## 📚 阶段7：Kubernetes部署

### ☸️ 部署配置

```yaml
# k8s/deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: code-assistant
spec:
  replicas: 3  # 3个副本
  selector:
    matchLabels:
      app: code-assistant
  template:
    metadata:
      labels:
        app: code-assistant
    spec:
      containers:
      - name: code-assistant
        image: code-assistant:v1
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1  # 每个pod一个GPU
            memory: "8Gi"
            cpu: "4"
          requests:
            nvidia.com/gpu: 1
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: code-assistant-service
spec:
  selector:
    app: code-assistant
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: code-assistant-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: code-assistant
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### 🚀 部署到K8s

```bash
# 部署
kubectl apply -f k8s/deployment.yaml

# 查看状态
kubectl get pods
kubectl get services

# 查看日志
kubectl logs -f deployment/code-assistant

# 测试服务
kubectl port-forward service/code-assistant-service 8000:80
curl http://localhost:8000/health
```

---

## 📚 阶段8：监控与运维

### 📊 添加Prometheus监控

```python
# serve_api_with_metrics.py

from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
from fastapi import FastAPI
import time

# 定义指标
request_count = Counter(
    'code_completion_requests_total',
    'Total code completion requests',
    ['status']
)

request_duration = Histogram(
    'code_completion_duration_seconds',
    'Code completion request duration',
    buckets=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
)

tokens_generated = Counter(
    'tokens_generated_total',
    'Total tokens generated'
)

active_requests = Gauge(
    'active_requests',
    'Number of active requests'
)

app = FastAPI()

# 添加Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.post("/complete")
async def complete_code(request: CompletionRequest):
    active_requests.inc()
    start_time = time.time()
    
    try:
        # ... 生成代码 ...
        
        # 记录指标
        duration = time.time() - start_time
        request_duration.observe(duration)
        request_count.labels(status='success').inc()
        tokens_generated.inc(len(output_ids) - len(input_ids))
        
        return response
    
    except Exception as e:
        request_count.labels(status='error').inc()
        raise
    
    finally:
        active_requests.dec()
```

### 📈 Grafana仪表板

```json
{
  "dashboard": {
    "title": "Code Assistant Metrics",
    "panels": [
      {
        "title": "Requests per Second",
        "targets": [{
          "expr": "rate(code_completion_requests_total[1m])"
        }]
      },
      {
        "title": "P95 Latency",
        "targets": [{
          "expr": "histogram_quantile(0.95, code_completion_duration_seconds)"
        }]
      },
      {
        "title": "Tokens per Second",
        "targets": [{
          "expr": "rate(tokens_generated_total[1m])"
        }]
      },
      {
        "title": "Active Requests",
        "targets": [{
          "expr": "active_requests"
        }]
      }
    ]
  }
}
```

---

## 📚 阶段9：性能优化

### 🎯 优化清单

```python
优化项目：

1. 模型优化
├── ✅ INT8量化 (4x压缩)
├── ⬜ INT4量化 (8x压缩, 需要GPTQ)
├── ⬜ 模型剪枝 (减少参数)
└── ⬜ 知识蒸馏 (训练小模型)

2. 推理优化
├── ✅ KV Cache (已在model.py中)
├── ⬜ vLLM (20x加速)
├── ⬜ TensorRT (极致性能)
└── ⬜ Continuous Batching

3. 服务优化
├── ✅ 异步API (FastAPI)
├── ⬜ 请求缓存 (Redis)
├── ⬜ 负载均衡 (Nginx)
└── ⬜ CDN加速

4. 基础设施优化
├── ✅ Docker容器化
├── ✅ K8s编排
├── ✅ 自动扩缩容 (HPA)
└── ⬜ Spot实例 (降低成本70%)
```

### 💰 成本分析

```python
# 成本计算
def calculate_monthly_cost():
    # 配置
    gpu_type = "A10G"  # $1.00/hour
    num_gpus = 3  # 3个副本
    hours_per_month = 730
    
    # GPU成本
    gpu_cost = 1.00 * num_gpus * hours_per_month
    
    # 预估请求量
    requests_per_second = 10
    total_requests = requests_per_second * 3600 * hours_per_month
    
    # 每千次请求成本
    cost_per_1k = (gpu_cost / total_requests) * 1000
    
    print(f"""
成本分析：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GPU配置: {num_gpus}x {gpu_type}
月度成本: ${gpu_cost:,.2f}
月度请求: {total_requests:,.0f}
每1K请求成本: ${cost_per_1k:.4f}

优化建议:
1. 使用Spot实例 → 节省70% → ${gpu_cost*0.3:,.2f}/月
2. 使用INT4量化 → GPU减半 → ${gpu_cost*0.5:,.2f}/月
3. 使用vLLM → 吞吐量5x → GPU减少80% → ${gpu_cost*0.2:,.2f}/月

综合优化后: ${gpu_cost*0.3*0.5*0.2:,.2f}/月 (节省97%!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)

calculate_monthly_cost()
```

---

## 🎯 总结：完整部署流程

```python
端到端流程回顾：

✅ 阶段1: 数据准备
  └── 收集Python代码，准备训练数据

✅ 阶段2: 模型训练
  └── 单GPU训练，2-3小时

✅ 阶段3: 分布式加速
  └── 4 GPU训练，30-40分钟 (4x加速)

✅ 阶段4: 模型优化
  └── INT8量化，4x压缩

✅ 阶段5: API服务
  └── FastAPI服务，延迟<200ms

✅ 阶段6: 容器化
  └── Docker镜像，便于部署

✅ 阶段7: K8s部署
  └── 3副本，自动扩缩容

✅ 阶段8: 监控运维
  └── Prometheus + Grafana

✅ 阶段9: 性能优化
  └── 成本优化，节省97%

最终成果：
  ✅ 生产级代码补全服务
  ✅ 支持100+并发用户
  ✅ 延迟 < 200ms
  ✅ 成本 < $0.01/1K tokens
  ✅ 99.9% 可用性
```

---

---

## 🎓 总结与检查

### ✅ 知识检查清单

完成学习后，你应该能够：

**基础概念（必须掌握）**
- [ ] 理解端到端部署的完整流程
- [ ] 知道如何准备和处理数据
- [ ] 能够训练一个基础模型
- [ ] 理解如何构建API服务
- [ ] 知道Docker容器化的基本步骤
- [ ] 能够部署一个简单的服务

**进阶理解（建议掌握）**
- [ ] 理解分布式训练的配置
- [ ] 知道如何量化和优化模型
- [ ] 理解Kubernetes的基本概念
- [ ] 能够配置监控和日志系统
- [ ] 知道如何进行性能优化
- [ ] 理解负载均衡和自动扩缩容

**实战能力（最终目标）**
- [ ] 能够独立完成端到端部署
- [ ] 会处理生产环境的问题
- [ ] 能够优化服务性能和成本
- [ ] 会设计高可用架构
- [ ] 能够监控和排查故障
- [ ] 理解如何持续改进系统

### 📊 部署阶段速查表

| 阶段 | 主要任务 | 关键技术 | 难度 | 重要性 | 预计时间 |
|------|---------|---------|------|--------|---------|
| **数据准备** | 收集、清洗、分词 | Python, tiktoken | ⭐ 简单 | ⭐⭐⭐⭐⭐ | 1-2天 |
| **模型训练** | 训练基础模型 | PyTorch, NanoGPT | ⭐⭐ 中等 | ⭐⭐⭐⭐⭐ | 3-7天 |
| **分布式训练** | 多GPU加速 | DDP, DeepSpeed | ⭐⭐⭐ 较难 | ⭐⭐⭐⭐ | 1-2天 |
| **模型优化** | 量化、加速 | INT8, vLLM | ⭐⭐ 中等 | ⭐⭐⭐⭐⭐ | 1-2天 |
| **API服务** | 构建REST API | FastAPI | ⭐⭐ 中等 | ⭐⭐⭐⭐⭐ | 1-2天 |
| **容器化** | Docker打包 | Docker | ⭐ 简单 | ⭐⭐⭐⭐ | 0.5天 |
| **K8s部署** | 生产环境部署 | Kubernetes | ⭐⭐⭐ 较难 | ⭐⭐⭐⭐ | 2-3天 |
| **监控运维** | 监控和日志 | Prometheus, Grafana | ⭐⭐ 中等 | ⭐⭐⭐⭐⭐ | 1-2天 |
| **性能优化** | 降低成本 | 各种优化技术 | ⭐⭐⭐ 较难 | ⭐⭐⭐⭐ | 持续进行 |

### 🎯 如何规划部署项目？

```python
# 决策树
if 你是初学者:
    # 第1周：基础
    学习数据准备和模型训练
    目标：能跑通训练流程
    
    # 第2周：部署
    学习FastAPI和Docker
    目标：能部署一个简单服务
    
    # 第3-4周：优化
    学习模型优化和监控
    目标：能优化性能
    
elif 你有经验:
    # 第1周：端到端
    快速搭建完整流程
    
    # 第2周：优化
    性能优化和成本优化
    
    # 第3周：生产化
    高可用、监控、运维

# 项目规模估算
小项目（个人/学习）:
  - 单GPU训练：1-2天
  - 简单部署：1天
  - 总计：3-5天

中项目（小团队）:
  - 多GPU训练：3-5天
  - K8s部署：2-3天
  - 监控运维：2天
  - 总计：1-2周

大项目（企业级）:
  - 大规模训练：1-2周
  - 完整部署：1周
  - 优化调试：1-2周
  - 总计：1-2个月
```

### 🚀 下一步学习

现在你已经掌握了生产级部署，接下来应该学习：

1. **11_multimodal_models.md** - 学习多模态模型
2. **12_mixture_of_experts.md** - 学习稀疏模型MoE
3. **13_rlhf_and_alignment.md** - 学习RLHF与模型对齐
4. **实践项目** - 部署一个真实的生产服务

### 💡 实践建议

**立即可做**：
```bash
# 1. 部署一个最小可行产品（MVP）
# 使用预训练模型快速验证
python deploy_mvp.py --model gpt2

# 2. 测试API性能
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 50}'

# 3. 监控资源使用
docker stats
nvidia-smi
```

**系统实验**：
```bash
# 实验1：端到端部署流程
# 从数据准备到服务上线，完整走一遍
./deploy_end_to_end.sh

# 实验2：性能压测
# 测试不同并发下的性能
for concurrency in 1 10 50 100; do
    ab -n 1000 -c $concurrency http://localhost:8000/generate
done

# 实验3：成本优化
# 对比不同配置的成本
python cost_analysis.py \
  --configs fp32,fp16,int8 \
  --instances t4,a10,a100

# 实验4：故障演练
# 模拟各种故障场景
kubectl delete pod api-server-xxx  # 测试自动恢复
```

**进阶研究**：
1. 研究大厂的LLM部署架构（OpenAI、Anthropic）
2. 学习更多优化技术（模型并行、流式推理）
3. 探索边缘部署（移动端、浏览器）
4. 研究成本优化策略（Spot实例、混合云）

---

## 📚 推荐资源

### 📖 必读文档
- [FastAPI Documentation](https://fastapi.tiangolo.com/) - 最好的Python API框架
- [Docker Documentation](https://docs.docker.com/) - 容器化必备
- [Kubernetes Documentation](https://kubernetes.io/docs/) - 容器编排
- [vLLM Documentation](https://docs.vllm.ai/) - 高性能推理引擎

### 📄 重要文章

**部署架构**：
1. **Building LLM applications for production** (Chip Huyen)
   - https://huyenchip.com/2023/04/11/llm-engineering.html
   - LLM工程化最佳实践

2. **Patterns for Building LLM-based Systems** (Eugene Yan)
   - https://eugeneyan.com/writing/llm-patterns/
   - LLM系统设计模式

3. **How to Deploy Large Language Models** (Hugging Face)
   - https://huggingface.co/blog/deploy-llms
   - 部署指南

**性能优化**：
4. **Optimizing LLMs for Speed and Memory** (Hugging Face)
   - https://huggingface.co/docs/transformers/llm_tutorial_optimization
   - 优化技术全面指南

5. **Cost-Effective LLM Serving** (Anyscale)
   - https://www.anyscale.com/blog/cost-effective-llm-serving
   - 成本优化策略

**监控运维**：
6. **Monitoring Machine Learning Models in Production** (Google)
   - https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning
   - MLOps最佳实践

### 🎥 视频教程
- [FastAPI Tutorial](https://www.youtube.com/watch?v=0sOvCWFmrtA)
- [Kubernetes Crash Course](https://www.youtube.com/watch?v=X48VuDVv0do)
- [Docker Tutorial for Beginners](https://www.youtube.com/watch?v=fqMOX6JJhGo)

### 🔧 实用工具

**开发工具**：
```bash
# FastAPI - API框架
pip install fastapi uvicorn

# vLLM - 推理引擎
pip install vllm

# Locust - 压测工具
pip install locust

# Prometheus Client - 监控
pip install prometheus-client
```

**部署工具**：
```bash
# Docker - 容器化
docker build -t my-llm-service .
docker run -p 8000:8000 my-llm-service

# Kubernetes - 编排
kubectl apply -f deployment.yaml
kubectl get pods

# Helm - K8s包管理
helm install my-release ./chart
```

**监控工具**：
```bash
# Prometheus - 指标收集
docker run -p 9090:9090 prom/prometheus

# Grafana - 可视化
docker run -p 3000:3000 grafana/grafana

# Jaeger - 分布式追踪
docker run -p 16686:16686 jaegertracing/all-in-one
```

---

## 🐛 常见问题 FAQ

### Q1: 如何选择部署方式？
**A**: 根据规模和需求选择。
```
个人项目/原型:
  ✅ 单机部署
  ✅ Docker Compose
  ✅ 简单快速
  ❌ 不适合生产
  
小团队/MVP:
  ✅ 云服务（AWS/GCP）
  ✅ 托管K8s（EKS/GKE）
  ✅ 易于扩展
  ⚠️ 成本较高
  
企业级/大规模:
  ✅ 自建K8s集群
  ✅ 多区域部署
  ✅ 完全控制
  ❌ 运维复杂

推荐路径：
  1. 开发：本地Docker
  2. 测试：单机部署
  3. 生产：K8s托管服务
```

### Q2: API延迟多少算正常？
**A**: 取决于模型大小和硬件。
```python
# 延迟基准（生成50 tokens）

小模型（<1B参数）:
  CPU: 5-10秒 ⚠️ 太慢
  T4 GPU: 500-1000ms ✅ 可接受
  A100 GPU: 100-200ms ✅ 很好

中模型（1-10B参数）:
  T4 GPU: 2-5秒 ⚠️ 较慢
  A10 GPU: 500-1000ms ✅ 可接受
  A100 GPU: 200-500ms ✅ 很好

大模型（>10B参数）:
  A10 GPU: 2-5秒 ⚠️ 较慢
  A100 GPU: 500-1000ms ✅ 可接受
  A100×4: 200-300ms ✅ 很好

优化目标：
  - 交互式应用：< 500ms
  - 批处理：< 5秒
  - 离线任务：不限制

如果延迟过高：
  1. 检查是否使用KV Cache
  2. 考虑模型量化
  3. 使用更快的GPU
  4. 减小batch_size
```

### Q3: 如何估算部署成本？
**A**: 计算公式和实际案例。
```python
# 成本 = GPU成本 + 其他成本

# 1. GPU成本（主要）
GPU小时成本:
  T4: $0.35/小时
  A10: $1.28/小时
  A100: $3.67/小时

每月成本（24×30）:
  T4: $252/月
  A10: $922/月
  A100: $2,642/月

# 2. 请求量估算
假设：
  - 1000 req/天
  - 每个请求50 tokens
  - T4 GPU，100 req/小时

GPU利用率: 1000/24/100 = 42%
实际成本: $252 × 0.42 = $106/月

# 3. 优化后
使用INT8量化 + vLLM:
  - 吞吐量提升3x
  - 同样1000 req/天
  - 利用率: 14%
  - 成本: $252 × 0.14 = $35/月

节省: $106 - $35 = $71/月 (67%)

# 4. 实际案例
小项目（1K req/天）:
  - 1×T4: $35-50/月
  
中项目（10K req/天）:
  - 2×A10: $200-300/月
  
大项目（100K req/天）:
  - 4×A100: $2000-3000/月
```

### Q4: Docker镜像太大怎么办？
**A**: 多种优化方法。
```dockerfile
# 问题：基础镜像太大
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
# 镜像大小：8GB+

# 优化1：使用更小的基础镜像
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04
# 镜像大小：2GB

# 优化2：多阶段构建
FROM python:3.10 as builder
RUN pip install --user torch transformers
# 只复制需要的文件

FROM python:3.10-slim
COPY --from=builder /root/.local /root/.local
# 最终镜像：1GB

# 优化3：清理缓存
RUN pip install --no-cache-dir torch
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# 优化4：只安装需要的包
# 不要：pip install transformers（包含所有依赖）
# 而是：pip install torch tokenizers（只装必需的）

# 效果对比
原始镜像: 8GB
优化后: 1-2GB
减少: 75%+
```

### Q5: 如何处理突发流量？
**A**: 自动扩缩容和限流。
```yaml
# 1. Kubernetes HPA（水平自动扩缩容）
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-api
  minReplicas: 2      # 最少2个
  maxReplicas: 10     # 最多10个
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70  # CPU>70%时扩容

# 2. 限流（Rate Limiting）
from fastapi import FastAPI
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()

@app.post("/generate")
@limiter.limit("10/minute")  # 每分钟最多10次
async def generate(request: Request):
    ...

# 3. 队列缓冲
# 突发流量进入队列，慢慢处理
from celery import Celery
app = Celery('tasks')

@app.task
def generate_task(prompt):
    return model.generate(prompt)

# 4. CDN缓存
# 对于相同的请求，返回缓存结果
@app.post("/generate")
@cache(expire=3600)  # 缓存1小时
async def generate(prompt: str):
    ...

# 效果
突发流量: 1000 req/s
限流后: 100 req/s（可处理）
其他900: 排队或返回429
```

### Q6: 如何监控模型质量？
**A**: 多维度监控。
```python
# 1. 输出质量指标
from prometheus_client import Histogram

response_length = Histogram('response_length', 'Response length')
response_time = Histogram('response_time', 'Response time')

@app.post("/generate")
async def generate(prompt: str):
    start = time.time()
    output = model.generate(prompt)
    
    # 记录指标
    response_length.observe(len(output))
    response_time.observe(time.time() - start)
    
    return output

# 2. 内容安全检查
def check_safety(text):
    # 检查有害内容
    if contains_harmful_content(text):
        alert("Harmful content detected!")
        return False
    return True

# 3. 用户反馈
@app.post("/feedback")
async def feedback(request_id: str, rating: int):
    # 收集用户评分
    store_feedback(request_id, rating)
    
    # 低分告警
    if rating < 3:
        alert(f"Low rating: {rating}")

# 4. A/B测试
def get_model_version(user_id):
    # 10%用户使用新版本
    if hash(user_id) % 10 == 0:
        return "model_v2"
    return "model_v1"

# 对比两个版本的指标

# 5. 定期评估
# 每天在测试集上评估
@scheduler.scheduled_job('cron', hour=2)
def daily_eval():
    metrics = evaluate_model(test_dataset)
    if metrics['accuracy'] < threshold:
        alert("Model quality degraded!")
```

### Q7: 如何处理模型更新？
**A**: 蓝绿部署或金丝雀发布。
```yaml
# 蓝绿部署（Blue-Green Deployment）

# 步骤1：部署新版本（Green）
kubectl apply -f deployment-v2.yaml
# 此时v1（Blue）还在运行

# 步骤2：测试新版本
kubectl port-forward svc/llm-api-v2 8001:8000
# 内部测试

# 步骤3：切换流量
kubectl patch service llm-api \
  -p '{"spec":{"selector":{"version":"v2"}}}'
# 流量从v1切到v2

# 步骤4：如果有问题，立即回滚
kubectl patch service llm-api \
  -p '{"spec":{"selector":{"version":"v1"}}}'
# 秒级回滚

# 金丝雀发布（Canary Deployment）

# 步骤1：部署新版本，只给5%流量
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: llm-api
spec:
  http:
  - match:
    - headers:
        canary:
          exact: "true"
    route:
    - destination:
        host: llm-api-v2
      weight: 5    # 5%流量
  - route:
    - destination:
        host: llm-api-v1
      weight: 95   # 95%流量

# 步骤2：观察指标，逐步增加
# 5% → 10% → 25% → 50% → 100%

# 步骤3：如果有问题，停止发布
# 流量回到v1
```

### Q8: 日志应该记录什么？
**A**: 结构化日志，包含关键信息。
```python
import logging
import json

# 配置结构化日志
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

@app.post("/generate")
async def generate(request: GenerateRequest):
    request_id = str(uuid.uuid4())
    
    # 1. 请求日志
    logging.info(json.dumps({
        "event": "request_received",
        "request_id": request_id,
        "prompt_length": len(request.prompt),
        "max_tokens": request.max_tokens,
        "user_id": request.user_id,
        "timestamp": time.time()
    }))
    
    try:
        # 2. 处理日志
        start = time.time()
        output = model.generate(request.prompt)
        duration = time.time() - start
        
        logging.info(json.dumps({
            "event": "request_completed",
            "request_id": request_id,
            "duration": duration,
            "output_length": len(output),
            "tokens_per_second": len(output) / duration
        }))
        
        return {"output": output}
        
    except Exception as e:
        # 3. 错误日志
        logging.error(json.dumps({
            "event": "request_failed",
            "request_id": request_id,
            "error": str(e),
            "traceback": traceback.format_exc()
        }))
        raise

# 关键指标：
# - request_id: 追踪请求
# - duration: 性能监控
# - error: 错误追踪
# - user_id: 用户分析
```

### Q9: 如何做灾难恢复？
**A**: 多层备份和恢复计划。
```bash
# 1. 模型备份
# 定期备份模型到多个位置
aws s3 sync ./models s3://backup-bucket/models/
gsutil rsync -r ./models gs://backup-bucket/models/

# 2. 数据库备份
# 每天自动备份
0 2 * * * pg_dump mydb > backup_$(date +%Y%m%d).sql

# 3. 配置备份
# 版本控制所有配置
git commit -am "Update config"
git push origin main

# 4. 多区域部署
# 在多个区域部署服务
kubectl apply -f deployment-us-east.yaml
kubectl apply -f deployment-eu-west.yaml

# 5. 恢复演练
# 定期测试恢复流程
./disaster_recovery_test.sh

# 恢复时间目标（RTO）:
# - 数据库：< 1小时
# - 模型服务：< 15分钟
# - 完整系统：< 4小时

# 恢复点目标（RPO）:
# - 数据丢失：< 1小时
# - 模型版本：最新版本
```

### Q10: 如何降低部署成本？
**A**: 多方面优化。
```python
# 1. 模型优化（最有效）
量化到INT8: 成本降低75%
量化到INT4: 成本降低87.5%

# 2. 使用Spot实例
AWS Spot: 成本降低70%
但需要处理中断

# 3. 自动扩缩容
高峰期: 10个实例
低峰期: 2个实例
平均成本: 降低60%

# 4. 批处理
实时请求: 立即处理（贵）
离线请求: 批处理（便宜80%）

# 5. 缓存
相同请求: 返回缓存
缓存命中率30%: 成本降低30%

# 6. 选择合适的GPU
不要总用A100:
  - 开发测试: T4 ($0.35/h)
  - 生产小模型: A10 ($1.28/h)
  - 生产大模型: A100 ($3.67/h)

# 7. 区域选择
不同区域价格不同:
  - us-east-1: 最便宜
  - eu-west-1: 贵10-20%
  - ap-southeast: 贵20-30%

# 实际案例
原始成本: $5000/月
  - A100×4, 24/7运行
  
优化后: $800/月
  - INT8量化: -75%
  - 自动扩缩容: -60%
  - Spot实例: -70%
  - 综合: -84%

节省: $4200/月！
```

---

**恭喜你完成第10章！** 🎉

你现在已经掌握了生产级部署的完整流程。从数据准备到模型训练，从API服务到容器化部署，从监控运维到成本优化，你已经具备了构建生产级AI服务的能力。

**准备好了吗？让我们继续前进！** → [11_multimodal_models.md](11_multimodal_models.md)

