# 生产级部署实战：端到端项目

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

## 📚 推荐资源

### 工具和框架
- [FastAPI](https://fastapi.tiangolo.com/)
- [vLLM](https://docs.vllm.ai/)
- [Kubernetes](https://kubernetes.io/)
- [Prometheus](https://prometheus.io/)

### 最佳实践
- [12-Factor App](https://12factor.net/)
- [Google SRE Book](https://sre.google/books/)

---

**恭喜！** 你已经完成了从训练到部署的完整流程！🎉

