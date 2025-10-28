# 模型部署完全指南

## 🎯 核心问题

**从研究到生产的鸿沟：**
- 训练好的模型在本地跑得很好
- 但如何让用户访问？
- 如何处理并发请求？
- 如何保证低延迟和高吞吐？
- 如何监控和维护？

**部署的挑战：**
```python
研究环境:
  python train.py  # 简单
  python sample.py # 生成文本

生产环境:
  ❓ 如何提供API服务？
  ❓ 如何处理1000个并发用户？
  ❓ 如何保证99.9%可用性？
  ❓ 如何监控和调试？
  ❓ 如何更新模型？
```

---

## 📚 第一部分：部署架构概览

### 🏗️ 部署层级

```python
Level 1: 本地脚本
  适用: 个人使用、实验
  方式: python sample.py
  
Level 2: API服务
  适用: 小团队、内部工具
  方式: Flask/FastAPI
  
Level 3: 生产级服务
  适用: 对外产品、大规模应用
  方式: Docker + Kubernetes + 负载均衡
  
Level 4: 专业推理平台
  适用: 企业级、高性能需求
  方式: TensorRT, Triton, vLLM
```

### 🎯 部署目标

```python
关键指标:

延迟 (Latency):
  < 100ms: 实时对话 ✅
  100-500ms: 可接受
  > 1s: 用户体验差 ❌
  
吞吐量 (Throughput):
  requests/second
  tokens/second
  
可用性 (Availability):
  99.9%: 每月停机43分钟
  99.99%: 每月停机4分钟
  
成本 (Cost):
  GPU利用率
  每个请求的成本
```

---

## 🚀 第二部分：FastAPI快速部署

### 📝 基础API服务

**创建 serve.py：**

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from model import GPT, GPTConfig
import pickle
from contextlib import asynccontextmanager

# 全局变量存储模型
model_cache = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时加载模型
    print("加载模型...")
    
    # 加载checkpoint
    checkpoint = torch.load('out/ckpt.pt', map_location='cuda')
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to('cuda')
    
    # 加载tokenizer
    with open('data/shakespeare_char/meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    
    # 存储到全局
    model_cache['model'] = model
    model_cache['meta'] = meta
    
    print("模型加载完成!")
    
    yield
    
    # 清理
    model_cache.clear()

# 创建FastAPI应用
app = FastAPI(title="NanoGPT API", lifespan=lifespan)

# 请求模型
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 200

class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int

@app.get("/")
async def root():
    """健康检查"""
    return {"status": "ok", "message": "NanoGPT API is running"}

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """生成文本"""
    try:
        model = model_cache['model']
        meta = model_cache['meta']
        
        # 编码
        stoi = meta['stoi']
        itos = meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
        
        # 生成
        input_ids = torch.tensor([encode(request.prompt)], dtype=torch.long, device='cuda')
        
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k
            )
        
        # 解码
        generated_text = decode(output[0].tolist())
        
        return GenerateResponse(
            text=generated_text,
            tokens_generated=len(output[0]) - len(input_ids[0])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """健康检查端点"""
    if 'model' in model_cache:
        return {"status": "healthy", "model_loaded": True}
    else:
        return {"status": "unhealthy", "model_loaded": False}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**启动服务：**

```bash
# 安装依赖
pip install fastapi uvicorn pydantic

# 启动服务
python serve.py

# 输出:
# 加载模型...
# 模型加载完成!
# INFO:     Started server process
# INFO:     Uvicorn running on http://0.0.0.0:8000
```

**测试API：**

```bash
# 方法1: curl
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "ROMEO:",
    "max_tokens": 100,
    "temperature": 0.8
  }'

# 输出:
# {
#   "text": "ROMEO:\nWhat lady is that, which doth enrich the hand\nOf yonder knight?",
#   "tokens_generated": 100
# }

# 方法2: Python客户端
import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "To be or not to be",
        "max_tokens": 50,
        "temperature": 0.8
    }
)

print(response.json()['text'])
```

---

## 🐳 第三部分：Docker容器化

### 📦 创建Dockerfile

```dockerfile
# Dockerfile

FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# 设置工作目录
WORKDIR /app

# 安装Python和依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 复制代码
COPY model.py .
COPY serve.py .
COPY out/ ./out/
COPY data/shakespeare_char/meta.pkl ./data/shakespeare_char/

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python3", "serve.py"]
```

### 📝 requirements.txt

```txt
torch==2.1.0
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
```

### 🚀 构建和运行

```bash
# 构建镜像
docker build -t nanogpt-api:latest .

# 运行容器
docker run -d \
  --name nanogpt \
  --gpus all \
  -p 8000:8000 \
  nanogpt-api:latest

# 查看日志
docker logs -f nanogpt

# 测试
curl http://localhost:8000/health
```

### 📊 Docker Compose（推荐）

```yaml
# docker-compose.yml

version: '3.8'

services:
  nanogpt-api:
    build: .
    container_name: nanogpt
    ports:
      - "8000:8000"
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./out:/app/out:ro  # 只读挂载模型
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    container_name: nanogpt-nginx
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - nanogpt-api
    restart: unless-stopped
```

**nginx.conf（负载均衡）：**

```nginx
events {
    worker_connections 1024;
}

http {
    upstream nanogpt_backend {
        # 多个实例负载均衡
        server nanogpt-api:8000;
        # server nanogpt-api-2:8000;
        # server nanogpt-api-3:8000;
    }
    
    server {
        listen 80;
        
        location / {
            proxy_pass http://nanogpt_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            
            # 超时设置
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }
    }
}
```

**启动完整服务：**

```bash
docker-compose up -d

# 扩展到3个实例
docker-compose up -d --scale nanogpt-api=3
```

---

## ⚡ 第四部分：性能优化

### 1️⃣ **批处理推理**

```python
# 优化版serve.py - 支持批处理

from collections import deque
import asyncio
import time

class BatchProcessor:
    def __init__(self, model, max_batch_size=8, max_wait_time=0.01):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        
        self.queue = deque()
        self.processing = False
    
    async def add_request(self, request):
        """添加请求到队列"""
        future = asyncio.Future()
        self.queue.append((request, future))
        
        # 如果没在处理，启动处理
        if not self.processing:
            asyncio.create_task(self._process_batch())
        
        return await future
    
    async def _process_batch(self):
        """批处理请求"""
        self.processing = True
        start_time = time.time()
        
        # 等待凑够一批或超时
        while len(self.queue) < self.max_batch_size:
            if time.time() - start_time > self.max_wait_time:
                break
            await asyncio.sleep(0.001)
        
        # 取出一批请求
        batch = []
        futures = []
        for _ in range(min(self.max_batch_size, len(self.queue))):
            if self.queue:
                req, future = self.queue.popleft()
                batch.append(req)
                futures.append(future)
        
        if batch:
            # 批量推理
            results = await self._batch_generate(batch)
            
            # 返回结果
            for future, result in zip(futures, results):
                future.set_result(result)
        
        self.processing = False
        
        # 如果还有请求，继续处理
        if self.queue:
            asyncio.create_task(self._process_batch())
    
    async def _batch_generate(self, batch):
        """批量生成"""
        # 编码所有prompt
        input_ids_list = []
        max_len = 0
        
        for req in batch:
            ids = encode(req.prompt)
            input_ids_list.append(ids)
            max_len = max(max_len, len(ids))
        
        # Padding到相同长度
        padded_ids = []
        for ids in input_ids_list:
            padded = ids + [0] * (max_len - len(ids))
            padded_ids.append(padded)
        
        input_ids = torch.tensor(padded_ids, dtype=torch.long, device='cuda')
        
        # 批量生成
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=batch[0].max_tokens,
                temperature=batch[0].temperature
            )
        
        # 解码所有输出
        results = []
        for output in outputs:
            text = decode(output.tolist())
            results.append(GenerateResponse(
                text=text,
                tokens_generated=len(output) - max_len
            ))
        
        return results

# 使用批处理器
batch_processor = None

@app.on_event("startup")
async def startup():
    global batch_processor
    # ... 加载模型 ...
    batch_processor = BatchProcessor(model)

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    return await batch_processor.add_request(request)
```

**效果：**

```python
性能对比:

单请求处理:
  延迟: 100ms
  吞吐: 10 req/s

批处理 (batch_size=8):
  延迟: 110ms (略增)
  吞吐: 70 req/s (7x!)
```

---

### 2️⃣ **KV Cache优化**

```python
# 在model.py中实现KV Cache

class GPT(nn.Module):
    def generate_with_cache(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """使用KV Cache的生成（更快）"""
        # 初始化cache
        past_key_values = None
        
        for _ in range(max_new_tokens):
            # 如果有cache，只处理最后一个token
            if past_key_values is not None:
                idx_cond = idx[:, [-1]]
            else:
                idx_cond = idx
            
            # 前向传播（返回cache）
            logits, past_key_values = self.forward_with_cache(
                idx_cond, 
                past_key_values
            )
            
            # 采样
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
    def forward_with_cache(self, idx, past_key_values=None):
        """支持KV Cache的前向传播"""
        # 实现细节...
        # 保存每层的K, V
        # 下次只计算新token的K, V
        pass

# 加速效果
标准生成: 100 tokens in 2.5s
KV Cache: 100 tokens in 0.8s (3x faster!)
```

---

### 3️⃣ **模型优化汇总**

```python
优化清单:

□ 量化 (INT8/INT4)
  压缩4-8x，加速2-3x
  
□ KV Cache
  生成加速2-3x
  
□ 批处理
  吞吐提升5-10x
  
□ TorchScript/ONNX
  移除Python开销
  
□ TensorRT
  GPU推理优化
  
□ Flash Attention
  内存效率提升

组合效果:
  基线: 10 req/s
  + 量化: 20 req/s
  + KV Cache: 40 req/s
  + 批处理: 200 req/s (20x!)
```

---

## 🔧 第五部分：专业推理引擎

### 🚀 vLLM（推荐）

**安装：**

```bash
pip install vllm
```

**使用代码：**

```python
from vllm import LLM, SamplingParams

# 加载模型
llm = LLM(
    model="gpt2",  # 或你的模型路径
    tensor_parallel_size=1,  # GPU数量
    dtype="float16",
)

# 采样参数
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100,
)

# 批量生成
prompts = [
    "Hello, my name is",
    "The capital of France is",
    "The future of AI is",
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
    print()
```

**创建API服务器：**

```python
# vllm_serve.py

from vllm import LLM, SamplingParams
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# 启动时加载模型
llm = LLM(model="gpt2", dtype="float16")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.8

@app.post("/generate")
async def generate(request: GenerateRequest):
    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )
    
    outputs = llm.generate([request.prompt], sampling_params)
    return {"text": outputs[0].outputs[0].text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**性能优势：**

```python
对比 (LLaMA-7B, batch_size=32):

方法                | Throughput
────────────────────┼────────────
HuggingFace         | 150 tokens/s
Custom FastAPI      | 450 tokens/s
vLLM               | 2100 tokens/s (14x!)

vLLM优势:
  ✅ PagedAttention (更好的内存管理)
  ✅ Continuous batching (持续批处理)
  ✅ 优化的CUDA kernels
  ✅ 开箱即用
```

---

### 🔥 TensorRT（NVIDIA）

**转换为TensorRT：**

```bash
# 安装
pip install tensorrt onnx

# 导出ONNX
python export_onnx.py

# 转换为TensorRT
trtexec --onnx=model.onnx \
        --saveEngine=model.trt \
        --fp16  # 使用FP16
```

**使用TensorRT：**

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class TensorRTInference:
    def __init__(self, engine_path):
        # 加载引擎
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
    
    def infer(self, input_data):
        # 推理
        # ... TensorRT推理代码 ...
        pass

# 使用
model = TensorRTInference("model.trt")
output = model.infer(input_data)
```

**加速效果：**

```
对比 (GPT-2, T4 GPU):

PyTorch FP32: 45 ms/iter
PyTorch FP16: 28 ms/iter
TensorRT FP16: 12 ms/iter (3.8x!)
TensorRT INT8: 7 ms/iter (6.4x!)
```

---

## 📊 第六部分：监控和日志

### 📈 添加监控

```python
# serve.py 添加监控

from prometheus_client import Counter, Histogram, make_asgi_app
import time

# Prometheus指标
REQUEST_COUNT = Counter(
    'nanogpt_requests_total',
    'Total requests',
    ['endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'nanogpt_request_latency_seconds',
    'Request latency',
    ['endpoint']
)

TOKENS_GENERATED = Counter(
    'nanogpt_tokens_generated_total',
    'Total tokens generated'
)

@app.post("/generate")
async def generate(request: GenerateRequest):
    start_time = time.time()
    
    try:
        # ... 生成逻辑 ...
        
        # 记录指标
        REQUEST_COUNT.labels(endpoint='generate', status='success').inc()
        TOKENS_GENERATED.inc(len(output))
        
        return response
        
    except Exception as e:
        REQUEST_COUNT.labels(endpoint='generate', status='error').inc()
        raise
    
    finally:
        # 记录延迟
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint='generate').observe(latency)

# 添加metrics端点
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

### 📝 结构化日志

```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
        }
        if hasattr(record, 'request_id'):
            log_obj['request_id'] = record.request_id
        return json.dumps(log_obj)

# 配置日志
logger = logging.getLogger('nanogpt')
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# 使用
@app.post("/generate")
async def generate(request: GenerateRequest):
    request_id = str(uuid.uuid4())
    logger.info(
        f"Received request: prompt='{request.prompt[:50]}...'",
        extra={'request_id': request_id}
    )
    
    # ... 生成 ...
    
    logger.info(
        f"Generated {tokens} tokens in {latency:.2f}s",
        extra={'request_id': request_id}
    )
```

---

## 🔒 第七部分：安全和认证

### 🔐 API认证

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

SECRET_KEY = "your-secret-key"  # 应该从环境变量读取

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """验证JWT token"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

@app.post("/generate")
async def generate(
    request: GenerateRequest,
    user = Depends(verify_token)  # 需要认证
):
    # ... 生成逻辑 ...
    pass
```

### 🛡️ 速率限制

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/generate")
@limiter.limit("10/minute")  # 每分钟最多10次
async def generate(request: Request, gen_request: GenerateRequest):
    # ... 生成逻辑 ...
    pass
```

---

## 🌐 第八部分：云平台部署

### ☁️ AWS部署

**使用Amazon SageMaker：**

```python
# deploy_sagemaker.py

from sagemaker.pytorch import PyTorchModel
from sagemaker import get_execution_role

role = get_execution_role()

# 创建模型
pytorch_model = PyTorchModel(
    model_data='s3://my-bucket/model.tar.gz',
    role=role,
    entry_point='inference.py',
    framework_version='2.0',
    py_version='py310',
)

# 部署
predictor = pytorch_model.deploy(
    instance_type='ml.g4dn.xlarge',  # GPU实例
    initial_instance_count=1,
)

# 调用
result = predictor.predict({
    'prompt': 'Hello, world',
    'max_tokens': 100
})
```

### 🔷 Azure部署

**使用Azure ML：**

```python
# deploy_azure.py

from azureml.core import Workspace, Model
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.model import InferenceConfig

# 连接workspace
ws = Workspace.from_config()

# 注册模型
model = Model.register(
    workspace=ws,
    model_path='out/ckpt.pt',
    model_name='nanogpt'
)

# 部署配置
aci_config = AciWebservice.deploy_configuration(
    cpu_cores=2,
    memory_gb=8,
    gpu_cores=1
)

# 部署
service = Model.deploy(
    workspace=ws,
    name='nanogpt-service',
    models=[model],
    inference_config=InferenceConfig(...),
    deployment_config=aci_config
)

service.wait_for_deployment(show_output=True)
print(service.scoring_uri)
```

---

## 💡 第九部分：最佳实践

### ✅ 部署清单

```python
□ 模型优化
  □ 量化 (INT8)
  □ KV Cache
  □ 批处理
  
□ 容器化
  □ Dockerfile
  □ Docker Compose
  □ 健康检查
  
□ API设计
  □ 清晰的端点
  □ 请求验证
  □ 错误处理
  □ 文档 (Swagger)
  
□ 性能
  □ 异步处理
  □ 连接池
  □ 缓存
  
□ 安全
  □ 认证
  □ 速率限制
  □ 输入验证
  
□ 监控
  □ 日志
  □ 指标 (Prometheus)
  □ 告警
  
□ 可靠性
  □ 负载均衡
  □ 自动扩展
  □ 故障恢复
```

### 🎯 性能调优

```python
优先级排序:

1. 模型量化 (INT8)
   影响: 4x压缩, 2x加速
   难度: ⭐
   
2. KV Cache
   影响: 2-3x加速
   难度: ⭐⭐
   
3. 批处理
   影响: 5-10x吞吐
   难度: ⭐⭐⭐
   
4. vLLM/TensorRT
   影响: 10-20x吞吐
   难度: ⭐⭐
   
5. 多GPU部署
   影响: Nx吞吐
   难度: ⭐⭐⭐⭐
```

---

## 🐛 第十部分：故障排查

### 常见问题

```python
问题1: OOM (Out of Memory)
解决:
  - 减小batch_size
  - 启用KV Cache复用
  - 使用量化模型
  - 增加GPU内存

问题2: 推理慢
解决:
  - 启用批处理
  - 使用量化
  - 检查CPU瓶颈
  - 使用vLLM/TensorRT

问题3: 并发请求失败
解决:
  - 增加worker数量
  - 使用异步处理
  - 添加请求队列
  - 负载均衡

问题4: 模型预测不一致
解决:
  - 检查随机种子
  - 验证模型版本
  - 确认温度参数
  - 检查tokenizer

问题5: 容器启动失败
解决:
  - 检查GPU驱动
  - 验证镜像
  - 查看日志
  - 检查资源限制
```

---

## 🎓 总结

### ✨ 核心要点

```python
1. 部署层级
   本地脚本 → API服务 → 生产级 → 专业平台
   
2. 关键技术
   - FastAPI: 简单易用
   - Docker: 容器化
   - vLLM: 高性能推理
   - TensorRT: 极致优化
   
3. 性能优化
   量化 + KV Cache + 批处理 = 20x+提升
   
4. 监控必备
   - 日志
   - 指标 (Prometheus)
   - 告警
   
5. 安全措施
   - 认证
   - 速率限制
   - 输入验证
```

### 🎯 推荐方案

```python
你的场景 → 推荐方案

个人项目/原型:
  → FastAPI + Docker
  → 简单快速

小团队/内部工具:
  → FastAPI + Docker + Nginx
  → 负载均衡

生产环境 (云端):
  → vLLM + Kubernetes
  → 高性能 + 自动扩展

边缘部署:
  → TensorRT + INT8
  → 极致优化

企业级:
  → Triton Inference Server
  → 多模型、多框架
```

### 🚀 完整流程

```python
1. 开发
   训练模型 → 验证效果
   
2. 优化
   量化 → 测试 → 选择最佳配置
   
3. 封装
   创建API → 容器化 → 测试
   
4. 部署
   选择平台 → 部署 → 验证
   
5. 监控
   日志 → 指标 → 告警
   
6. 迭代
   收集反馈 → 优化 → 更新
```

---

**记住：**

> 部署不是终点，而是起点。
> 从简单开始，逐步优化。
> 监控是关键，用数据驱动决策。
>
> 一个好的部署方案，应该：
> - 快速迭代
> - 稳定可靠  
> - 易于监控
> - 成本可控

🎉 恭喜你完成了从训练到部署的完整学习！
