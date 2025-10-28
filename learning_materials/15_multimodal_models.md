# 多模态模型完全指南

## 🎯 核心问题

**为什么需要多模态？**

```python
单模态的局限:
  纯文本: "一只猫坐在垫子上"
  → 无法理解具体长什么样
  
  纯图像: [猫的图片]
  → 无法理解上下文和语义

多模态的优势:
  文本 + 图像:
  "一只橘猫坐在红色垫子上" + [图片]
  → 完整的理解和描述能力
  
  应用场景:
  ✅ 图像问答 (VQA)
  ✅ 图像描述 (Image Captioning)
  ✅ 文生图 (Text-to-Image)
  ✅ 视频理解
  ✅ 医学影像分析
```

**多模态 = 融合不同类型的信息**

```
视觉 👁️  +  语言 📝  +  音频 🔊  =  完整理解 🧠

就像人类:
  看图片 → 理解内容 → 用语言描述
  听声音 → 理解意思 → 做出反应
  读文字 → 想象场景 → 生成图像
```

---

## 📚 第一部分：多模态基础

### 🔍 核心概念

```python
多模态学习的三大任务:

1. 表示学习 (Representation)
   将不同模态映射到统一空间
   
   例: CLIP
   图像 → 图像embedding (512维)
   文本 → 文本embedding (512维)
   在同一向量空间中！

2. 转换 (Translation)
   从一种模态生成另一种模态
   
   例:
   文本 → 图像 (DALL-E, Stable Diffusion)
   图像 → 文本 (Image Captioning)
   语音 → 文本 (ASR)

3. 对齐 (Alignment)
   找到不同模态间的对应关系
   
   例:
   图像中的猫 ↔ 文本中的"cat"
   视频的动作 ↔ 描述的句子

4. 融合 (Fusion)
   综合多个模态的信息
   
   例:
   图像特征 + 文本特征 → VQA答案
   视频 + 音频 → 视频理解
```

### 📊 模态对比

```python
┌─────────┬──────────┬──────────┬──────────┐
│ 模态    │ 表示     │ 特点     │ 挑战     │
├─────────┼──────────┼──────────┼──────────┤
│ 文本    │ 离散token│ 语义丰富 │ 歧义     │
│ 图像    │ 像素矩阵 │ 视觉信息 │ 高维度   │
│ 音频    │ 波形/频谱│ 时序信号 │ 噪声     │
│ 视频    │ 帧序列   │ 时空信息 │ 计算量大 │
└─────────┴──────────┴──────────┴──────────┘

融合方式:

早期融合 (Early Fusion):
  原始数据 → 直接拼接 → 统一处理
  
  优势: 简单
  劣势: 难以利用模态特性

晚期融合 (Late Fusion):
  各模态独立处理 → 最后融合决策
  
  优势: 灵活
  劣势: 可能丢失交互信息

混合融合 (Hybrid):
  部分早期 + 部分晚期
  
  优势: 平衡
  劣势: 复杂
```

---

## 🖼️ 第二部分：视觉-语言模型

### 1️⃣ CLIP (Contrastive Language-Image Pre-training)

**核心思想：对比学习**

```python
CLIP做什么？
  学习图像和文本的对齐

训练数据:
  400M个(图像, 描述)对
  从互联网收集

训练目标:
  匹配的(图像, 文本)对 → 相似度高
  不匹配的对 → 相似度低
```

**CLIP架构：**

```
┌─────────────────────────────────────────┐
│            CLIP Architecture             │
└─────────────────────────────────────────┘

图像分支:
  图像 (224×224×3)
    ↓
  Vision Transformer / ResNet
    ↓
  图像embedding (512维)

文本分支:
  文本 "A photo of a cat"
    ↓
  Text Transformer
    ↓
  文本embedding (512维)

对比学习:
  Similarity = cosine(image_emb, text_emb)
  
  训练: 最大化匹配对的相似度
       最小化不匹配对的相似度
```

**CLIP实现（简化版）：**

```python
# clip_simple.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageEncoder(nn.Module):
    """图像编码器（简化版ViT）"""
    def __init__(self, image_size=224, patch_size=16, embed_dim=512):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8),
            num_layers=12
        )
        
        # Projection head
        self.proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        # x: [batch, 3, 224, 224]
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, 14, 14]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        
        # Take CLS token
        x = x[:, 0]
        
        # Project
        x = self.proj(x)
        
        # L2 normalize
        x = F.normalize(x, dim=-1)
        
        return x

class TextEncoder(nn.Module):
    """文本编码器"""
    def __init__(self, vocab_size=50000, embed_dim=512, max_len=77):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim))
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8),
            num_layers=12
        )
        
        self.proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        # x: [batch, seq_len]
        x = self.token_embed(x)  # [B, seq_len, embed_dim]
        x = x + self.pos_embed[:, :x.size(1), :]
        
        x = self.transformer(x)
        
        # Take last token (or use pooling)
        x = x[:, -1, :]
        
        x = self.proj(x)
        x = F.normalize(x, dim=-1)
        
        return x

class CLIP(nn.Module):
    """CLIP模型"""
    def __init__(self, embed_dim=512):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim=embed_dim)
        self.text_encoder = TextEncoder(embed_dim=embed_dim)
        
        # Learnable temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, images, texts):
        # Encode
        image_features = self.image_encoder(images)  # [B, embed_dim]
        text_features = self.text_encoder(texts)     # [B, embed_dim]
        
        # Compute similarity
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text
    
    def contrastive_loss(self, logits_per_image, logits_per_text):
        """对比损失"""
        batch_size = logits_per_image.shape[0]
        labels = torch.arange(batch_size, device=logits_per_image.device)
        
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        
        loss = (loss_i + loss_t) / 2
        return loss

# 训练
def train_clip(model, dataloader, epochs=10):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        total_loss = 0
        for images, texts in dataloader:
            # 前向
            logits_img, logits_txt = model(images, texts)
            
            # 损失
            loss = model.contrastive_loss(logits_img, logits_txt)
            
            # 更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch}: Loss = {total_loss/len(dataloader):.4f}")

# 使用CLIP
def zero_shot_classification(model, image, candidate_texts):
    """零样本分类"""
    with torch.no_grad():
        # 编码图像
        image_features = model.image_encoder(image.unsqueeze(0))
        
        # 编码所有候选文本
        text_features = []
        for text in candidate_texts:
            text_ids = tokenize(text)
            feat = model.text_encoder(text_ids.unsqueeze(0))
            text_features.append(feat)
        
        text_features = torch.cat(text_features, dim=0)
        
        # 计算相似度
        logit_scale = model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        probs = F.softmax(logits, dim=-1)
        
        return probs[0].cpu().numpy()

# 示例
model = CLIP()
image = load_image("cat.jpg")
texts = ["a photo of a cat", "a photo of a dog", "a photo of a bird"]
probs = zero_shot_classification(model, image, texts)

print("Probabilities:")
for text, prob in zip(texts, probs):
    print(f"{text}: {prob:.2%}")
# 输出:
# a photo of a cat: 92.5%
# a photo of a dog: 5.2%
# a photo of a bird: 2.3%
```

---

### 2️⃣ LLaVA (Large Language and Vision Assistant)

**核心思想：视觉指令微调**

```python
LLaVA = CLIP视觉编码器 + LLaMA语言模型

架构:
  图像 → CLIP Visual Encoder → 视觉特征
                                    ↓
  文本 ─────────────────────→ 投影层 → 语言模型输入
                                    ↓
                                LLaMA → 生成回答

特点:
  ✅ 可以对话（"这图里有什么？"）
  ✅ 可以推理（"为什么天空是蓝色？"）
  ✅ 可以描述细节
```

**LLaVA实现框架：**

```python
# llava_simple.py

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, LlamaForCausalLM

class LLaVA(nn.Module):
    """简化的LLaVA模型"""
    def __init__(self, vision_model_name='openai/clip-vit-large-patch14',
                 llm_model_name='meta-llama/Llama-2-7b-hf'):
        super().__init__()
        
        # 视觉编码器（冻结）
        self.vision_tower = CLIPVisionModel.from_pretrained(vision_model_name)
        for param in self.vision_tower.parameters():
            param.requires_grad = False
        
        # 语言模型
        self.llm = LlamaForCausalLM.from_pretrained(llm_model_name)
        
        # 投影层（将视觉特征映射到LLM空间）
        vision_hidden_size = self.vision_tower.config.hidden_size  # 1024
        llm_hidden_size = self.llm.config.hidden_size  # 4096
        
        self.mm_projector = nn.Sequential(
            nn.Linear(vision_hidden_size, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )
    
    def encode_images(self, images):
        """编码图像为特征"""
        with torch.no_grad():
            vision_outputs = self.vision_tower(images)
            image_features = vision_outputs.last_hidden_state  # [B, num_patches, hidden]
        
        # 投影到LLM空间
        image_features = self.mm_projector(image_features)
        
        return image_features
    
    def forward(self, images, input_ids, attention_mask=None, labels=None):
        """
        前向传播
        
        Args:
            images: [B, 3, H, W]
            input_ids: [B, seq_len] 文本tokens
            labels: [B, seq_len] 用于计算loss
        """
        batch_size = images.shape[0]
        
        # 编码图像
        image_features = self.encode_images(images)  # [B, num_patches, llm_hidden]
        
        # 获取文本embeddings
        text_embeds = self.llm.model.embed_tokens(input_ids)  # [B, seq_len, llm_hidden]
        
        # 拼接图像和文本特征
        # 假设格式: [IMAGE] text...
        # 找到[IMAGE] token的位置，替换为图像特征
        
        # 简化版：直接在开头拼接
        combined_embeds = torch.cat([image_features, text_embeds], dim=1)
        
        # 调整attention_mask
        image_attention_mask = torch.ones(batch_size, image_features.shape[1],
                                         dtype=torch.long, device=images.device)
        if attention_mask is not None:
            combined_attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)
        else:
            combined_attention_mask = None
        
        # 调整labels
        if labels is not None:
            image_labels = torch.full((batch_size, image_features.shape[1]), -100,
                                     dtype=torch.long, device=images.device)
            combined_labels = torch.cat([image_labels, labels], dim=1)
        else:
            combined_labels = None
        
        # 通过LLM
        outputs = self.llm(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            labels=combined_labels,
            return_dict=True
        )
        
        return outputs
    
    def generate(self, images, input_ids, max_new_tokens=100, temperature=0.7):
        """生成回答"""
        # 编码图像
        image_features = self.encode_images(images)
        
        # 获取文本embeddings
        text_embeds = self.llm.model.embed_tokens(input_ids)
        
        # 拼接
        combined_embeds = torch.cat([image_features, text_embeds], dim=1)
        
        # 生成
        # 注意：需要特殊处理，这里简化
        outputs = self.llm.generate(
            inputs_embeds=combined_embeds,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True
        )
        
        return outputs

# 训练LLaVA
def train_llava(model, dataloader, epochs=3):
    """
    训练数据格式:
    {
        'image': image tensor,
        'conversations': [
            {'from': 'human', 'value': 'What is in this image?'},
            {'from': 'gpt', 'value': 'This image shows a cat sitting on a mat...'}
        ]
    }
    """
    # 只训练投影层和LLM
    optimizer = torch.optim.AdamW([
        {'params': model.mm_projector.parameters(), 'lr': 2e-5},
        {'params': model.llm.parameters(), 'lr': 2e-6}
    ])
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch in dataloader:
            images = batch['images']
            input_ids = batch['input_ids']
            labels = batch['labels']
            
            # 前向
            outputs = model(images, input_ids, labels=labels)
            loss = outputs.loss
            
            # 更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch}: Loss = {total_loss/len(dataloader):.4f}")

# 推理示例
def chat_with_image(model, image, question, tokenizer):
    """与图像对话"""
    # 格式化输入
    prompt = f"### Human: {question}\n### Assistant:"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # 生成
    output_ids = model.generate(
        images=image.unsqueeze(0),
        input_ids=input_ids,
        max_new_tokens=200
    )
    
    # 解码
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

# 使用
model = LLaVA()
image = load_image("scene.jpg")
response = chat_with_image(model, image, "Describe this image in detail.")
print(response)
# 输出: "This image shows a beautiful sunset over the ocean..."
```

---

## 🎥 第三部分：视频理解

### 🎬 视频模型架构

```python
视频 = 时序的图像序列

挑战:
  1. 时间维度 (T帧)
  2. 空间维度 (H×W×C)
  3. 计算量: T×H×W×C

解决方案:

方法1: 3D CNN
  扩展2D卷积到3D
  同时捕获时空信息
  
  例: I3D, C3D

方法2: Two-Stream
  空间流: 处理单帧
  时间流: 处理光流
  最后融合

方法3: Transformer
  视频patch → Transformer
  
  例: TimeSformer, VideoMAE
```

**简化视频模型：**

```python
# video_model.py

import torch
import torch.nn as nn

class VideoTransformer(nn.Module):
    """视频理解Transformer"""
    def __init__(self, num_frames=16, image_size=224, patch_size=16,
                 embed_dim=768, num_classes=400):
        super().__init__()
        
        self.num_frames = num_frames
        self.num_patches_per_frame = (image_size // patch_size) ** 2
        
        # Patch embedding（3D）
        self.patch_embed = nn.Conv3d(
            3, embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size)
        )
        
        # Position embedding
        num_patches = num_frames * self.num_patches_per_frame
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=12),
            num_layers=12
        )
        
        # 分类头
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        """
        x: [B, T, C, H, W] - batch, time, channels, height, width
        """
        B, T, C, H, W = x.shape
        
        # Reshape for conv3d
        x = x.transpose(1, 2)  # [B, C, T, H, W]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, T, H', W']
        
        # Flatten patches
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        
        # Classification
        x = x[:, 0]  # CLS token
        x = self.head(x)
        
        return x

# 使用
model = VideoTransformer(num_classes=400)  # Kinetics-400
video = torch.randn(2, 16, 3, 224, 224)  # 2 videos, 16 frames each
logits = model(video)
print(logits.shape)  # [2, 400]
```

---

## 🎵 第四部分：音频-语言模型

### 🎤 音频处理基础

```python
音频表示:

1. 波形 (Waveform)
   原始信号: [时间点 × 幅度]
   
2. 频谱 (Spectrogram)
   时频表示: [时间 × 频率]
   
3. Mel频谱 (Mel-Spectrogram)
   人耳感知: [时间 × Mel频段]
   更适合语音

模型:
  Wav2Vec 2.0: 自监督音频学习
  Whisper: 语音识别
  AudioLM: 音频生成
```

**音频编码器：**

```python
# audio_encoder.py

import torch
import torch.nn as nn
import torchaudio

class AudioEncoder(nn.Module):
    """音频编码器"""
    def __init__(self, sample_rate=16000, n_mels=80, embed_dim=512):
        super().__init__()
        
        # Mel-Spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=400,
            hop_length=160
        )
        
        # CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Projection
        self.proj = nn.Linear(256, embed_dim)
    
    def forward(self, waveform):
        """
        waveform: [B, time]
        """
        # Mel spectrogram
        mel = self.mel_transform(waveform)  # [B, n_mels, time]
        mel = mel.unsqueeze(1)  # [B, 1, n_mels, time]
        
        # Encode
        features = self.encoder(mel)  # [B, 256, 1, 1]
        features = features.squeeze(-1).squeeze(-1)  # [B, 256]
        
        # Project
        embedding = self.proj(features)  # [B, embed_dim]
        
        return embedding

# 音频-文本对比学习（类似CLIP）
class AudioCLIP(nn.Module):
    """音频-文本对齐"""
    def __init__(self, embed_dim=512):
        super().__init__()
        self.audio_encoder = AudioEncoder(embed_dim=embed_dim)
        self.text_encoder = TextEncoder(embed_dim=embed_dim)  # 复用CLIP的
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, audio, text):
        audio_features = self.audio_encoder(audio)
        text_features = self.text_encoder(text)
        
        # Normalize
        audio_features = F.normalize(audio_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Similarity
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * audio_features @ text_features.t()
        
        return logits
```

---

## 🔧 第五部分：构建多模态GPT

### 🎯 统一多模态架构

```python
目标: 一个模型处理所有模态

设计思路:
  1. 将所有模态映射到统一token空间
  2. 用Transformer处理
  3. 根据任务生成相应模态的输出

架构:
  [图像tokens] [文本tokens] [音频tokens]
           ↓
      Transformer
           ↓
  [输出tokens] → 解码到各模态
```

**完整多模态模型：**

```python
# multimodal_gpt.py

class MultiModalTokenizer:
    """多模态tokenizer"""
    def __init__(self):
        # 特殊tokens
        self.IMG_START = 50257
        self.IMG_END = 50258
        self.AUD_START = 50259
        self.AUD_END = 50260
        
        # 各模态的codebook大小
        self.image_vocab_size = 8192  # 来自VQ-VAE
        self.audio_vocab_size = 1024
        self.text_vocab_size = 50257  # GPT-2
    
    def encode_image(self, image):
        """图像 → tokens"""
        # 使用VQ-VAE编码
        image_tokens = vqvae.encode(image)  # [H/8, W/8]
        image_tokens = image_tokens.flatten()  # [num_tokens]
        
        # 添加特殊tokens
        tokens = [self.IMG_START] + image_tokens.tolist() + [self.IMG_END]
        return tokens
    
    def encode_text(self, text):
        """文本 → tokens"""
        return gpt2_tokenizer.encode(text)
    
    def encode_audio(self, audio):
        """音频 → tokens"""
        audio_tokens = audio_vqvae.encode(audio)
        tokens = [self.AUD_START] + audio_tokens.tolist() + [self.AUD_END]
        return tokens

class MultiModalGPT(nn.Module):
    """统一多模态GPT"""
    def __init__(self, vocab_size=60000, embed_dim=768, num_layers=12):
        super().__init__()
        
        # Token embedding（包含所有模态）
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        
        # Position embedding
        self.pos_embed = nn.Embedding(2048, embed_dim)
        
        # Transformer
        self.transformer = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads=12)
            for _ in range(num_layers)
        ])
        
        # 各模态的输出头
        self.text_head = nn.Linear(embed_dim, 50257)
        self.image_head = nn.Linear(embed_dim, 8192)
        self.audio_head = nn.Linear(embed_dim, 1024)
    
    def forward(self, input_ids, modality_ids=None):
        """
        input_ids: [B, seq_len] 混合的多模态tokens
        modality_ids: [B, seq_len] 指示每个token的模态类型
        """
        B, T = input_ids.shape
        
        # Embeddings
        token_emb = self.token_embed(input_ids)
        pos = torch.arange(T, device=input_ids.device)
        pos_emb = self.pos_embed(pos)
        
        x = token_emb + pos_emb
        
        # Transformer
        for block in self.transformer:
            x = block(x)
        
        # 根据模态选择输出头
        logits = self.route_to_heads(x, modality_ids)
        
        return logits
    
    def route_to_heads(self, x, modality_ids):
        """根据模态路由到相应的输出头"""
        if modality_ids is None:
            # 默认使用文本头
            return self.text_head(x)
        
        # 分别处理各模态
        # 这里简化，实际需要更复杂的逻辑
        outputs = []
        for i in range(x.size(0)):
            # 判断当前位置的模态
            if is_text_position(modality_ids[i]):
                outputs.append(self.text_head(x[i]))
            elif is_image_position(modality_ids[i]):
                outputs.append(self.image_head(x[i]))
            else:
                outputs.append(self.audio_head(x[i]))
        
        return torch.stack(outputs)
    
    def generate_multimodal(self, prompt_tokens, modality_ids, max_length=100):
        """多模态生成"""
        generated = prompt_tokens
        
        for _ in range(max_length):
            # 前向传播
            logits = self.forward(generated, modality_ids)
            
            # 取最后一个token的logits
            next_token_logits = logits[:, -1, :]
            
            # 采样
            next_token = torch.multinomial(
                F.softmax(next_token_logits, dim=-1),
                num_samples=1
            )
            
            # 添加到序列
            generated = torch.cat([generated, next_token], dim=1)
            
            # 停止条件
            if next_token.item() in [IMG_END, AUD_END, EOS]:
                break
        
        return generated

# 使用示例
def image_captioning(model, tokenizer, image):
    """图像描述"""
    # 编码图像
    image_tokens = tokenizer.encode_image(image)
    
    # 添加提示
    prompt = image_tokens + tokenizer.encode_text("Caption: ")
    prompt_tensor = torch.tensor([prompt])
    
    # 生成描述
    output = model.generate_multimodal(prompt_tensor, max_length=50)
    
    # 解码
    caption = tokenizer.decode_text(output[0])
    return caption

def text_to_image(model, tokenizer, text):
    """文生图"""
    # 编码文本
    text_tokens = tokenizer.encode_text(text)
    
    # 添加图像开始token
    prompt = text_tokens + [tokenizer.IMG_START]
    prompt_tensor = torch.tensor([prompt])
    
    # 生成图像tokens
    output = model.generate_multimodal(prompt_tensor, max_length=256)
    
    # 提取图像tokens并解码
    image_tokens = extract_between(output, IMG_START, IMG_END)
    image = vqvae.decode(image_tokens)
    
    return image
```

---

## 💡 第六部分：训练技巧

### ⚡ 1. 数据对齐

```python
挑战: 不同模态的数据不总是对齐的

解决方案:

1. 弱监督学习
   利用自然配对数据
   例: 网页的图片+alt文本

2. 跨模态检索
   找到相关的跨模态样本
   
3. 自监督学习
   掩码预测、对比学习

数据增强:
  图像: 裁剪、翻转、颜色变换
  文本: 回译、同义替换
  音频: 时间拉伸、添加噪声
  
  关键: 保持跨模态一致性！
```

### ⚡ 2. 训练策略

```python
阶段1: 单模态预训练
  分别预训练各编码器
  文本: GPT预训练
  图像: ViT on ImageNet
  音频: Wav2Vec 2.0

阶段2: 跨模态对齐
  冻结编码器
  只训练投影层
  使用对比学习

阶段3: 端到端微调
  解冻所有参数
  在下游任务上微调
  小学习率！

超参数:
  学习率: 1e-5 ~ 1e-4
  Batch size: 尽可能大 (256-4096)
  温度: 0.07 (对比学习)
  权重衰减: 0.1
```

### ⚡ 3. 评估指标

```python
图像-文本:
  检索任务:
    - Recall@K
    - Mean Rank
  
  生成任务:
    - BLEU, ROUGE (描述质量)
    - CLIPScore (语义对齐)
    - FID (图像质量)

视频理解:
  - Top-1 Accuracy
  - mAP (多标签)

音频:
  - WER (词错误率, ASR)
  - MOS (平均意见分, 生成质量)
```

---

## 🎓 总结

### ✨ 核心要点

```python
多模态学习的核心:
  1. 表示学习: 统一空间
  2. 对齐: 跨模态对应
  3. 融合: 综合信息
  4. 转换: 模态互译

关键模型:
  CLIP: 图像-文本对齐
  LLaVA: 视觉对话
  Whisper: 语音识别
  DALL-E/Stable Diffusion: 文生图

训练技巧:
  ✅ 大规模配对数据
  ✅ 对比学习
  ✅ 分阶段训练
  ✅ 适当的数据增强
```

### 🎯 实践建议

```python
你的项目 → 推荐方案

图像问答:
  CLIP + GPT-2
  或使用LLaVA

图像生成:
  Stable Diffusion
  或从VQGAN+Transformer开始

视频理解:
  预训练ViT + 时序建模
  或TimeSformer

语音识别:
  Whisper (开箱即用)
  或Wav2Vec 2.0微调

统一多模态:
  从小规模开始
  2个模态 → 3个模态 → ...
```

### 📚 学习资源

```python
论文:
  ⭐⭐⭐⭐⭐ CLIP (OpenAI, 2021)
  ⭐⭐⭐⭐⭐ DALL-E / Stable Diffusion
  ⭐⭐⭐⭐ LLaVA (2023)
  ⭐⭐⭐⭐ Flamingo (DeepMind, 2022)

数据集:
  - COCO (图像描述)
  - Conceptual Captions (大规模)
  - AudioSet (音频)
  - Kinetics (视频)

工具:
  - CLIP (OpenAI)
  - transformers (Hugging Face)
  - torchvision, torchaudio
  - diffusers (Stable Diffusion)
```

---

**最后一句话：**

> 多模态是AI理解真实世界的关键。
> 就像人类用眼睛看、耳朵听、嘴巴说，
> 多模态AI整合各种感知，
> 才能真正理解和创造。
>
> 未来的AI必然是多模态的。
> 现在就是最好的学习时机！

🌈 **开启你的多模态AI之旅！** 🚀
