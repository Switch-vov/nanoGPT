# 第11章：多模态模型完全指南

> **学习目标**: 理解如何融合视觉、语言等多种模态的信息  
> **难度等级**: 🌿🌿🌿🌿 高级（前沿技术）  
> **预计时间**: 5-6小时  
> **前置知识**: 05模型架构基础

## 🎯 你将学到什么

学完本章，你将能够：
- ✅ 理解多模态学习的基本原理
- ✅ 掌握CLIP、ViT等视觉编码器
- ✅ 理解视觉-语言对齐技术
- ✅ 掌握LLaVA等视觉语言模型
- ✅ 了解文生图、视频理解等应用
- ✅ 能够构建简单的多模态模型

## 💭 开始之前：为什么要学这个？

**场景**：单一模态有局限，多模态才能完整理解世界。

**比喻**：就像人的感官：
- 👁️ 眼睛看：视觉信息
- 👂 耳朵听：听觉信息
- 🧠 大脑：综合理解

**学完之后**：
- ✅ 理解多模态融合原理
- ✅ 能读懂GPT-4V等模型
- ✅ 了解最新多模态技术
- ✅ 能设计多模态应用

---

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

---

## 🎓 总结与检查

### ✅ 知识检查清单

完成学习后，你应该能够：

**基础概念（必须掌握）**
- [ ] 理解什么是多模态学习
- [ ] 知道CLIP的基本原理
- [ ] 理解视觉编码器（ViT）的作用
- [ ] 知道如何对齐不同模态
- [ ] 理解图像描述和VQA的区别
- [ ] 能够使用预训练的多模态模型

**进阶理解（建议掌握）**
- [ ] 理解对比学习的原理
- [ ] 知道LLaVA的架构设计
- [ ] 理解文生图的工作原理
- [ ] 能够微调多模态模型
- [ ] 理解跨模态检索的方法
- [ ] 知道如何评估多模态模型

**实战能力（最终目标）**
- [ ] 能够构建简单的多模态模型
- [ ] 会使用CLIP进行图文检索
- [ ] 能够微调视觉语言模型
- [ ] 会处理多模态数据
- [ ] 能够设计多模态应用
- [ ] 理解多模态模型的局限性

### 📊 多模态模型速查表

| 模型 | 任务 | 模态 | 参数量 | 特点 | 推荐场景 |
|------|------|------|--------|------|---------|
| **CLIP** | 图文匹配 | 图像+文本 | 400M | 零样本能力强 | 图文检索 ⭐⭐⭐⭐⭐ |
| **ViT** | 图像分类 | 图像 | 86M-632M | Transformer架构 | 图像理解 ⭐⭐⭐⭐ |
| **BLIP** | 图文理解 | 图像+文本 | 200M | 统一框架 | 多任务 ⭐⭐⭐⭐ |
| **LLaVA** | 视觉问答 | 图像+文本 | 7B-13B | 基于LLM | 对话应用 ⭐⭐⭐⭐⭐ |
| **GPT-4V** | 通用理解 | 图像+文本 | 未知 | 最强性能 | 商业应用 ⭐⭐⭐⭐⭐ |
| **Stable Diffusion** | 文生图 | 文本→图像 | 1B | 开源可控 | 图像生成 ⭐⭐⭐⭐⭐ |

### 🎯 如何选择多模态模型？

```python
# 决策树
if 任务 == "图文检索":
    使用 CLIP  # 最经典，效果好
    
elif 任务 == "图像描述":
    if 需要高质量:
        使用 BLIP-2 或 LLaVA
    else:
        使用 BLIP  # 更轻量
        
elif 任务 == "视觉问答":
    if 需要对话能力:
        使用 LLaVA 或 GPT-4V  # 基于LLM
    else:
        使用 BLIP  # 简单任务
        
elif 任务 == "文生图":
    if 需要开源:
        使用 Stable Diffusion  # 可控性强
    elif 追求质量:
        使用 DALL-E 3  # 最好但闭源
        
elif 任务 == "视频理解":
    使用 VideoLLaMA 或 Video-ChatGPT
    
# 实际例子
电商搜索: CLIP ✅
智能客服: LLaVA ✅
内容创作: Stable Diffusion ✅
医学影像: 自定义多模态模型 ✅
```

### 🚀 下一步学习

现在你已经掌握了多模态模型，接下来应该学习：

1. **12_mixture_of_experts.md** - 学习稀疏模型MoE
2. **13_rlhf_and_alignment.md** - 学习RLHF与模型对齐
3. **实践项目** - 构建一个多模态应用

### 💡 实践建议

**立即可做**：
```python
# 1. 使用CLIP进行图文检索
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 图文相似度
inputs = processor(text=["a cat", "a dog"], images=image, return_tensors="pt")
outputs = model(**inputs)
similarity = outputs.logits_per_image

# 2. 使用LLaVA进行视觉问答
from llava import LlavaForConditionalGeneration

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
# 问答：What's in this image?

# 3. 使用Stable Diffusion生成图像
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
image = pipe("a cat sitting on a mat").images[0]
```

**系统实验**：
```bash
# 实验1：CLIP零样本分类
python clip_zero_shot.py \
  --images ./test_images/ \
  --labels "cat,dog,bird,car"
# 测试：不同类别的准确率

# 实验2：图文检索系统
python image_text_retrieval.py \
  --image_dir ./images/ \
  --queries "red car,cute cat,sunset"
# 评估：检索准确率

# 实验3：微调视觉语言模型
python finetune_vlm.py \
  --model llava-7b \
  --dataset custom_vqa \
  --epochs 3
# 对比：微调前后的性能

# 实验4：文生图质量对比
python text_to_image_compare.py \
  --models sd2.1,sdxl,dalle3 \
  --prompts prompts.txt
# 评估：生成质量
```

**进阶研究**：
1. 阅读CLIP、LLaVA论文，理解设计思想
2. 研究对比学习的理论基础
3. 探索新的模态组合（音频+视频+文本）
4. 研究多模态在特定领域的应用

---

## 📚 推荐资源

### 📖 必读文档
- [CLIP Documentation](https://github.com/openai/CLIP) - OpenAI官方
- [Hugging Face Multimodal](https://huggingface.co/docs/transformers/model_doc/clip) - 最全的模型库
- [LLaVA Project](https://llava-vl.github.io/) - 视觉语言模型
- [Stable Diffusion Guide](https://stability.ai/stable-diffusion) - 文生图

### 📄 重要论文

**基础模型**：
1. **Learning Transferable Visual Models From Natural Language Supervision (CLIP)** (Radford et al., 2021)
   - https://arxiv.org/abs/2103.00020
   - 多模态学习的里程碑

2. **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)** (Dosovitskiy et al., 2020)
   - https://arxiv.org/abs/2010.11929
   - 视觉Transformer

3. **BLIP: Bootstrapping Language-Image Pre-training** (Li et al., 2022)
   - https://arxiv.org/abs/2201.12086
   - 统一的视觉语言框架

**视觉语言模型**：
4. **Visual Instruction Tuning (LLaVA)** (Liu et al., 2023)
   - https://arxiv.org/abs/2304.08485
   - 视觉指令微调

5. **Flamingo: a Visual Language Model for Few-Shot Learning** (Alayrac et al., 2022)
   - https://arxiv.org/abs/2204.14198
   - DeepMind的多模态模型

6. **GPT-4V(ision) System Card** (OpenAI, 2023)
   - https://openai.com/research/gpt-4v-system-card
   - 最强多模态模型

**文生图**：
7. **High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)** (Rombach et al., 2022)
   - https://arxiv.org/abs/2112.10752
   - 开源文生图

8. **DALL-E 2: Hierarchical Text-Conditional Image Generation with CLIP Latents** (Ramesh et al., 2022)
   - https://arxiv.org/abs/2204.06125
   - OpenAI的文生图

### 🎥 视频教程
- [CLIP Explained](https://www.youtube.com/watch?v=T9XSU0pKX2E)
- [Stable Diffusion Tutorial](https://www.youtube.com/watch?v=1CIpzeNxIhU)
- [LLaVA Demo](https://www.youtube.com/watch?v=qWB4JgCguHw)

### 🔧 实用工具

**模型库**：
```bash
# CLIP - 图文匹配
pip install transformers
from transformers import CLIPModel

# LLaVA - 视觉问答
pip install llava
# 或使用Hugging Face版本

# Stable Diffusion - 文生图
pip install diffusers
from diffusers import StableDiffusionPipeline

# BLIP - 图像描述
pip install salesforce-lavis
```

**数据集**：
```python
# COCO - 图像描述
from datasets import load_dataset
coco = load_dataset("coco")

# Conceptual Captions - 大规模图文对
cc = load_dataset("conceptual_captions")

# VQA - 视觉问答
vqa = load_dataset("vqa_v2")
```

**评估工具**：
```bash
# CLIP Score - 评估图文匹配
pip install clip-score

# FID - 评估图像生成质量
pip install pytorch-fid

# BLEU/CIDEr - 评估图像描述
pip install pycocoevalcap
```

---

## 🐛 常见问题 FAQ

### Q1: 多模态模型和单模态模型有什么区别？
**A**: 核心是信息融合。
```
单模态（如GPT）:
  输入：文本
  处理：文本编码器
  输出：文本
  局限：只能理解文字

多模态（如CLIP）:
  输入：图像 + 文本
  处理：图像编码器 + 文本编码器 + 融合层
  输出：跨模态表示
  优势：理解多种信息

例子：
  问题："这张图里有什么？"
  单模态：无法回答（看不到图）
  多模态：能看图并描述 ✅

结论：多模态更接近人类的感知方式
```

### Q2: CLIP为什么这么强大？
**A**: 对比学习 + 大规模数据。
```python
# CLIP的核心思想
训练数据: 4亿图文对（从互联网收集）

训练目标:
  匹配的图文对 → 相似度高
  不匹配的图文对 → 相似度低

# 例子
图片: [一只猫的照片]
文本1: "a cat" → 相似度 0.95 ✅
文本2: "a dog" → 相似度 0.10 ❌

优势:
  1. 零样本能力：不需要训练就能分类
  2. 通用性强：适用于各种图像任务
  3. 可解释性：通过文本描述控制

实测:
  ImageNet零样本: 76.2%准确率
  （很多监督学习模型才80%+）
```

### Q3: 如何对齐不同模态？
**A**: 通过共享表示空间。
```python
# 方法1：对比学习（CLIP）
图像编码器(image) → image_embedding
文本编码器(text) → text_embedding

loss = contrastive_loss(image_embedding, text_embedding)
# 匹配的拉近，不匹配的推远

# 方法2：交叉注意力（Flamingo）
text_features = text_encoder(text)
image_features = image_encoder(image)

# 文本关注图像的哪些部分
attended_features = cross_attention(text_features, image_features)

# 方法3：适配器（LLaVA）
image_features = vision_encoder(image)
# 通过线性层映射到LLM的空间
adapted_features = projection(image_features)
# 直接输入到LLM

# 选择建议
简单任务: 对比学习（CLIP）
复杂任务: 交叉注意力或适配器
```

### Q4: 视觉编码器用CNN还是Transformer？
**A**: 现在主流是Transformer（ViT）。
```
CNN（传统）:
  代表：ResNet, EfficientNet
  优点：
    - 归纳偏置强（局部性）
    - 训练数据需求少
    - 速度快
  缺点：
    - 感受野有限
    - 难以捕获长距离依赖

ViT（现代）:
  代表：ViT, CLIP Vision
  优点：
    - 全局感受野
    - 可扩展性好
    - 与语言模型统一架构
  缺点：
    - 需要大量数据
    - 计算量大

实际选择：
  小数据集: CNN ✅
  大数据集: ViT ✅
  多模态: ViT ✅（与Transformer统一）

趋势：ViT正在取代CNN
```

### Q5: 如何评估多模态模型？
**A**: 多维度评估。
```python
# 1. 图文检索（CLIP）
from sklearn.metrics import accuracy_score

# 图像检索文本
image_to_text_recall = recall_at_k(predictions, ground_truth, k=5)

# 文本检索图像  
text_to_image_recall = recall_at_k(predictions, ground_truth, k=5)

# 2. 图像描述（BLIP）
from pycocoevalcap.cider.cider import Cider

cider_score = Cider().compute_score(references, predictions)
# CIDEr越高越好（通常0-10）

# 3. 视觉问答（LLaVA）
accuracy = sum(pred == gt for pred, gt in zip(predictions, ground_truth)) / len(predictions)

# 4. 文生图（Stable Diffusion）
from pytorch_fid import fid_score

fid = fid_score.calculate_fid_given_paths([real_path, generated_path])
# FID越低越好（<50算好）

# 5. CLIP Score（图文一致性）
from clip_score import CLIPScore

clip_score = CLIPScore()(images, texts)
# 越高越好（0-100）
```

### Q6: 多模态模型需要多少数据？
**A**: 取决于任务和方法。
```
从头训练（如CLIP）:
  需要：数亿图文对
  时间：数周到数月
  成本：数百万美元
  适合：大公司

微调预训练模型:
  需要：数千到数万样本
  时间：数小时到数天
  成本：数百到数千美元
  适合：大多数场景 ✅

零样本使用:
  需要：0样本！
  时间：立即
  成本：免费
  适合：快速原型 ✅

实际建议：
  1. 先尝试零样本（CLIP）
  2. 如果不够好，收集数据微调
  3. 通常1K-10K样本就能显著提升
```

### Q7: LLaVA和GPT-4V有什么区别？
**A**: 开源vs闭源，性能差距。
```
LLaVA（开源）:
  参数：7B-13B
  性能：很好（80-85分）
  成本：免费
  部署：可自己部署
  定制：可以微调
  适合：研究、定制化需求

GPT-4V（闭源）:
  参数：未知（可能>1T）
  性能：最强（95+分）
  成本：API调用（$0.01-0.03/image）
  部署：只能通过API
  定制：不能微调
  适合：商业应用、追求极致性能

实测对比（VQA任务）:
  LLaVA-13B: 80.0%
  GPT-4V: 93.1%
  
  差距：13.1%
  但LLaVA免费且可定制！

选择建议：
  - 预算充足：GPT-4V
  - 需要定制：LLaVA
  - 数据敏感：LLaVA（本地部署）
```

### Q8: 如何微调多模态模型？
**A**: 类似微调语言模型，但要注意模态对齐。
```python
# 微调LLaVA的例子
from transformers import LlavaForConditionalGeneration, Trainer

# 1. 加载预训练模型
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")

# 2. 准备数据
# 格式：{"image": image_path, "question": "...", "answer": "..."}
train_dataset = load_custom_dataset("train.json")

# 3. 冻结部分参数（可选）
# 只微调语言模型部分，冻结视觉编码器
for param in model.vision_tower.parameters():
    param.requires_grad = False

# 4. 训练
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-5,
        fp16=True,
    )
)

trainer.train()

# 5. 评估
results = trainer.evaluate(eval_dataset)

# 微调技巧：
# - 学习率要小（1e-5 to 5e-5）
# - 可以冻结视觉编码器
# - 使用LoRA减少显存
# - 数据质量比数量重要
```

### Q9: 文生图模型如何工作？
**A**: 扩散模型 + 文本条件。
```python
# Stable Diffusion的工作流程

# 1. 文本编码
text = "a cat sitting on a mat"
text_embedding = text_encoder(text)  # CLIP文本编码器

# 2. 从噪声开始
noise = torch.randn(latent_shape)  # 随机噪声

# 3. 逐步去噪（50步）
for t in range(50, 0, -1):
    # 预测噪声
    predicted_noise = unet(noise, t, text_embedding)
    
    # 去除一点噪声
    noise = noise - predicted_noise * step_size

# 4. 解码到图像
image = vae_decoder(noise)

# 关键参数：
# - guidance_scale: 文本引导强度（7-15）
#   太低：图像与文本不符
#   太高：图像质量下降
# - num_steps: 去噪步数（20-50）
#   太少：质量差
#   太多：慢但质量好

# 实际使用
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
image = pipe(
    prompt="a cat",
    negative_prompt="ugly, blurry",  # 不想要的
    guidance_scale=7.5,
    num_inference_steps=50
).images[0]
```

### Q10: 多模态模型的未来方向？
**A**: 更多模态、更强能力、更易用。
```
趋势1：更多模态
  现在：图像 + 文本
  未来：图像 + 文本 + 音频 + 视频 + 3D
  例子：ImageBind（Meta）- 6种模态

趋势2：更强的推理能力
  现在：识别和描述
  未来：深度推理和规划
  例子：GPT-4V能解数学题、写代码

趋势3：更好的世界模型
  现在：静态理解
  未来：动态预测
  例子：World Models、Sora（视频生成）

趋势4：更易用
  现在：需要技术背景
  未来：人人可用
  例子：ChatGPT式的多模态界面

趋势5：端侧部署
  现在：云端运行
  未来：手机、眼镜上运行
  例子：MobileVLM、TinyLLaVA

研究热点：
  - 统一的多模态架构
  - 高效的模态融合
  - 视频理解和生成
  - 3D场景理解
  - 具身智能（机器人）

机会：
  - 垂直领域应用（医疗、教育）
  - 创作工具（设计、视频）
  - 辅助技术（盲人导航）
  - 元宇宙和AR/VR
```

---

**恭喜你完成第11章！** 🎉

你现在已经掌握了多模态模型的核心技术。从CLIP到LLaVA，从图文检索到视觉问答，从文生图到视频理解，你已经具备了构建多模态AI应用的能力。

**准备好了吗？让我们继续前进！** → [12_mixture_of_experts.md](12_mixture_of_experts.md)
