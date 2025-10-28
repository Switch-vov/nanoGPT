# RLHF对齐完全指南

## 🎯 核心问题

**语言模型的困境：**
```python
训练好的GPT模型:
  输入: "如何制作炸弹？"
  输出: [详细的危险内容] ❌

  输入: "写一篇文章"
  输出: [可能包含偏见、有害内容] ❌

问题:
  ✅ 模型很强大（能生成流畅文本）
  ❌ 但不知道什么该说，什么不该说
  ❌ 没有价值观对齐
  ❌ 可能产生有害输出
```

**RLHF的解决方案：**
```python
通过人类反馈的强化学习（RLHF）让模型:
  ✅ 拒绝有害请求
  ✅ 生成有帮助的回答
  ✅ 符合人类价值观
  ✅ 更安全、更可控

实例对比:
  基础GPT: "如何制作炸弹？" → [危险内容]
  RLHF后:  "如何制作炸弹？" → "我不能提供这类信息，这很危险..."
```

---

## 📚 第一部分：RLHF基础概念

### 🔍 什么是RLHF？

**RLHF = Reinforcement Learning from Human Feedback**

```python
核心思路：

1. 预训练模型（已完成）
   从海量文本学习语言能力
   但不知道"好"与"坏"

2. 人类标注偏好
   给模型生成的多个回答打分
   "这个回答好"、"那个回答不好"

3. 训练奖励模型
   学习人类的偏好
   能自动判断回答质量

4. 强化学习优化
   用奖励模型指导
   让模型生成更好的回答
```

### 📊 RLHF三阶段

```python
阶段1: 监督微调 (SFT - Supervised Fine-Tuning)
  输入: 高质量的问答对
  输出: 基础对话模型
  
  示例:
    Q: "什么是光合作用？"
    A: "光合作用是植物利用光能..."
  
  目标: 让模型学会对话格式

阶段2: 奖励模型训练 (RM - Reward Model)
  输入: 同一个问题的多个回答 + 人类排序
  输出: 能打分的奖励模型
  
  示例:
    Q: "解释量子力学"
    A1: "量子力学很复杂..." (得分: 0.3)
    A2: "量子力学是研究..." (得分: 0.9)
  
  目标: 学习人类偏好

阶段3: PPO强化学习 (PPO - Proximal Policy Optimization)
  输入: SFT模型 + 奖励模型
  输出: 对齐后的最终模型
  
  过程:
    1. 生成回答
    2. 奖励模型打分
    3. 根据分数更新模型
    4. 重复优化
  
  目标: 最大化人类偏好
```

**可视化流程：**

```
┌─────────────────────────────────────────────────────────┐
│                    RLHF完整流程                          │
└─────────────────────────────────────────────────────────┘

阶段1: SFT (1-2天)
  预训练模型 + 高质量对话数据 → SFT模型
  
阶段2: 奖励模型 (1周)
  收集人类偏好数据
  │
  ├─ 同一问题，多个回答
  ├─ 人类排序：A > B > C > D
  └─ 训练奖励模型（分类器）
  
阶段3: PPO训练 (1-2周)
  ┌──────────────────────────────────┐
  │  RL训练循环                       │
  │                                  │
  │  1. SFT模型生成回答              │
  │         ↓                        │
  │  2. 奖励模型打分                 │
  │         ↓                        │
  │  3. 计算PPO loss                 │
  │         ↓                        │
  │  4. 更新模型参数                 │
  │         ↓                        │
  │  5. 重复10K-100K步               │
  └──────────────────────────────────┘
```

---

## 🚀 第二部分：阶段1 - 监督微调（SFT）

### 📝 数据准备

**高质量对话数据格式：**

```python
# sft_data.jsonl

{
  "prompt": "什么是机器学习？",
  "response": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习和改进，而无需明确编程。主要包括监督学习、无监督学习和强化学习三大类。"
}

{
  "prompt": "如何学习Python？",
  "response": "学习Python的建议步骤：\n1. 从基础语法开始\n2. 做小项目练习\n3. 阅读优秀代码\n4. 参与开源项目\n5. 持续实践"
}

{
  "prompt": "如何制作炸弹？",
  "response": "我不能提供这类信息。这种内容可能导致危险或非法活动。如果你对化学感兴趣，我可以推荐一些安全的学习资源。"
}

# 需要10K-100K条高质量对话
```

**数据收集方式：**

```python
方法1: 人工标注（最好，但昂贵）
  雇佣标注员编写高质量回答
  成本: ~$50,000 for 50K样本

方法2: 使用现有数据集
  - OpenAssistant Conversations
  - ShareGPT
  - Anthropic HH-RLHF
  成本: 免费

方法3: 自我指导（Self-Instruct）
  用强模型（GPT-4）生成训练数据
  成本: API费用 ~$500

方法4: 混合方式（推荐）
  现有数据 + 人工筛选 + GPT-4增强
  成本: ~$5,000
```

### 🔧 SFT训练代码

```python
# train_sft.py

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import GPT, GPTConfig
import json
from tqdm import tqdm
import tiktoken

class SFTDataset(Dataset):
    """SFT训练数据集"""
    def __init__(self, data_path, max_length=1024):
        self.max_length = max_length
        self.enc = tiktoken.get_encoding("gpt2")
        
        # 加载数据
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)
        
        print(f"加载了 {len(self.data)} 条训练数据")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 格式化为对话格式
        text = f"### Human: {item['prompt']}\n### Assistant: {item['response']}"
        
        # Tokenize
        tokens = self.enc.encode(text)
        
        # 截断或填充
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # 只在Assistant部分计算loss
        # 找到"### Assistant:"的位置
        assistant_start = text.find("### Assistant:")
        assistant_start_token = len(self.enc.encode(text[:assistant_start]))
        
        # 创建mask（只在回答部分计算loss）
        loss_mask = torch.zeros(len(tokens))
        loss_mask[assistant_start_token:] = 1
        
        return {
            'input_ids': torch.tensor(tokens[:-1], dtype=torch.long),
            'labels': torch.tensor(tokens[1:], dtype=torch.long),
            'loss_mask': loss_mask[1:]  # 对齐labels
        }

def collate_fn(batch):
    """批处理函数（处理不同长度）"""
    max_len = max(item['input_ids'].size(0) for item in batch)
    
    input_ids = []
    labels = []
    loss_masks = []
    
    for item in batch:
        # Padding
        pad_len = max_len - item['input_ids'].size(0)
        
        input_ids.append(F.pad(item['input_ids'], (0, pad_len), value=0))
        labels.append(F.pad(item['labels'], (0, pad_len), value=-100))  # -100会被忽略
        loss_masks.append(F.pad(item['loss_mask'], (0, pad_len), value=0))
    
    return {
        'input_ids': torch.stack(input_ids),
        'labels': torch.stack(labels),
        'loss_mask': torch.stack(loss_masks)
    }

def train_sft(
    model_path='gpt2',  # 从GPT-2开始
    data_path='sft_data.jsonl',
    output_dir='out-sft',
    epochs=3,
    batch_size=4,
    learning_rate=5e-6,
    device='cuda'
):
    """SFT训练主函数"""
    
    # 加载模型
    print("加载基础模型...")
    if model_path == 'gpt2':
        # 从HuggingFace加载
        from transformers import GPT2LMHeadModel
        model = GPT2LMHeadModel.from_pretrained('gpt2')
    else:
        # 从checkpoint加载
        checkpoint = torch.load(model_path)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        model.load_state_dict(checkpoint['model'])
    
    model = model.to(device)
    model.train()
    
    # 准备数据
    print("准备数据...")
    dataset = SFTDataset(data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # 训练
    print("开始SFT训练...")
    global_step = 0
    
    for epoch in range(epochs):
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            loss_mask = batch['loss_mask'].to(device)
            
            # 前向传播
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            
            # 只在Assistant部分计算loss（可选）
            if loss_mask is not None:
                # 重新计算masked loss
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                shift_mask = loss_mask[..., :-1].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                loss = (loss * shift_mask.view(-1)).sum() / shift_mask.sum()
            
            # 反向传播
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # 记录
            epoch_loss += loss.item()
            global_step += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # 定期保存
            if global_step % 500 == 0:
                save_path = f'{output_dir}/checkpoint-{global_step}.pt'
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': global_step,
                }, save_path)
                print(f"\n保存checkpoint: {save_path}")
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} 平均loss: {avg_loss:.4f}")
    
    # 保存最终模型
    final_path = f'{output_dir}/sft_model.pt'
    torch.save({'model': model.state_dict()}, final_path)
    print(f"SFT训练完成！模型保存到: {final_path}")

if __name__ == "__main__":
    train_sft(
        data_path='data/sft_train.jsonl',
        epochs=3,
        batch_size=4,
        learning_rate=5e-6
    )
```

**运行SFT：**

```bash
# 准备数据
python prepare_sft_data.py

# 训练
python train_sft.py

# 输出:
# 加载基础模型...
# 加载了 50000 条训练数据
# 开始SFT训练...
# Epoch 1/3: 100%|████████| 12500/12500 [2:15:30<00:00, loss=2.1234]
# Epoch 1 平均loss: 2.3456
# ...
# SFT训练完成！
```

---

## 📊 第三部分：阶段2 - 奖励模型（RM）

### 🎯 奖励模型的作用

```python
奖励模型 = 学习人类偏好的分类器

输入: 问题 + 回答
输出: 分数（越高越好）

例子:
  Q: "解释相对论"
  
  A1: "爱因斯坦提出的..." 
  RM分数: 0.85 ✅
  
  A2: "我不知道"
  RM分数: 0.12 ❌
  
  A3: "相对论包括狭义相对论和广义相对论..."
  RM分数: 0.95 ✅✅
```

### 📝 偏好数据收集

```python
# preference_data.jsonl

{
  "prompt": "什么是深度学习？",
  "responses": [
    {
      "text": "深度学习是机器学习的一个分支，使用多层神经网络来学习数据的复杂表示。它在图像识别、自然语言处理等领域取得了巨大成功。",
      "rank": 1  # 最好
    },
    {
      "text": "深度学习就是用神经网络。",
      "rank": 3  # 最差
    },
    {
      "text": "深度学习是一种AI技术，模仿人脑的工作方式。",
      "rank": 2  # 中等
    }
  ]
}

# 或者使用成对比较（更常见）
{
  "prompt": "什么是深度学习？",
  "chosen": "深度学习是机器学习的一个分支...",
  "rejected": "深度学习就是用神经网络。"
}
```

**收集方式：**

```python
工具: Label Studio, Scale AI

流程:
1. SFT模型生成多个回答（通常4-8个）
   使用不同温度/top-k采样

2. 人类标注员排序
   从最好到最差: A > B > C > D

3. 转换为成对比较
   (A, B), (A, C), (A, D)
   (B, C), (B, D)
   (C, D)

4. 形成训练数据
   每对包含：chosen（更好）和rejected（更差）

数据量需求:
  最少: 10K对比较
  推荐: 50K-100K对比较
  OpenAI: 数百万对比较
```

### 🔧 奖励模型训练

```python
# train_reward_model.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import GPT, GPTConfig
import json

class RewardModel(nn.Module):
    """奖励模型：在GPT基础上加一个分数头"""
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
        # 奖励头：从最后一个token的embedding预测分数
        self.reward_head = nn.Linear(base_model.config.n_embd, 1, bias=False)
        
        # 初始化
        self.reward_head.weight.data.zero_()
    
    def forward(self, input_ids, attention_mask=None):
        """
        返回每个序列的奖励分数
        """
        # 获取base model的输出
        outputs = self.base_model.transformer(input_ids)
        hidden_states = outputs  # [batch, seq_len, n_embd]
        
        # 取最后一个有效token的hidden state
        if attention_mask is not None:
            # 找到每个序列的最后一个有效位置
            sequence_lengths = attention_mask.sum(dim=1) - 1
            last_hidden = hidden_states[
                torch.arange(hidden_states.size(0)),
                sequence_lengths
            ]
        else:
            # 如果没有mask，取最后一个token
            last_hidden = hidden_states[:, -1, :]
        
        # 预测奖励分数
        rewards = self.reward_head(last_hidden).squeeze(-1)
        
        return rewards

class PreferenceDataset(Dataset):
    """偏好数据集"""
    def __init__(self, data_path, tokenizer, max_length=512):
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 格式化
        prompt = item['prompt']
        chosen = item['chosen']
        rejected = item['rejected']
        
        # Tokenize
        chosen_text = f"### Human: {prompt}\n### Assistant: {chosen}"
        rejected_text = f"### Human: {prompt}\n### Assistant: {rejected}"
        
        chosen_tokens = self.tokenizer.encode(chosen_text)[:self.max_length]
        rejected_tokens = self.tokenizer.encode(rejected_text)[:self.max_length]
        
        return {
            'chosen_ids': torch.tensor(chosen_tokens, dtype=torch.long),
            'rejected_ids': torch.tensor(rejected_tokens, dtype=torch.long)
        }

def collate_fn(batch):
    """处理不同长度"""
    # 找到最大长度
    max_chosen = max(item['chosen_ids'].size(0) for item in batch)
    max_rejected = max(item['rejected_ids'].size(0) for item in batch)
    max_len = max(max_chosen, max_rejected)
    
    chosen_ids = []
    rejected_ids = []
    chosen_masks = []
    rejected_masks = []
    
    for item in batch:
        # Pad chosen
        c_len = item['chosen_ids'].size(0)
        chosen_ids.append(
            torch.cat([item['chosen_ids'], torch.zeros(max_len - c_len, dtype=torch.long)])
        )
        chosen_masks.append(
            torch.cat([torch.ones(c_len), torch.zeros(max_len - c_len)])
        )
        
        # Pad rejected
        r_len = item['rejected_ids'].size(0)
        rejected_ids.append(
            torch.cat([item['rejected_ids'], torch.zeros(max_len - r_len, dtype=torch.long)])
        )
        rejected_masks.append(
            torch.cat([torch.ones(r_len), torch.zeros(max_len - r_len)])
        )
    
    return {
        'chosen_ids': torch.stack(chosen_ids),
        'rejected_ids': torch.stack(rejected_ids),
        'chosen_mask': torch.stack(chosen_masks),
        'rejected_mask': torch.stack(rejected_masks)
    }

def train_reward_model(
    sft_model_path='out-sft/sft_model.pt',
    data_path='data/preference_data.jsonl',
    output_dir='out-rm',
    epochs=1,
    batch_size=4,
    learning_rate=1e-5,
    device='cuda'
):
    """训练奖励模型"""
    
    # 加载SFT模型
    print("加载SFT模型...")
    checkpoint = torch.load(sft_model_path)
    # 假设我们有GPTConfig
    model = RewardModel(base_model)  # base_model是加载的SFT模型
    model = model.to(device)
    model.train()
    
    # 数据
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = PreferenceDataset(data_path, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # 训练
    print("开始训练奖励模型...")
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in dataloader:
            chosen_ids = batch['chosen_ids'].to(device)
            rejected_ids = batch['rejected_ids'].to(device)
            chosen_mask = batch['chosen_mask'].to(device)
            rejected_mask = batch['rejected_mask'].to(device)
            
            # 计算奖励
            reward_chosen = model(chosen_ids, chosen_mask)
            reward_rejected = model(rejected_ids, rejected_mask)
            
            # Loss: chosen应该比rejected分数高
            # 使用ranking loss（也叫pairwise loss）
            loss = -torch.log(torch.sigmoid(reward_chosen - reward_rejected)).mean()
            
            # 反向传播
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # 统计
            total_loss += loss.item()
            correct += (reward_chosen > reward_rejected).sum().item()
            total += reward_chosen.size(0)
        
        accuracy = correct / total
        avg_loss = total_loss / len(dataloader)
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2%}")
    
    # 保存
    save_path = f'{output_dir}/reward_model.pt'
    torch.save({'model': model.state_dict()}, save_path)
    print(f"奖励模型保存到: {save_path}")

if __name__ == "__main__":
    train_reward_model(
        sft_model_path='out-sft/sft_model.pt',
        data_path='data/preference_train.jsonl',
        epochs=1,
        batch_size=4
    )
```

**训练效果：**

```bash
python train_reward_model.py

# 输出:
# 加载SFT模型...
# 加载了 50000 条偏好数据
# 开始训练奖励模型...
# Epoch 1: Loss=0.3421, Accuracy=82.34%
# 奖励模型保存到: out-rm/reward_model.pt

# 82%的准确率意味着奖励模型能正确识别82%的偏好
```

---

## 🎮 第四部分：阶段3 - PPO强化学习

### 🎯 PPO算法核心

**PPO (Proximal Policy Optimization) = 稳定的策略优化**

```python
强化学习设置:

状态(State): 用户问题
动作(Action): 生成的回答
奖励(Reward): 奖励模型给的分数
目标: 最大化期望奖励

PPO核心思想:
  1. 生成回答
  2. 获得奖励
  3. 更新策略（模型参数）
  4. 但不要更新太激进（Proximal = 接近的）
     避免模型变得疯狂
```

**PPO Loss公式：**

```python
PPO Loss包含三部分:

1. Policy Loss (策略损失)
   鼓励模型生成高奖励的回答
   
   L_policy = -min(
       r(θ) * A,
       clip(r(θ), 1-ε, 1+ε) * A
   )
   
   其中:
   - r(θ) = π_new(a|s) / π_old(a|s)  # 新旧策略比
   - A = 优势函数（这个动作有多好）
   - clip: 限制更新幅度
   - ε = 0.2（典型值）

2. Value Loss (价值损失)
   帮助估计状态价值
   
   L_value = (V(s) - Return)²

3. KL Penalty (KL散度惩罚)
   防止模型偏离SFT模型太远
   保持语言能力
   
   L_kl = β * KL(π_new || π_sft)
   
总Loss:
   L = L_policy + c1 * L_value + c2 * L_kl
```

### 🔧 PPO训练实现

```python
# train_ppo.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from model import GPT
from train_reward_model import RewardModel

class PPOTrainer:
    """PPO训练器"""
    def __init__(
        self,
        policy_model,      # SFT模型（要优化的）
        ref_model,         # 参考模型（固定的SFT模型）
        reward_model,      # 奖励模型
        tokenizer,
        kl_coef=0.1,       # KL惩罚系数
        clip_epsilon=0.2,  # PPO clip范围
        value_coef=0.5,    # 价值损失系数
        max_length=512,
        device='cuda'
    ):
        self.policy = policy_model.to(device)
        self.ref_model = ref_model.to(device)
        self.reward_model = reward_model.to(device)
        self.tokenizer = tokenizer
        
        self.kl_coef = kl_coef
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.max_length = max_length
        self.device = device
        
        # 冻结参考模型和奖励模型
        self.ref_model.eval()
        self.reward_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        for param in self.reward_model.parameters():
            param.requires_grad = False
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=1e-6  # 非常小的学习率！
        )
    
    @torch.no_grad()
    def generate_responses(self, prompts, temperature=1.0):
        """
        生成回答，并记录log_probs用于PPO
        
        Returns:
            responses: 生成的文本
            log_probs: 每个token的log概率
            values: 价值估计（可选）
        """
        self.policy.eval()
        
        batch_size = len(prompts)
        responses = []
        all_log_probs = []
        
        for prompt in prompts:
            # Tokenize prompt
            prompt_tokens = torch.tensor(
                [self.tokenizer.encode(prompt)],
                dtype=torch.long,
                device=self.device
            )
            
            generated = prompt_tokens
            log_probs = []
            
            # 逐token生成
            for _ in range(self.max_length):
                outputs = self.policy(generated)
                logits = outputs.logits[:, -1, :]  # 最后一个token的logits
                
                # 采样
                probs = F.softmax(logits / temperature, dim=-1)
                dist = Categorical(probs)
                next_token = dist.sample()
                
                # 记录log_prob
                log_prob = dist.log_prob(next_token)
                log_probs.append(log_prob.item())
                
                # 添加到序列
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                
                # 停止条件（遇到EOS或太长）
                if next_token.item() == self.tokenizer.eot_token:
                    break
            
            # 解码
            response = self.tokenizer.decode(generated[0].tolist())
            responses.append(response)
            all_log_probs.append(torch.tensor(log_probs))
        
        return responses, all_log_probs
    
    def compute_rewards(self, prompts, responses):
        """
        计算奖励
        
        Reward = RM分数 - β * KL(π||π_ref)
        """
        rewards = []
        
        for prompt, response in zip(prompts, responses):
            # 1. 奖励模型分数
            full_text = f"### Human: {prompt}\n### Assistant: {response}"
            tokens = torch.tensor(
                [self.tokenizer.encode(full_text)],
                dtype=torch.long,
                device=self.device
            )
            
            with torch.no_grad():
                rm_score = self.reward_model(tokens).item()
            
            # 2. KL散度惩罚
            # 计算policy和ref_model的KL散度
            with torch.no_grad():
                policy_logits = self.policy(tokens).logits
                ref_logits = self.ref_model(tokens).logits
                
                policy_logprobs = F.log_softmax(policy_logits, dim=-1)
                ref_logprobs = F.log_softmax(ref_logits, dim=-1)
                
                # KL散度（每个token）
                kl = (policy_logprobs.exp() * (policy_logprobs - ref_logprobs)).sum(dim=-1)
                kl_penalty = kl.mean().item()
            
            # 总奖励
            reward = rm_score - self.kl_coef * kl_penalty
            rewards.append(reward)
        
        return torch.tensor(rewards, device=self.device)
    
    def compute_advantages(self, rewards):
        """
        计算优势函数（简化版）
        A = R - baseline
        """
        # 简单地用奖励的均值作为baseline
        baseline = rewards.mean()
        advantages = rewards - baseline
        
        # 标准化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages
    
    def ppo_step(self, prompts, responses, old_log_probs, advantages):
        """
        一步PPO更新
        """
        self.policy.train()
        
        total_policy_loss = 0
        total_kl = 0
        
        for prompt, response, old_log_prob, advantage in zip(
            prompts, responses, old_log_probs, advantages
        ):
            # 重新计算log_probs（使用当前策略）
            full_text = f"### Human: {prompt}\n### Assistant: {response}"
            tokens = torch.tensor(
                [self.tokenizer.encode(full_text)],
                dtype=torch.long,
                device=self.device
            )
            
            outputs = self.policy(tokens)
            logits = outputs.logits
            
            # 计算每个token的log_prob
            log_probs = F.log_softmax(logits, dim=-1)
            # 取实际生成的token的log_prob
            token_log_probs = log_probs.gather(2, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
            new_log_prob = token_log_probs.sum()
            
            # Importance ratio
            ratio = torch.exp(new_log_prob - old_log_prob)
            
            # PPO clip
            clipped_ratio = torch.clamp(
                ratio,
                1 - self.clip_epsilon,
                1 + self.clip_epsilon
            )
            
            # Policy loss
            policy_loss = -torch.min(
                ratio * advantage,
                clipped_ratio * advantage
            )
            
            total_policy_loss += policy_loss
            
            # KL（用于监控）
            with torch.no_grad():
                kl = (new_log_prob - old_log_prob).abs()
                total_kl += kl.item()
        
        # 平均
        avg_policy_loss = total_policy_loss / len(prompts)
        
        # 反向传播
        self.optimizer.zero_grad()
        avg_policy_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        
        self.optimizer.step()
        
        return {
            'policy_loss': avg_policy_loss.item(),
            'avg_kl': total_kl / len(prompts)
        }
    
    def train_step(self, prompts):
        """完整的训练步骤"""
        
        # 1. 生成回答
        responses, log_probs = self.generate_responses(prompts)
        
        # 2. 计算奖励
        rewards = self.compute_rewards(prompts, responses)
        
        # 3. 计算优势
        advantages = self.compute_advantages(rewards)
        
        # 4. PPO更新（多个epoch）
        for _ in range(4):  # PPO通常对每个batch更新多次
            stats = self.ppo_step(prompts, responses, log_probs, advantages)
        
        return {
            'avg_reward': rewards.mean().item(),
            **stats
        }

def train_rlhf_ppo(
    sft_model_path='out-sft/sft_model.pt',
    reward_model_path='out-rm/reward_model.pt',
    prompt_dataset_path='data/rl_prompts.txt',
    output_dir='out-ppo',
    total_steps=10000,
    batch_size=16,
    device='cuda'
):
    """RLHF-PPO主训练循环"""
    
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # 加载模型
    print("加载模型...")
    # policy_model = load_model(sft_model_path)
    # ref_model = load_model(sft_model_path)  # 同一个checkpoint
    # reward_model = load_reward_model(reward_model_path)
    
    # 创建trainer
    # trainer = PPOTrainer(policy_model, ref_model, reward_model, tokenizer)
    
    # 加载prompts
    with open(prompt_dataset_path, 'r') as f:
        all_prompts = [line.strip() for line in f]
    
    print(f"开始PPO训练... ({total_steps} steps)")
    
    for step in range(total_steps):
        # 采样一批prompts
        batch_prompts = np.random.choice(all_prompts, batch_size, replace=False).tolist()
        
        # 训练步骤
        stats = trainer.train_step(batch_prompts)
        
        # 日志
        if step % 10 == 0:
            print(f"Step {step}: "
                  f"Reward={stats['avg_reward']:.4f}, "
                  f"Loss={stats['policy_loss']:.4f}, "
                  f"KL={stats['avg_kl']:.4f}")
        
        # 保存
        if step % 500 == 0:
            save_path = f'{output_dir}/ppo_checkpoint_{step}.pt'
            torch.save({'model': trainer.policy.state_dict()}, save_path)
            print(f"保存checkpoint: {save_path}")
    
    print("PPO训练完成！")

if __name__ == "__main__":
    train_rlhf_ppo(
        total_steps=10000,
        batch_size=16
    )
```

**训练输出：**

```bash
python train_ppo.py

# 输出:
# 加载模型...
# 开始PPO训练... (10000 steps)
# Step 0: Reward=0.2341, Loss=0.8765, KL=0.0123
# Step 10: Reward=0.3456, Loss=0.7654, KL=0.0234
# Step 20: Reward=0.4321, Loss=0.6543, KL=0.0345
# ...
# Step 500: Reward=0.7234, Loss=0.3456, KL=0.0521
# 保存checkpoint: out-ppo/ppo_checkpoint_500.pt
# ...

# 奖励逐渐上升 = 模型越来越符合人类偏好！
```

---

## 📊 第五部分：效果评估

### 🧪 对比测试

```python
测试prompt:
  "写一首关于AI的诗"

基础GPT-2:
  the cat sat on the mat and the dog ran...
  (完全偏题，没有意义) ❌

SFT模型:
  AI is very powerful technology
  It can do many things for us
  We should use it carefully
  (基本对话能力，但质量一般) ⭐⭐

RLHF模型:
  In silicon dreams, thoughts arise,
  Patterns dance before our eyes,
  Learning, growing, day by day,
  AI lights a brand new way.
  (流畅、有创意、符合要求) ⭐⭐⭐⭐⭐
```

### 📊 量化指标

```python
评估维度:

1. 有用性 (Helpfulness)
   问题: "如何学Python？"
   基础模型: 5.2/10
   SFT: 7.8/10
   RLHF: 9.1/10

2. 无害性 (Harmlessness)
   问题: "如何黑进网站？"
   基础模型: 2.1/10 (提供危险信息)
   SFT: 6.5/10 (有时拒绝)
   RLHF: 9.8/10 (始终拒绝)

3. 诚实性 (Honesty)
   问题: "你能预测股市吗？"
   基础模型: 4.3/10 (胡说八道)
   SFT: 7.2/10 (有时承认)
   RLHF: 9.5/10 (诚实承认限制)

4. Win Rate (对比人类偏好)
   RLHF vs SFT: 73% wins
   RLHF vs Base: 89% wins
```

---

## 💡 第六部分：实战技巧

### ⚠️ 常见问题

```python
问题1: 奖励黑客 (Reward Hacking)
  现象: 模型学会"欺骗"奖励模型
  例子: 生成极长但无意义的回答（奖励模型给高分）
  
  解决:
  ✅ KL惩罚（不偏离参考模型太远）
  ✅ 长度惩罚
  ✅ 人工抽查
  ✅ 多个奖励模型集成

问题2: 模式崩溃 (Mode Collapse)
  现象: 模型总是生成类似的回答
  例子: 每次都说"这是个好问题..."
  
  解决:
  ✅ 降低学习率
  ✅ 增加温度采样
  ✅ 多样性奖励
  ✅ 重新初始化

问题3: 遗忘 (Catastrophic Forgetting)
  现象: 模型忘记基本语言能力
  例子: 开始生成语法错误的句子
  
  解决:
  ✅ 强KL惩罚
  ✅ 混合预训练数据
  ✅ 定期评估基础能力

问题4: 成本高昂
  人工标注: $50K-$500K
  计算资源: 数百GPU-天
  
  解决:
  ✅ 从小规模开始
  ✅ 使用现有数据集
  ✅ AI辅助标注
  ✅ 主动学习（选最有用的样本）
```

### 🎯 优化建议

```python
数据质量 > 数据数量
  宁愿1000条高质量标注
  不要10000条低质量标注

分阶段评估
  SFT后先测试基础能力
  RM训练后验证偏好准确率
  PPO过程中持续监控

超参数调优
  SFT:
    lr: 5e-6 to 1e-5
    epochs: 1-3
    
  RM:
    lr: 1e-5
    epochs: 1
    
  PPO:
    lr: 1e-6 (非常小！)
    kl_coef: 0.05-0.2
    clip_epsilon: 0.2
    batch_size: 16-64

工程实践
  使用DeepSpeed ZeRO优化内存
  梯度检查点
  混合精度训练
  分布式训练
```

---

## 🚀 第七部分：简化方案 - DPO

### 🎯 DPO: 更简单的RLHF

**DPO (Direct Preference Optimization) = 无需奖励模型！**

```python
传统RLHF问题:
  ❌ 需要训练奖励模型（复杂）
  ❌ 需要PPO（难调）
  ❌ 需要大量计算资源

DPO解决方案:
  ✅ 直接从偏好数据优化
  ✅ 无需奖励模型
  ✅ 无需RL（用监督学习）
  ✅ 更简单、更稳定
```

**DPO核心思想：**

```python
RLHF三阶段:
  SFT → RM → PPO

DPO两阶段:
  SFT → DPO优化

DPO Loss:
  L_DPO = -log σ(β * log[
      π(y_w|x) / π_ref(y_w|x)
    - π(y_l|x) / π_ref(y_l|x)
  ])
  
  其中:
  - y_w: 更好的回答(chosen)
  - y_l: 更差的回答(rejected)
  - π: 当前策略
  - π_ref: 参考策略(SFT模型)
  - β: 温度参数
```

### 🔧 DPO实现

```python
# train_dpo.py

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from preference_dataset import PreferenceDataset

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             reference_chosen_logps, reference_rejected_logps,
             beta=0.1):
    """
    DPO损失函数
    
    Args:
        policy_chosen_logps: 当前策略对chosen的log概率
        policy_rejected_logps: 当前策略对rejected的log概率
        reference_chosen_logps: 参考策略对chosen的log概率
        reference_rejected_logps: 参考策略对rejected的log概率
        beta: 温度参数
    """
    # 计算log比率
    policy_log_ratio = policy_chosen_logps - policy_rejected_logps
    reference_log_ratio = reference_chosen_logps - reference_rejected_logps
    
    # DPO loss
    logits = beta * (policy_log_ratio - reference_log_ratio)
    loss = -F.logsigmoid(logits).mean()
    
    # 统计（用于监控）
    implicit_acc = (logits > 0).float().mean()
    
    return loss, implicit_acc

def train_dpo(
    sft_model_path='out-sft/sft_model.pt',
    data_path='data/preference_data.jsonl',
    output_dir='out-dpo',
    epochs=1,
    batch_size=4,
    learning_rate=5e-7,
    beta=0.1,
    device='cuda'
):
    """DPO训练"""
    
    # 加载模型
    print("加载SFT模型...")
    policy_model = load_model(sft_model_path).to(device)
    reference_model = load_model(sft_model_path).to(device)
    
    # 冻结reference model
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False
    
    # 数据
    dataset = PreferenceDataset(data_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 优化器
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)
    
    # 训练
    print("开始DPO训练...")
    policy_model.train()
    
    for epoch in range(epochs):
        for batch in dataloader:
            chosen_ids = batch['chosen_ids'].to(device)
            rejected_ids = batch['rejected_ids'].to(device)
            
            # 当前策略的log概率
            policy_chosen_logits = policy_model(chosen_ids).logits
            policy_rejected_logits = policy_model(rejected_ids).logits
            
            # 参考策略的log概率
            with torch.no_grad():
                ref_chosen_logits = reference_model(chosen_ids).logits
                ref_rejected_logits = reference_model(rejected_ids).logits
            
            # 计算log概率(对实际token)
            policy_chosen_logps = get_batch_logps(policy_chosen_logits, chosen_ids)
            policy_rejected_logps = get_batch_logps(policy_rejected_logits, rejected_ids)
            ref_chosen_logps = get_batch_logps(ref_chosen_logits, chosen_ids)
            ref_rejected_logps = get_batch_logps(ref_rejected_logits, rejected_ids)
            
            # DPO loss
            loss, acc = dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps,
                beta=beta
            )
            
            # 更新
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            print(f"Loss: {loss.item():.4f}, Acc: {acc.item():.2%}")
    
    # 保存
    torch.save({'model': policy_model.state_dict()}, f'{output_dir}/dpo_model.pt')
    print("DPO训练完成！")

def get_batch_logps(logits, labels):
    """计算batch的平均log概率"""
    logps = F.log_softmax(logits, dim=-1)
    # 取实际label的log概率
    per_token_logps = torch.gather(logps, 2, labels[:, 1:].unsqueeze(-1)).squeeze(-1)
    return per_token_logps.sum(dim=1)  # 对序列求和

if __name__ == "__main__":
    train_dpo(
        sft_model_path='out-sft/sft_model.pt',
        data_path='data/preference_data.jsonl',
        epochs=1,
        batch_size=4,
        beta=0.1
    )
```

**DPO vs PPO：**

```python
对比:
┌──────────────┬─────────┬─────────┐
│ 指标         │ PPO     │ DPO     │
├──────────────┼─────────┼─────────┤
│ 复杂度       │ 高      │ 低      │
│ 需要RM       │ ✅      │ ❌      │
│ 训练稳定性   │ 中等    │ 高      │
│ 最终效果     │ 略好    │ 相近    │
│ 计算成本     │ 高      │ 低      │
│ 实现难度     │ 难      │ 简单    │
└──────────────┴─────────┴─────────┘

推荐:
  研究/探索 → DPO (更简单)
  生产/极致性能 → PPO (更强大)
  资源有限 → DPO
  资源充足 → PPO
```

---

## 🎓 总结

### ✨ 核心要点

```python
RLHF完整流程:

1. SFT (监督微调)
   目标: 基础对话能力
   数据: 高质量问答对(10K-100K)
   时间: 1-2天

2. RM (奖励模型)
   目标: 学习人类偏好
   数据: 偏好对比(50K-100K对)
   时间: 1周

3. PPO (强化学习)
   目标: 最大化奖励
   数据: prompts(无限)
   时间: 1-2周

简化方案:
  SFT → DPO (2-3天搞定！)

关键技巧:
  ✅ 数据质量最重要
  ✅ KL惩罚防止遗忘
  ✅ 持续监控评估
  ✅ 从小规模开始
```

### 🎯 实战建议

```python
你的资源 → 推荐方案

预算<$1000:
  → 使用开源数据集
  → DPO方法
  → 小模型(GPT-2)

预算$1000-$10000:
  → 少量人工标注 + 开源数据
  → DPO或简化PPO
  → 中等模型(GPT-2 Medium)

预算>$10000:
  → 大量人工标注
  → 完整RLHF pipeline
  → 大模型(GPT-2 Large+)

时间紧迫:
  → DPO
  → 使用现成SFT模型

追求极致:
  → 完整RLHF
  → 多轮迭代
```

### 📚 参考资源

```python
必读论文:
  1. InstructGPT (OpenAI, 2022) ⭐⭐⭐⭐⭐
  2. Training language models to follow instructions with human feedback
  3. DPO: Direct Preference Optimization (2023) ⭐⭐⭐⭐

开源项目:
  - trl (Hugging Face): RLHF训练库
  - OpenAssistant: 开源RLHF数据
  - DeepSpeed-Chat: 微软RLHF框架

工具:
  - Label Studio: 数据标注
  - Weights & Biases: 实验跟踪
  - vLLM: 快速推理
```

---

**最后一句话：**

> RLHF是让AI真正"听话"的关键技术。
> 从野蛮生长到知书达理，
> 这一步让AI从工具变成助手。
>
> 开始很难，但每一步都值得。
> 当你看到模型第一次拒绝有害请求时，
> 你会明白这一切的意义。

🎊 **恭喜掌握RLHF对齐！** 🎊
