# 🌟 [ICCV 2025] Few-Shot Image Quality Assessment via Adaptation of Vision-Language Models

<div align="center">

![GRMP-IQA Overview](https://tc.z.wiki/autoupload/f/D-USErAe7s9fQrmIZD6eFLN2XAfHAV9CE7VqFrO7cHayl5f0KlZfm6UsKj-HyTuv/20250723/Q4Ew/14657X5000/framework_final_0413.jpg)

*Figure 1: Overview of our GRMP-IQA framework. (a) Pre-training stage: Meta-Prompt Pre-training Module; (b) Fine-tuning stage: Quality-Aware Gradient Regularization*

[![GitHub stars](https://img.shields.io/github/stars/LXDxmu/GRMP-IQA.svg?style=social&label=Star)](https://github.com/LXDxmu/GRMP-IQA)
[![arXiv](https://img.shields.io/arxiv/2409.05381?style=for-the-badge)](https://arxiv.org/abs/2409.05381)
[![Python](https://img.shields.io/badge/Python-3.8+-red.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-orange.svg)](https://pytorch.org/)
[![CLIP](https://img.shields.io/badge/CLIP-ViT--B%2F16-green.svg)](https://github.com/openai/CLIP)

</div>

## 📖 Introduction

This repository contains the official open-source code implementation for the paper **"Few-Shot Image Quality Assessment via Adaptation of Vision-Language Models"** (ICCV 2025).

We propose **GRMP-IQA**, a few-shot image quality assessment framework based on vision-language model adaptation. Our method achieves superior IQA performance on new datasets using only a small number of labeled samples through meta-learning pre-training and quality-aware gradient regularization.

## 🛠️ Environment Setup

```bash
# Create virtual environment
conda create -n grmp_iqa python=3.8
conda activate grmp_iqa

# Install dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
ICCV_opensource_code/
├── README.md                 # Project documentation
├── requirements.txt          # 🛠️ Environment dependencies
├── pretrain.py              # 🔥 Meta-learning pre-training script
├── finetune.py              # 🎯 Few-shot fine-tuning script  
├── logger.py                # Logging utility
├── CLIP/                    # 📚 CLIP model related code
│   ├── clip.py             # CLIP core implementation
│   ├── model.py            # Model architecture definition
│   └── simple_tokenizer.py # Text tokenizer
├── livew_244.mat           # 📊 CLIVE dataset
├── Koniq_244.mat           # 📊 KonIQ dataset
└── model_checkpoint/        # 💾 Pre-trained model checkpoints
```

## 🚀 Quick Start

### Step 1: Data Preparation 📊

1. **Download Datasets**:
- [CLIVE](https://live.ece.utexas.edu/research/ChallengeDB/index.html) 
 - [KonIQ-10K](http://database.mmsp-kn.de/koniq-10k-database.html)
 - [PIPAL](https://www.jasongt.com/projectpages/pipal.html)

2. **Data Preprocessing**:
   ```bash
   # Data has been preprocessed into .mat format, ready to use
   # livew_244.mat - CLIVE dataset (244x244 resolution)
   # Koniq_244.mat - KonIQ dataset (244x244 resolution)
   ```

### Step 2: Meta-Learning Pre-training 🎓

```bash
# Run meta-learning pre-training (on TID2013 and KADID-10K)
python pretrain.py
```
### Step 3: Few-Shot Fine-tuning 🎯

```bash
# 50-shot fine-tuning on CLIVE dataset
python finetune.py --dataset clive --num_image 50 --lda 5.0

# Fine-tuning on KonIQ dataset  
python finetune.py --dataset koniq --num_image 50 --lda 5.0

# Fine-tuning with pre-trained model
python finetune.py --dataset clive --num_image 50 --pretrained --lda 5.0
```

**Fine-tuning Parameters** ⚙️:
- `--dataset`: Target dataset [clive|koniq|pipal]
- `--num_image`: Number of few-shot samples (default: 50)
- `--pretrained`: Whether to use pre-trained model
- `--lda`: Gradient regularization weight (default: 5.0)

## 📚 Citation

If our work is helpful for your research, please consider citing:

```bibtex
@article{li2024boosting,
  title={Few-Shot Image Quality Assessment via Adaptation of Vision-Language Models},
  author={Li, Xudong and Huang, Zihao and Hu, Runze and Zhang, Yan and Cao, Liujuan and Ji, Rongrong},
  journal={arXiv preprint arXiv:2409.05381},
  year={2024}
}
```


## 📄 License

This project is licensed under the MIT License.



## 📞 Contact

For any questions, please feel free to contact us via:

- 📧 Email: [lxd761050753@gmail.com] 
            [huangzihhhh@gmail.com]
- 🐛 Issue: [GitHub Issues](https://github.com/LXDxmu/GRMP-IQA/issues)

---

<div align="center">

**⭐ If this project helps you, please give us a Star! ⭐**
</div>
