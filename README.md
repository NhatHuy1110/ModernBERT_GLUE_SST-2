# ModernBERT_GLUE_SST-2

## 📌 Introduction  

This repository reproduces and extends experiments from the paper:  
[**"Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference" (Warner et al., 2024).**](https://arxiv.org/pdf/2412.13663)  

The project focuses on:  
- Analyzing the differences and improvements of **ModernBERT** compared to older encoders, especially **SBERT**.  
- Fine-tuning [**ModernBERT-base**](https://huggingface.co/answerdotai/ModernBERT-base) on the **GLUE/SST-2** benchmark to evaluate sentiment classification performance.  

---

## 🔍 Research Background   

### ModernBERT vs SBERT  

| Feature                  | SBERT (2019)                                | ModernBERT (2024) |
|--------------------------|----------------------------------------------|--------------------|
| Architecture             | Encoder-only (BERT base/large)              | Encoder-only (Deep & Narrow, GPU-optimized) |
| Max sequence length      | 512 tokens                                  | 8192 tokens |
| Positional Embedding     | Absolute                                    | Rotary Positional Embedding (RoPE) |
| Attention                | Full self-attention                         | Alternating Global-Local + FlashAttention |
| Tokenizer                | WordPiece (BERT original)                   | Modern BPE, supports text + code |
| Memory efficiency        | Moderate                                    | High (unpadded batching + FlashAttention) |
| Training data            | ~3.3B tokens                                | 2 trillion tokens (web, code, scientific) |
| GLUE (SST-2) performance | ~92–93%                                     | ~94–95% |

**Conclusion:** ModernBERT delivers a **state-of-the-art encoder**, outperforming SBERT and other previous encoders in both accuracy and efficiency.

---

## 🧪 Experiment 

Main notebook: **`modernbert-on-glue-sst-2.ipynb`**  
- Uses HuggingFace Transformers to load ModernBERT-base.  
- Fine-tunes on **SST-2** (Sentiment Analysis).  
- Evaluates accuracy on the validation set.
- Time for training ModernBERT: 7 hours.

### Training steps:  
1. Load GLUE/SST-2 dataset via `datasets`.  
2. Tokenize using the ModernBERT tokenizer.  
3. Fine-tune with HuggingFace `Trainer` using hyperparameters aligned with the paper:  
   - Learning Rate: `8e-5`  
   - Epochs: `20`  
   - Batch size: depends on GPU (see notebook).  
4. Save the fine-tuned model for inference.  

---

## 📊 Results  

- **SBERT (baseline)**: ~92% accuracy.  
- **ModernBERT-base (fine-tuned)**: ~94–95% accuracy (depending on seed & hyperparameters).  

These results align with the original paper, where ModernBERT surpassed DeBERTa-v3 and other strong encoders on GLUE. 

