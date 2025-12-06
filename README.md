# BioBERT Drug Sentiment Classification Model

## Model Information
- **Base Model**: dmis-lab/biobert-base-cased-v1.2
- **Task**: 3-class sentiment classification (Negative, Neutral, Positive)
- **Training Data**: Drug reviews from drugLib dataset
- **Training Samples**: 2900
- **Test Samples**: 622

## Performance Metrics
- **Test Accuracy**: 0.6913 (69.13%)
- **Test F1 (Macro)**: 0.6533
- **Training Time**: 6.53 minutes

## Training Configuration
- **Epochs**: 3
- **Learning Rate**: 2e-05
- **Batch Size**: 16
- **Optimizer**: AdamW with weight decay = 0.01
- **Scheduler**: Linear warmup + decay
- **Max Sequence Length**: 256 tokens

## Novel Components
1. Fine-tuning pre-trained BioBERT (medical domain knowledge)
2. AdamW optimizer with decoupled weight decay
3. Linear warmup learning rate scheduling
4. Class-weighted loss for imbalanced data

## Usage
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained('output/biobert_drug_sentiment')
tokenizer = AutoTokenizer.from_pretrained('output/biobert_drug_sentiment')

# Predict on new review
text = "This medication works great with no side effects!"
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256)
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits, dim=1).item()
# 0=Negative, 1=Neutral, 2=Positive
```

## Per-Class Performance
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Negative | 0.7557 | 0.7333 | 0.7444 | 135 |
| Neutral | 0.4025 | 0.4672 | 0.4324 | 137 |
| Positive | 0.8042 | 0.7629 | 0.7830 | 350 |

## Citation
If you use this model, please cite:
- BioBERT: Lee et al., 2020 (https://arxiv.org/abs/1901.08746)
- This implementation: CIS 602 Project 4, 2025
