# TinyLlama Structured Output LoRA

A QLoRA fine-tuning project that trains TinyLlama to generate structured JSON responses from natural language instructions.

This project demonstrates:
- instruction fine-tuning
- structured output generation
- QLoRA training
- 4-bit quantization
- parameter-efficient fine-tuning using LoRA

 

# Project Overview

Large Language Models often struggle with:
- strict JSON formatting
- schema consistency
- deterministic structured outputs

This project fine-tunes TinyLlama on a transformed version of the Databricks Dolly 15K dataset to improve structured response generation.

The model learns:
```json
{
  "question": "...",
  "context_summary": "...",
  "answer": "...",
  "category": "...",
  "difficulty": "..."
}
```

instead of plain conversational responses.

 

# Base Model

- TinyLlama/TinyLlama-1.1B-Chat-v1.0

 # Fine-Tuning Method

- QLoRA (4-bit quantization)
- PEFT LoRA adapters
- Supervised Fine-Tuning (SFT)

 

# Dataset

Dataset used:
- Databricks Dolly 15K

Original dataset format:
```json
{
  "instruction": "...",
  "context": "...",
  "response": "...",
  "category": "..."
}
```

Transformed into structured instruction-to-JSON format for fine-tuning.

 

# Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT
- TRL
- BitsAndBytes

 

# Hardware

Training was performed on:
- Google Colab T4 GPU (16GB VRAM)

 

# Training Configuration

| Parameter | Value |
| | |
| Quantization | 4-bit NF4 |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Batch Size | 2 |
| Gradient Accumulation | 4 |
| Learning Rate | 2e-4 |
| Epochs | 1 |
| Precision | FP16 |

 

# Repository Structure

```bash
.
├── adapter_config.json
├── adapter_model.safetensors
├── training_notebook.ipynb
├── README.md
```

 

# Example Output

## Input

```text
### Instruction:
Explain recursion

### Response:
```

## Model Output

```json
{
  "question": "Explain recursion",
  "context_summary": "",
  "answer": "Recursion is a programming concept...",
  "category": "education",
  "difficulty": "easy"
}
```

 

# Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

model = PeftModel.from_pretrained(
    base_model,
    "./fine_tuned_model"
)

prompt = """
### Instruction:
Explain recursion

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=120,
    temperature=0.1,
    do_sample=False
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

 

# Key Learning Outcomes

This project demonstrates:
- instruction tuning workflows
- tokenizer pipelines
- dataset preprocessing
- QLoRA training
- PEFT adapter fine-tuning
- structured generation
- LLM deployment workflow

 

# Limitations

- Small model size limits reasoning quality
- JSON generation may occasionally be incomplete
- Optimized for structure rather than factual correctness
- Intended primarily for educational and experimental purposes

 

# Future Improvements

- Better schema enforcement
- JSON validation during inference
- Multi-schema generation
- Function calling style outputs
- Larger base models
- Evaluation metrics pipeline

 

# License

This project follows the license compatibility of the TinyLlama base model and associated libraries.

 

# Author

ABISHEAK
