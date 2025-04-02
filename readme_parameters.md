# GemmaXRayAnalyzer: Model Parameters and Usage Guide

This document provides detailed information about the parameters and recommended usage for the GemmaXRayAnalyzer model.

## Model Loading Parameters

When loading the model, you can use the following parameters:

```python
model, tokenizer = FastModel.from_pretrained(
    model_name="YOUR_USERNAME/GemmaXRayAnalyzer_Finetune_Gemma_3_4b",
    max_seq_length=2048,        # Maximum sequence length (adjust based on needs)
    dtype=None,                 # Auto-detect best dtype
    load_in_4bit=True,          # Use 4-bit quantization (for GPU memory efficiency)
    device_map="auto"           # Automatically distribute model across available devices
)
```

### Key Loading Parameters:

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `max_seq_length` | Maximum sequence length | 2048 (or 4096 if memory allows) |
| `load_in_4bit` | Use 4-bit quantization | `True` for GPU, `False` for CPU |
| `device_map` | Device allocation strategy | "auto" |
| `dtype` | Data type for model weights | `None` (auto-detect) |

## Generation Parameters

For optimal results when generating text with the model, we recommend the following parameters:

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=256,         # Adjust based on desired response length
    temperature=0.7,            # Controls randomness (0.5-0.8 recommended)
    top_p=0.9,                  # Nucleus sampling parameter
    top_k=50,                   # Top-k sampling parameter
    repetition_penalty=1.1,     # Helps avoid repetitive text
    do_sample=True              # Enable sampling (vs greedy decoding)
)
```

### Key Generation Parameters:

| Parameter | Description | Range | Recommended |
|-----------|-------------|-------|-------------|
| `max_new_tokens` | Maximum length of generated response | 64-512 | 256 |
| `temperature` | Randomness of generation | 0.1-1.5 | 0.7 |
| `top_p` | Nucleus sampling (probability cutoff) | 0.5-1.0 | 0.9 |
| `top_k` | Limits vocabulary to top K tokens | 10-100 | 50 |
| `repetition_penalty` | Penalizes repetition | 1.0-1.2 | 1.1 |

### Parameter Selection Guide:

- **For deterministic/consistent responses**: Lower temperature (0.3-0.5)
- **For creative/varied responses**: Higher temperature (0.7-1.0)
- **For detailed medical analyses**: Use longer `max_new_tokens` (256-384)
- **For concise summaries**: Use shorter `max_new_tokens` (128-192)

## Prompt Formatting

The model works best with prompts formatted using Gemma's chat template. Here are example formats:

### Basic Format:

```
<start_of_turn>model

```

## Example Prompt Templates

### Clinical Case Analysis:

```
<start_of_turn>user
You are an expert radiologist. Analyze this chest X-ray of a 65-year-old male with a history of smoking, presenting with shortness of breath and cough for 3 days. The X-ray shows increased opacity in the right lower lobe with some air bronchograms visible.
<end_of_turn>
<start_of_turn>model
```

### Educational Explanation:

```
<start_of_turn>user
You are an expert radiologist teaching medical students. Explain what characteristics differentiate bacterial pneumonia from viral pneumonia on a chest X-ray. Describe the typical presentation of each.
<end_of_turn>
<start_of_turn>model
```

### Comparative Analysis:

```
<start_of_turn>user
You are an expert radiologist. Compare the findings in a chest X-ray showing pulmonary edema versus one showing pneumonia. What are the key differentiating features?
<end_of_turn>
<start_of_turn>model
```

## Memory Usage Optimization

When using the model on resource-constrained systems, consider these optimizations:

1. **Reduce Sequence Length**: Lower `max_seq_length` to reduce memory usage
2. **Use 4-bit Quantization**: Enable `load_in_4bit=True` for significant memory savings
3. **CPU Offloading**: Use `device_map="auto"` to offload parts of the model to CPU when needed
4. **Batch Size**: Use smaller batch sizes for inference (`batch_size=1`)

## Performance Benchmarks

The model has been tested on various GPU configurations with the following performance characteristics:

| Hardware | Loading Time | Inference Time (256 tokens) | Memory Usage |
|----------|-------------|----------------------------|--------------|
| NVIDIA A100 | ~10s | ~1.5s | ~8 GB |
| NVIDIA T4 | ~15s | ~3.0s | ~6 GB |
| CPU (16 cores) | ~40s | ~15.0s | ~12 GB |

## Command-Line Usage (10_load_finetuned.py)

The included utility script supports these parameters:

```bash
# Interactive mode
python 10_load_finetuned.py --interactive --temperature 0.7

# Single prompt
python 10_load_finetuned.py --prompt "Analyze this chest X-ray showing a small pleural effusion on the right side."

# Load from Hugging Face Hub
python 10_load_finetuned.py --hub YOUR_USERNAME/GemmaXRayAnalyzer_Finetune_Gemma_3_4b --max-tokens 384

# With system prompt
python 10_load_finetuned.py --interactive --system-prompt "You are an expert pediatric radiologist specializing in chest X-rays."
```

## API Usage (11_use_model_api.py)

Parameters for using the model via the Hugging Face Inference API:

```bash
# Interactive API session
python 11_use_model_api.py --interactive --model-id YOUR_USERNAME/GemmaXRayAnalyzer_Finetune_Gemma_3_4b

# Single prompt via API
python 11_use_model_api.py --prompt "Analyze this chest X-ray showing cardiomegaly." --temperature 0.5
```

## Fine-tuning Parameters

If you wish to further fine-tune the model, these parameters were used in the original training and are recommended:

```python
training_args = SFTConfig(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    max_steps=50,
    learning_rate=2e-4,
    fp16=is_bf16_supported(),  # Use bf16 if available
    bf16=is_bf16_supported(),
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=10,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    warmup_steps=5,
)
```

## Limitations and Best Practices

- **Not for Clinical Use**: This model is for research and educational purposes only
- **Verify Outputs**: Always have a medical professional verify any generated analyses
- **Complex Cases**: The model may struggle with rare or complex pathologies
- **Prompt Design**: Be specific in your prompts about the features visible in the X-ray
- **Temperature Tuning**: Lower temperatures (0.3-0.5) provide more conservative and factual responses

## Ethical Considerations

When using this model, please adhere to these ethical guidelines:

1. Do not present the model's outputs as medical advice
2. Clearly distinguish between AI-generated content and human expert analyses
3. Respect patient privacy and confidentiality
4. Consider potential biases in the model's training data
5. Use the model as an assistive tool, not a replacement for professional expertise
turn>user
You are an expert radiologist. Analyze this chest X-ray showing [description of X-ray findings].
<end_of_turn>
<start_of_turn>model
```

### With System Prompt:

```
<start_of_turn>system
You are an expert radiologist with 20 years of experience in chest radiology. Provide detailed, accurate analyses of X-ray images.
<end_of_turn>
<start_of_turn>user
Analyze this chest X-ray showing [description of X-ray findings].
<end_of_turn>
<start_of_