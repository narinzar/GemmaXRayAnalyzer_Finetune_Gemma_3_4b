# GemmaXRayAnalyzer: Fine-tuned Gemma 3 for X-ray Analysis

This model is a fine-tuned version of [unsloth/gemma-3-4b-it](https://huggingface.co/unsloth/gemma-3-4b-it) specifically optimized for analyzing and describing medical X-ray images. It leverages LoRA (Low-Rank Adaptation) to efficiently adapt the model's capabilities to the medical imaging domain.

## Model Description

GemmaXRayAnalyzer was trained to provide detailed, medically accurate descriptions of X-ray images. It can identify common radiological findings, describe anatomical structures, and suggest potential conditions based on the image characteristics described in the prompt.

### Model Architecture

- **Base Model**: Gemma 3 4B (Instruction Tuned)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **LoRA Parameters**:
  - Rank: 16
  - Alpha: 16
  - Dropout: 0
  - Target Modules: Language layers, attention modules, MLP modules

### Training Data

This model was fine-tuned on the [unsloth/Radiology_mini](https://huggingface.co/datasets/unsloth/Radiology_mini) dataset, which contains a collection of X-ray images paired with radiological descriptions.

The dataset includes various types of X-ray images with corresponding professional radiological assessments, covering a wide range of anatomical structures and pathological conditions.

## Intended Use

This model is designed for:

- Medical education and training
- Assisting radiologists with preliminary analyses
- Research in medical NLP and diagnostic AI
- Development of medical imaging tools

### Limitations and Biases

- This model is **NOT intended for clinical use** and should not be used for making medical diagnoses without professional oversight.
- The model may have limited knowledge of rare conditions or specialized imaging techniques.
- The model may reflect biases present in its training data, including potential geographic, demographic, or institutional biases in radiological practice.
- This model is not a replacement for professional medical advice or diagnosis.

## How to Use the Model

You can use this model to generate radiological descriptions based on X-ray image descriptions:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_id = "YOUR_USERNAME/GemmaXRayAnalyzer_Finetune_Gemma_3_4b"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Create a prompt that describes an X-ray image
prompt = """<start_of_turn>user
You are an expert radiologist. Analyze this chest X-ray showing increased opacity in the right lower lobe.
<end_of_turn>
<start_of_turn>model
"""

# Generate a response
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7, top_p=0.9)
response = tokenizer.batch_decode(outputs)[0]
print(response)
```

### Sample Input/Output

**Input:**
```
You are an expert radiologist. Analyze this chest X-ray showing increased opacity in the right lower lobe.
```

**Output:**
```
The chest X-ray demonstrates an area of increased opacity in the right lower lobe, which appears as a patchy consolidation with air bronchograms visible within it. This finding is most consistent with pneumonia, particularly bacterial pneumonia given the lobar distribution. 

The remainder of the lung fields appears clear without evidence of pleural effusion, pneumothorax, or pulmonary edema. The cardiac silhouette is normal in size, and the mediastinum is not widened. No bony abnormalities are evident.

Impression: Right lower lobe consolidation, most likely representing pneumonia. Clinical correlation and follow-up imaging after treatment would be recommended to ensure resolution.
```

## Training Details

- **Training Process**: The model was fine-tuned using the Unsloth library for efficient training
- **Optimization**: AdamW 8-bit optimizer
- **Learning Rate**: 2e-4 with linear scheduler
- **Batch Size**: 2 per device with gradient accumulation steps of 4
- **Training Steps**: Approximately 50 steps

## Citation

If you use this model in your research or applications, please cite:

```
@misc{gemmaXrayAnalyzer2025,
  author = {Your Name},
  title = {GemmaXRayAnalyzer: Fine-tuned Gemma 3 for X-ray Analysis},
  year = {2025},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/YOUR_USERNAME/GemmaXRayAnalyzer_Finetune_Gemma_3_4b}}
}
```

### Dataset Citation

This model was fine-tuned on the Radiology_mini dataset. Please also cite:

```
@misc{unslothRadiologyMini2024,
  author = {Unsloth Team},
  title = {Radiology_mini: A Dataset for Medical X-ray Image Analysis},
  year = {2024},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/datasets/unsloth/Radiology_mini}}
}
```

## Ethical Considerations

This model is intended for research, educational purposes, and assisted diagnosis only. It should not be used as the sole basis for clinical decisions. Always consult with qualified healthcare professionals for medical advice and diagnosis.

## License

This model is subject to the same license as the base Gemma 3 model. Please refer to the [Gemma 3 license](https://huggingface.co/unsloth/gemma-3-4b-it) for details.

## Acknowledgments

- DeepMind for creating the Gemma 3 model
- Unsloth team for providing efficient fine-tuning tools
- Hugging Face for the infrastructure to share models
- The creators of the Radiology_mini dataset
