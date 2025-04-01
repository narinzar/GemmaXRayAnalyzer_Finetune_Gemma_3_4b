# GemmaXRayAnalyzer: Fine-tuning Gemma 3 for Medical X-ray Analysis

This repository contains a complete pipeline for fine-tuning DeepMind's Gemma 3 model for medical X-ray analysis. The pipeline is designed to be modular, efficient, and easy to understand, with each step in its own script.

## 🎯 Project Overview

This project demonstrates how to fine-tune Gemma 3 (4B) on a medical X-ray dataset to improve its ability to analyze and describe radiological findings. It uses LoRA (Low-Rank Adaptation) for efficient fine-tuning with minimal GPU requirements.

## 🛠️ Project Structure

```
.
├── config.py                     # Central configuration
├── .env.example                  # Template for environment variables
├── .gitignore                    # Git ignore file
├── run_all.py                    # Run the entire pipeline
├── load_finetuned_model.py       # Utility for using the model locally
├── use_model_api.py              # Utility for using the model via API
├── app.py                        # Gradio app for interactive testing
├── requirements_space.txt        # Requirements for HF Space
├── push_to_space.py              # Script to push to HF Space
├── huggingface-space-config.yml  # Configuration for HF Space
├── 01_data_load.py               # Load and split dataset
├── 02_data_analysis.py           # Exploratory data analysis
├── 03_model_load.py              # Load the base model
├── 04_lora_config.py             # Configure LoRA parameters
├── 05_prepare_training.py        # Prepare datasets for training
├── 06_train_model.py             # Execute training
├── 07_evaluate_model.py          # Evaluate and compare models
└── 08_push_to_hf.py              # Push to Hugging Face Hub
```

## ⚙️ Setup and Installation

1. Clone this repository
```bash
git clone https://github.com/yourusername/GemmaXRayAnalyzer_Finetune_Gemma_3_4b.git
cd GemmaXRayAnalyzer_Finetune_Gemma_3_4b
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up environment variables
```bash
cp .env.example .env
# Edit .env with your Hugging Face credentials
```

## 🚀 Running the Pipeline

### Option 1: Run the entire pipeline
```bash
python run_all.py
```

### Option 2: Run specific steps
```bash
# Run from step 3 to step 6
python run_all.py --start-step 3 --end-step 6

# Skip evaluation and pushing to HF
python run_all.py --skip-evaluation --skip-push
```

### Option 3: Run steps individually
```bash
python 01_data_load.py
python 02_data_analysis.py
# ... and so on
```

## 📊 Dataset and Analysis

The pipeline includes comprehensive exploratory data analysis, which generates:

- Caption length distribution chart
- Common medical terms visualization
- Word cloud of X-ray descriptions
- Statistical analysis of vocabulary

Results are saved to the `figures/` and `results/` directories.

## 📋 Evaluation

The pipeline evaluates the model before and after fine-tuning using:

- ROUGE scores (R1, R2, Rouge-L)
- Generation time comparison
- Side-by-side output comparisons

Evaluation results are saved to the `results/` directory.

## 🤗 Using the Fine-tuned Model

### Using the Local Model

#### Through the utility script
```bash
# Interactive mode
python load_finetuned_model.py --interactive

# Single prompt
python load_finetuned_model.py --prompt "Analyze this chest X-ray showing opacity in the lower right lung"
```

#### In your own code
```python
from unsloth import FastModel
from transformers import AutoTokenizer

# Load the model
model = FastModel.from_pretrained("models/gemma_xray_model")
tokenizer = AutoTokenizer.from_pretrained("models/gemma_xray_model")

# Format input
prompt = "Analyze this chest X-ray showing opacity in the lower right lung"
formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

# Generate response
inputs = tokenizer(formatted_prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Using the Model from Hugging Face Hub

#### Via the API utility
```bash
# Using the Inference API
python use_model_api.py --method api --prompt "Analyze this chest X-ray showing opacity in the lower right lung"

# Loading the model directly
python use_model_api.py --method direct --prompt "Analyze this chest X-ray showing opacity in the lower right lung"
```

#### In Python code
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load from HF Hub
model_id = "your-username/GemmaXRayAnalyzer_Finetune_Gemma_3_4b"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Format input
prompt = "Analyze this chest X-ray showing opacity in the lower right lung"
formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

# Generate response
inputs = tokenizer(formatted_prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7, top_p=0.9)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

#### Via the Hugging Face Inference API (REST)
```python
import requests

API_URL = "https://api-inference.huggingface.co/models/your-username/GemmaXRayAnalyzer_Finetune_Gemma_3_4b"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

output = query({
    "inputs": "<start_of_turn>user\nAnalyze this chest X-ray showing opacity in the lower right lung<end_of_turn>\n<start_of_turn>model\n",
    "parameters": {"max_new_tokens": 256, "temperature": 0.7, "top_p": 0.9}
})
```

## 🌐 Deploying to Hugging Face Spaces

### Automatic Deployment with Script

The included script makes it easy to deploy your Gradio app to Hugging Face Spaces:

```bash
python push_to_space.py
```

This will automatically:
1. Use your credentials from the `.env` file
2. Upload the necessary files to your Space
3. Configure the Space with your model

### Manual Deployment

1. Create a Space on Hugging Face (if not already created):
```bash
huggingface-cli repo create your-space-name --type space
```

2. Clone the Space repository:
```bash
git clone https://huggingface.co/spaces/your-username/your-space-name
cd your-space-name
```

3. Copy the necessary files:
```bash
cp ../app.py ../huggingface-space-config.yml ../requirements_space.txt ./
```

4. Push to Hugging Face:
```bash
git add .
git commit -m "Add X-ray analyzer app"
git push
```

## 📱 Gradio App

To test the model interactively, you can run the included Gradio app:

```bash
python app.py
```

This will launch a web interface where you can input prompts and generate analyses.

## ⚠️ Disclaimer

This project is for educational and research purposes only. The fine-tuned model should not be used for actual medical diagnosis. Always consult qualified healthcare professionals for medical advice.

## 🙏 Acknowledgments

- DeepMind for creating the Gemma 3 model
- Unsloth for providing efficient fine-tuning tools
- The medical imaging community for advancing the field

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.