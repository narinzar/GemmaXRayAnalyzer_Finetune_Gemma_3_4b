# config.py
# Configuration for GemmaXRayAnalyzer_Finetune_Gemma_3_4b project

# Model configuration
MODEL_NAME = "unsloth/gemma-3-4b-it"  # Base Gemma 3 model
DATASET_NAME = "unsloth/Radiology_mini"  # X-ray dataset

# Output repository name
REPO_NAME = "GemmaXRayAnalyzer_Finetune_Gemma_3_4b"
HF_MODEL_NAME = "GemmaXRayAnalyzer_Finetune_Gemma_3_4b"

# Training parameters
MAX_STEPS = 50
LEARNING_RATE = 2e-4
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
SEQUENCE_LENGTH = 2048
USE_FP16 = True
USE_4BIT = True
WARMUP_STEPS = 5
LOGGING_STEPS = 5
EVAL_STEPS = 10
WEIGHT_DECAY = 0.01
OPTIMIZER = "adamw_8bit"

# LoRA parameters
LORA_RANK = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0
LORA_BIAS = "none"
RANDOM_SEED = 3407

# Inference parameters
GENERATION_MAX_NEW_TOKENS = 256
GENERATION_TEMPERATURE = 0.7
GENERATION_TOP_P = 0.9
GENERATION_TOP_K = 50

# Paths
DATA_DIR = "data"
CHECKPOINT_DIR = "checkpoints"
MODEL_OUTPUT_DIR = "models/gemma_xray_model"
RESULTS_DIR = "results"
FIGURES_DIR = "figures"
OUTPUT_DIR = "outputs"

# Evaluation
EVAL_NUM_SAMPLES = 10
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# Prompt template
INSTRUCTION = "You are an expert radiologist. Analyze this X-ray image and describe what you see in detail."
