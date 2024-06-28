# Import the Libraries
import torch
import pandas as pd
from datasets import Dataset

from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import SFTTrainer, SFTConfig

import wandb
from huggingface_hub import login
from dotenv import load_dotenv
import os

# Set Global Parameters

# Model and dataset identifiers
model_id = "microsoft/Phi-3-medium-128k-instruct" 
model_name = "microsoft/Phi-3-medium-128k-instruct" 
dataset_name = "/home/llm-server/jak/cook/subsections.csv"

# Dataset split and new model name
dataset_split = "train"
new_model = "phi3_medium_finetuned"

# Hugging Face repository details
hf_model_repo = "jak6/" + new_model

# Device and quantization settings
device_map = {"": 0}
use_4bit = True
bnb_4bit_compute_dtype = "bfloat16"
bnb_4bit_quant_type = "nf4"
use_double_quant = True

# LoRA configuration
lora_r = 16
lora_alpha = 16
lora_dropout = 0.05
target_modules = ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]

# Set random seed for reproducibility
set_seed(1234)

# Connect to Hugging Face Hub
load_dotenv()
login(token=os.getenv("HF_HUB_TOKEN"))

# Load the Dataset

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(dataset_name)

# Convert the DataFrame to a Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Display the size and a random sample from the dataset
# print(f"dataset size: {len(dataset)}")
# print(dataset[randrange(len(dataset))])

# Load the Tokenizer

# Load the tokenizer
tokenizer_id = model_id
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
tokenizer.padding_side = 'right'  # to prevent warnings

# Print tokenizer information to verify it loaded correctly
# print(tokenizer)

# Generate the Suitable Format for the Model
def create_message_column(row):
    messages = []
    user = {
        "content": f"{row['instruction']}\n Input: {row['input']}",
        "role": "user"
    }
    messages.append(user)
    assistant = {
        "content": f"{row['output']}",
        "role": "assistant"
    }
    messages.append(assistant)
    return {"messages": messages}

def format_dataset_chatml(row):
    return {"text": tokenizer.apply_chat_template(row["messages"], add_generation_prompt=False, tokenize=False)}

# Apply the create_message_column function to the dataset
dataset_chatml = dataset.map(create_message_column)

# Apply the format_dataset_chatml function to the dataset
dataset_chatml = dataset_chatml.map(format_dataset_chatml)

# Check the first example in the transformed dataset
# print(dataset_chatml[0])

# Split the dataset into training and test sets
dataset_chatml = dataset_chatml.train_test_split(test_size=0.05, seed=1234)

# Display the structure of the training set and the test set
# print(dataset_chatml)

#Instruction fine-tune a Phi-3-mini model using QLORA and trl

# Check if BF16 is supported on the current GPU
if torch.cuda.is_bf16_supported():
    compute_dtype = torch.bfloat16
    attn_implementation = 'flash_attention_2'
else:
    compute_dtype = torch.float16
    attn_implementation = 'sdpa'

# Print the chosen attention implementation and compute data type
# print(attn_implementation)
# print(compute_dtype)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, add_eos_token=True, use_fast=True)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = 'left'  # Set padding side to 'left' to prevent warnings

# Set up Bits and Bytes configuration for quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_double_quant,
)

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=compute_dtype,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map=device_map,
    attn_implementation=attn_implementation
)

# Prepare the model for k-bit training
model = prepare_model_for_kbit_training(model)

#Establish a connection with wandb and enlist the project and experiment.

# Log in to your wandb account. If not logged in, it will prompt for the API key.
wandb.login(relogin=True)

# Use the same name as the Hugging Face repository
hf_model_repo = "jak6/phi3_medium_finetuned"

# Extract project name from the Hugging Face repository name
project_name = hf_model_repo.split('/')[1]

# Set the environment variable for the project name
os.environ["PROJECT"] = project_name

# Initialize a new wandb run with the specified project name
wandb.init(project=project_name, name="Phi-3-medium-QLoRA")

# Define the LoRA configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj']
)

# Define the training arguments
args = SFTConfig(
    output_dir="./results_medium", # The output directory where the model predictions and checkpoints will be written.
    overwrite_output_dir=True, # Overwrite the content of the output directory if it exists.
    evaluation_strategy="steps", # Evaluation will be performed after a certain number of training steps.
    do_train=True, # Whether to run training.
    do_eval=True, # Whether to run evaluation on the dev set.
    optim="adamw_torch", # The AdamW optimizer from PyTorch
    per_device_train_batch_size=2, # Batch size per device for training.
    per_device_eval_batch_size=2, # Batch size per device for evaluation.
    gradient_accumulation_steps=4, # Number of updates steps to accumulate the gradients for before performing a backward/update pass.
    log_level="debug", # log_level' is set to "debug", meaning that all log messages will be printed.
    save_strategy="epoch", # 'save_strategy' is set to "epoch", meaning that the model will be saved after each epoch.
    logging_steps=100, # Number of update steps between two logs.
    learning_rate=1e-4, # The initial learning rate for Adam.
    fp16 = not torch.cuda.is_bf16_supported(), # Is set to the opposite of whether bfloat16 is supported on the current CUDA device.
    bf16 = torch.cuda.is_bf16_supported(), # Is set to whether bfloat16 is supported on the current CUDA device.
    eval_steps=100, # Number of update steps between two evaluations.
    num_train_epochs=3, # Total number of training epochs to perform.
    warmup_ratio=0.1, # Is set to 0.1, meaning that 10% of the total training steps will be used for the warmup phase.
    lr_scheduler_type="linear", # Is set to "linear", meaning that a linear learning rate scheduler will be used.
    report_to="wandb", # Report to Weights & Biases.
    seed=42, # Random seed for initialization.
    
)

# Construct the SFTTrainer
trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_chatml['train'],
        eval_dataset=dataset_chatml['test'],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=args,
)

# Train the model and save it locally
trainer.train()
trainer.save_model()

# Define and push the repository name for the Hugging Face model adapter
hf_adapter_repo="jak6/phi3_medium_finetuned"
trainer.push_to_hub(hf_adapter_repo)

# Empty VRAM after training
# del model
# del trainer
# import gc
# gc.collect()
# gc.collect()
# torch.cuda.empty_cache()
