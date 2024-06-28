from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Define model paths and repository names
hf_adapter_repo = "jak6/results_Phi3_medium_4k"
base_model_name = "microsoft/Phi-3-medium-4k-instruct"
compute_dtype = torch.bfloat16

# Load the base model and apply the adapter weights
model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True, torch_dtype=compute_dtype)
model = PeftModel.from_pretrained(model, hf_adapter_repo)
model = model.merge_and_unload()


# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(hf_adapter_repo)

# Save model and tokenizer
model.save_pretrained("merged_model", trust_remote_code=True, safe_serialization=True)
tokenizer.save_pretrained("merged_model")

# Save the merged model and tokenizer to the Hugging Face Model Hub
merged_model_id = "jak6/Phi3_medium_4k_finetuned"
model.push_to_hub(merged_model_id)
tokenizer.push_to_hub(merged_model_id)
