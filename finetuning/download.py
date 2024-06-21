from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "jak6/Phi3_medium_4k_finetuned"
current_directory = "."

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model.save_pretrained(current_directory)
tokenizer.save_pretrained(current_directory)
