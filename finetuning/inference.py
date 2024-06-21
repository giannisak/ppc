from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Define the Hugging Face repository and other parameters
hf_model_repo = 'jak6/results'
# hf_model_repo = 'microsoft/Phi-3-mini-4k-instruct'
device_map = {"": 0}
compute_dtype = torch.bfloat16

# Set seed for reproducibility
# set_seed(1234)

# Load the tokenizer and model from the Hugging Face Model Hub
tokenizer = AutoTokenizer.from_pretrained(hf_model_repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(hf_model_repo, trust_remote_code=True, torch_dtype=compute_dtype, device_map=device_map)

# Create a text generation pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Function to test inference
def test_inference(prompt):
    prompt = pipe.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=1024, do_sample=True, num_beams=1, temperature=0.3, top_k=50, top_p=0.95, max_time=180)
    return outputs[0]['generated_text'][len(prompt):].strip()

# Example to test the model
input_text = ("You are a proposal writer for a consortium that consists of Public Power Corporation as the energy provider, TU Eindhoven as the stream processing experts, Athena Research Center and Uni Athens as AI, ApeiroPlus as risk assessment expert, Uni Trento as Earth Observation expert, and EuroControl as a use case provider. The topic of the proposal is the following: \
ExpectedOutcome: \
Projects results are expected to contribute to all of the following outcomes:\
Critical infrastructure operators are more resilient to threats and natural and human-made hazards;\
Improved monitoring, risk assessment, forecast, mitigation and modelling techniques aimed at increasing the resilience of critical infrastructures, validating multi-hazard scenarios, creating interactive hazard maps supported by Earth Observation and other data sources.\
Scope:\
Under the Open Topic, proposals are welcome to address new, upcoming or unforeseen challenges and/or creative or disruptive solutions for increasing the resilience of critical infrastructure, that are not covered by the other topics of Horizon Europe Calls Resilient Infrastructure 2021-2022, Resilient Infrastructure 2023 and Resilient Infrastructure 2024.\
Adapted to the nature, scope and type of proposed activities, proposals should convincingly explain how they will plan and/or carry out demonstration, testing or validation of developed tools and solutions. Proposals should also delineate the plans to develop possible future uptake and upscaling at local, national and EU level for possible next steps after the project.\
In this topic the integration of the gender dimension (sex and gender analysis) in research and innovation content should be addressed only if relevant in relation to the objectives of the research effort.\
Proposals should consider, build on if appropriate and not duplicate previous research, including but not limited to research by other Framework Programmes’ projects. When applicable, the successful proposal should build on the publicly available achievements and findings of related previous national or EU-funded projects.\
Activities are expected to achieve TRL 6-8 by the end of the project – see General Annex B. Please provide the Subsection: 1.2.5: Gender Dimension"
)

# Generate and print the response
print(test_inference(input_text))
