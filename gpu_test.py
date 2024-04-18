import torch
from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer
from utils import get_data

def get_model(model_name, vocab_size=49152, seq_length=2048):
    configuration = LlamaConfig()
    
    configuration.vocab_size = vocab_size
    configuration.max_position_embeddings = seq_length
    
    if model_name == "llama_1b":
        configuration.hidden_size = 1536
        configuration.num_hidden_layers = 24
        configuration.num_attention_heads = 32
    elif model_name == "llama_2b":
        configuration.hidden_size = 2176
        configuration.num_hidden_layers = 24
        configuration.num_attention_heads = 32
    elif model_name == "llama_3b":
        configuration.hidden_size = 2688
        configuration.num_hidden_layers = 24
        configuration.num_attention_heads = 32
    elif model_name == "llama_4b":
        configuration.hidden_size = 3136
        configuration.num_hidden_layers = 24
        configuration.num_attention_heads = 32
    elif model_name == "llama_5b":
        configuration.hidden_size = 3520
        configuration.num_hidden_layers = 24
        configuration.num_attention_heads = 32 
    elif model_name == "llama_6.7b":
        configuration.hidden_size = 4096
        configuration.num_hidden_layers = 32
        configuration.num_attention_heads = 32
    elif model_name == "llama_13b":
        configuration.hidden_size = 5120
        configuration.num_hidden_layers = 40
        configuration.num_attention_heads = 40
    elif model_name == "llama_33b":
        configuration.hidden_size = 6656
        configuration.num_hidden_layers = 60
        configuration.num_attention_heads = 52
    elif model_name == "llama_65b":
        configuration.hidden_size = 8192
        configuration.num_hidden_layers = 80
        configuration.num_attention_heads = 64
    elif model_name == "llama_tiny":
        configuration.hidden_size = 2048
        configuration.num_hidden_layers = 22
        configuration.num_attention_heads = 32
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Set FFN dimension to 4x d_model
    configuration.intermediate_size = configuration.hidden_size * 4
    
    model = LlamaForCausalLM(configuration)
    
    return model

def load_and_test_model(model_name):
    model = get_model(model_name)
    train_data = get_data(config)[1]
    x, y = next(train_data)

    # Generate output
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        print("Loss:", outputs.loss.item())

        generated_ids = model.generate(inputs['input_ids'], max_length=50)
        print("Generated text:", tokenizer.decode(generated_ids[0], skip_special_tokens=True))


def list_nvidia_gpus():
    if torch.cuda.is_available():
        # Print the total number of GPUs available
        num_gpus = torch.cuda.device_count()
        print(f"Total number of GPUs available: {num_gpus}")
        
        # List each GPU along with its ID
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU ID: {i}, GPU Name: {gpu_name}")
    else:
        print("No CUDA GPUs are available.")
        

if __name__ == "__main__":
    load_and_test_model("llama_tiny")
