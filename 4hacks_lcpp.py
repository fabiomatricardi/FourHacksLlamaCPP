#load the model with llama-cpp-python
from llama_cpp import Llama
print("\033[95;3;6m")
print("1. loading the model...")
print("\033[91;1m")  #red
print("2. Model qwen2-0_5b-instruct-q8_0.gguf loaded with LlamaCPP...")
print("\033[0m")  #reset all
llm = Llama(
            model_path='models/qwen2-0_5b-instruct-q8_0.gguf',
            n_gpu_layers=0,
            temperature=0.1,
            n_ctx=8192,
            max_tokens=600,
            repeat_penalty=1.7,
            stop=['<|im_end|>','<|endoftext|>'],
            verbose=True,
            )
# set the messages and prompt
prompt = 'Explain in simple terms what is history as a discipline.'
messages = [
    {"role": "system", "content": "You are Qwen2, a helpful AI assistant."},
    {"role": "user", "content": prompt}
]
# run the inference and streaming the output
full_response = ""
print("\033[92;1m") #green
for chunk in llm.create_chat_completion(
    messages=messages,
    temperature=0.25,
    repeat_penalty= 1.6,
    stop=['<|im_end|>','<|endoftext|>'],
    max_tokens=600,
    stream=True,):
    try:
        if chunk["choices"][0]["delta"]["content"]:
            print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
            full_response += chunk["choices"][0]["delta"]["content"]                              
    except:
        pass        
# append the new message to the chat history
print("\033[0m")  #reset all color and text formatting
messages.append({"role": "assistant", "content": full_response})

