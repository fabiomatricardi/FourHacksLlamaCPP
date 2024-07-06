#load the model with OpenAI and llamafile
from openai import OpenAI
print("\033[95;3;6m")
print("1. loading the model...")
client = OpenAI(base_url="http://localhost:8001/v1", api_key="not-needed")
print("\033[91;1m")  #red
print("2. Model qwen2-0_5b-instruct-q8_0.gguf loaded with OpenAI and Llamfile...")
print("\033[0m")  #reset all
# set the messages and prompt
prompt = 'Explain in simple terms what is history as a discipline.'
messages = [
    {"role": "system", "content": "You are Qwen2, a helpful AI assistant."},
    {"role": "user", "content": prompt}
]
# run the inference API call
completion = client.chat.completions.create(
    model="qwen2-0_5b-instruct", # this field is currently unused
    messages=messages,
    temperature=0.3,
    frequency_penalty  = 1.6,
    max_tokens = 600,
    stream=True,
    stop=['<|im_end|>','<|endoftext|>']
)    

new_message = {"role": "assistant", "content": ""}
# and streaming the output
for chunk in completion:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
        new_message["content"] += chunk.choices[0].delta.content
# append the new message to the chat history
print("\033[0m")  #reset all color and text formatting
messages.append(new_message) 