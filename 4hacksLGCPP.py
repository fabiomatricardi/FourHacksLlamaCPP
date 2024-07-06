from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


# Loading the model
llm = LlamaCpp(
    model_path="models\qwen2-0_5b-instruct-q8_0.gguf",
    temperature=0.25,
    max_tokens=500,
    n_ctx = 10000,
    repeat_penalty=1.45,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
    stop = ['<|endoftext|>']
)
#preparing the prompt
question = 'What is Science?'
template = f"""<|im_start|>system\nYou are a helpful assistant.<|im_end|>
<|im_start|>user
Question: {question}

Answer: <|im_end|>\n<|im_start|>assistant\n
"""
print(llm.invoke(template))

print('\n \n \n \n')
# remove instruction format llm from memory
del llm
# import the Chat class
from langchain_community.chat_models import ChatLlamaCpp
# Loading the model
llm = ChatLlamaCpp(
    model_path="models\qwen2-0_5b-instruct-q8_0.gguf",
    temperature=0.25,
    max_tokens=500,
    n_ctx = 10000,
    repeat_penalty=1.5,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
    stop = ['<|endoftext|>'],
    stream=True
)
#preparing the prompt in CHAT format style
question = 'What is Artificial Intelligence?'
messages = history = [
    {"role": "system", "content": "You are QWEN05, an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful. Always reply in the language of the instructions."},
    {"role": "user", "content": question},
]
print(llm.invoke(messages, stop=['<|endoftext|>']))