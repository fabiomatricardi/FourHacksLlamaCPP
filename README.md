# FourHacksLlamaCPP
Four hacks to run LLMs with your CPU. Repo of the Medium article

<img src='https://github.com/fabiomatricardi/FourHacksLlamaCPP/raw/main/social-banner.png' height=400>

### Requirements
you can find everything iin the requirements.txt

To install by yourself:
```
pip install 'llama-cpp-python[server]'
pip install --upgrade langchain langchain-community openai
```

### Llamafile
we used for this project version 0.8.6
```
wget https://github.com/Mozilla-Ocho/llamafile/releases/download/0.8.6/llamafile-0.8.6 -OutFile llamafile-0.8.6.exe
```


### Model
we are using Qwen2-0.5b-instruct in the Q8 quantized version

https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF

