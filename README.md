to run the script:
uvicorn ai_endpoints:app --reload

# Text generation

This feature is based on the Llama-2 provided by Facebook. To run it on my local computer, namely MacBook, I had to utilize:
1. llama.cpp - In order to quantize a model to be able to run the hardware without any GPU. It uses only CPU + memory.
2. llama-cpp-python - It is a wrapper between python and llama.cpp. Using this library, we can actually access llama.cpp from Python.

## How to setup the environment for text generation

```
git clone git@github.com:facebookresearch/llama.git
```

You need to provide the credentials that were sent to you by email. It is a normal procedure. If you don't have access to the models, you need to request it from the llama-2 page. 

The next step is to download the llama.cpp package in order to quantize the models, for example, from F32 to 4-bit integer.

```
git clone git@github.com:ggerganov/llama.cpp.git
```

Mandatory packages:
```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

xcode-select –install 

```
Build the package:
```
cd llama.cpp

LLAMA_METAL=1 make
```

The next step is to convert the model that has been created while installing llama in the first step. It will convert the model to the ggml format. "GGML is a tensor library for machine learning to enable large models and high performance on commodity hardware."[x]

[x] - https://ggml.ai
```
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt

python convert.py ../llama/llama-2-7b-chat
```

Inside the llama-2-7b-chat folder, you should see a file similar to this one: ggml-model-f16.gguf

In order to make this model smaller (quantize it), we will quantize the model from f16 to 4-bit integer.

```
./quantize ../llama/llama-2-7b-chat/ ggml-model-f16.gguf ../llama/llama-2-7b-chat/ggml-model-f16_q4_0.bin q4_0
```
Instead of q4_0 you can choose different options. It mostly refers to the accuracy.
More details:
https://github.com/ggerganov/llama.cpp/discussions/406
https://github.com/ggerganov/llama.cpp/discussions/1121

Now, if we want to use it, for example, as the Python API, we need to wrap it. To do so, it is mandatory to use **llama-cpp-python**.

```
CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install llama-cpp-python
```


The aforementioned steps are based on this tutorial:
https://medium.com/@auslei/llama-2-for-mac-m1-ed67bbd9a0c2

However, instructions vary a bit because of many problems while carrying out the steps presented in the tutorial. 