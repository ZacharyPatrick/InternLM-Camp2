# 书生·浦语大模型实战营第二期（初夏专场）：（二）轻松玩转书生·浦语大模型趣味 Demo

## 任务1：**部署 `InternLM2-Chat-1.8B` 模型进行智能对话**

### 1.1 配置基础环境

首先输入所给命令进行环境的配置安装，经过较长时间的安装等待，安装成功后显示成功进入`demo`虚拟环境：

![image-20240524195054494](img/image-20240524195054494.png)

### 1.2 下载`InternLM2-Chat-1.8B` 模型

输入如下命令创建`demo`文件夹，并进入目录中：

```python
mkdir -p /root/demo
touch /root/demo/cli_demo.py
touch /root/demo/download_mini.py
cd /root/demo
```

![image-20240524195502654](img/image-20240524195502654.png)

随后双击打开`download_mini.py`文件，并复制以下代码：

```python
import os
from modelscope.hub.snapshot_download import snapshot_download

# 创建保存模型目录
os.system("mkdir /root/models")

# save_dir是模型保存到本地的目录
save_dir="/root/models"

snapshot_download("Shanghai_AI_Laboratory/internlm2-chat-1_8b", 
                  cache_dir=save_dir, 
                  revision='v1.1.0')
```

执行以下命令，下载模型参数文件：

```python
python /root/demo/download_mini.py
```

![image-20240524195959707](img/image-20240524195959707.png)

### 1.3 运行`cli_demo`

双击打开`cli_demo.py`文件，复制以下代码输入到文件中：

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_name_or_path = "/root/models/Shanghai_AI_Laboratory/internlm2-chat-1_8b"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, device_map='cuda:0')
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='cuda:0')
model = model.eval()

system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""

messages = [(system_prompt, '')]

print("=============Welcome to InternLM chatbot, type 'exit' to exit.=============")

while True:
    input_text = input("\nUser  >>> ")
    input_text = input_text.replace(' ', '')
    if input_text == "exit":
        break

    length = 0
    for response, _ in model.stream_chat(tokenizer, input_text, messages):
        if response is not None:
            print(response[length:], flush=True, end="")
            length = len(response)
```

同样输入命令执行Demo程序：

```python
conda activate demo
python /root/demo/cli_demo.py
```

等待模型加载完成后，键入提示语，效果如下：

![image-20240524200841500](img/image-20240524200841500.png)

## 任务2：**部署实战营优秀作品 `八戒-Chat-1.8B` 模型**

### 2.1 配置基础环境

激活`demo`虚拟环境后，使用`git`命令获得仓库内的Demo文件：

```python
cd /root/
git clone https://gitee.com/InternLM/Tutorial -b camp2
# git clone https://github.com/InternLM/Tutorial -b camp2
cd /root/Tutorial
```

![image-20240524201355890](img/image-20240524201355890.png)

### 2.2 下载运行**Chat-八戒 Demo**

输入以下命令下载模型：

```python
python /root/Tutorial/helloworld/bajie_download.py
```

![image-20240524201858213](img/image-20240524201858213.png)

模型下载结束后，输入以下运行命令：

```python
streamlit run /root/Tutorial/helloworld/bajie_chat.py --server.address 127.0.0.1 --server.port 6006
```

随后打开`PowerShell`，查询开发机端口为34637：

![image-20240524202207913](img/image-20240524202207913.png)

之后则在`PowerShell`中输入以下命令：

```python
ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p 34637
```

随后再复制输入密码：

![image-20240524205320334](img/image-20240524205320334.png)

等待页面加载完成后，输入提示语如下，效果图如下：

![image-20240524210003592](img/image-20240524210003592.png)