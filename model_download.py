from modelscope.models import Model
from modelscope.hub.snapshot_download import snapshot_download
from transformers import AutoTokenizer

import os
import shutil

# # 下载模型到本地指定目录
# save_dir = '/home/root/shared_tmp/models/llama'
# model_id = 'LLM-Research/Meta-Llama-3.1-8B-Instruct'  # 替换为 ModelScope 上的模型 ID

raw_cache_dir = '/home/root/shared_tmp/models/llama'
model_name = 'LLM-Research/Llama-3.2-1B-Instruct'

# 提取模型名称部分
# model_name = model_id.split("/")[-1]
target_dir = os.path.join(raw_cache_dir, model_name)
print(target_dir)

# 下载模型
model_dir = snapshot_download(model_name, cache_dir=raw_cache_dir)

# 如果目标目录不存在，就移动过去
if model_dir != target_dir:
    os.makedirs(os.path.dirname(target_dir), exist_ok=True)
    # shutil.move(model_dir, target_dir)

# 删除空的 LLM-Research 文件夹（可选）
namespace_dir = os.path.dirname(model_dir)
if os.path.isdir(namespace_dir) and not os.listdir(namespace_dir):
    os.rmdir(namespace_dir)

print(target_dir)
# # 加载模型（使用新目录）
# model = Model.from_pretrained("shared_tmp/models/llama/LLM-Research/Meta-Llama-3___1-8B-Instruct")
# tokenizer = AutoTokenizer.from_pretrained("shared_tmp/models/llama/LLM-Research/Meta-Llama-3___1-8B-Instruct")