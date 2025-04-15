import csv
import torch
# import numpy as np
import pandas as pd

from datasets import load_from_disk
from tqdm import tqdm

# from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from speculative_decoding import sample, norm_logits


model_pt = "/home/root/shared_tmp/models/llama/LLM-Research/Meta-Llama-3___1-8B-Instruct"
dataset_pt = "/home/root/shared_tmp/dataset/dolly_hhrlhf/dolly_sample_dataset"
# output_pt = ""

class Infer(object):

    def __init__(self):
        # self.model_path = "/root/autodl-tmp/lrx/JackFram/llama-68m"
        self.model_path = model_pt
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        # self.config = AutoConfig.from_pretrained(
        #     self.model_path,
        #     trust_remote_code=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            # device_map="cpu",
            device_map="cuda:0",
            # config=self.config,
            trust_remote_code=True)
        self.model.eval()

    def infer(self):
        # 加载 Hugging Face 本地保存的数据集
        dataset_path = "/home/root/shared_tmp/dataset/dolly_hhrlhf/dolly_sample_dataset"
        dataset = load_from_disk(dataset_path)["validation"]

        # 保存推理结果
        output_csv = "data300_alespalla_llama-1b_infer_one_by_one_150_from_hf.csv"
        with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["prompt", "generated_text", "response"])

            # 主进度条：遍历所有样本
            for i in tqdm(range(150), desc="Inference Progress", unit="sample", position=1):
                example = dataset[i]
                prompt = example["prompt"]
                response = example["response"]

                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
                x = input_ids

                # 子进度条：当前样本逐 token 推理（200 tokens max）
                for _ in tqdm(range(200), desc=f"Sample {i}", leave=False, position=0, unit="token"):
                    q = self.model(x).logits
                    next_tok = sample(norm_logits(q[:, -1, :], 1, 10, 0.9))
                    x = torch.cat((x, next_tok), dim=1)

                    if next_tok[0][0].item() == self.tokenizer.eos_token_id:
                        break  # 提前结束

                # decode 最终文本
                result = self.tokenizer.decode(x[0], skip_special_tokens=True)

                # 保存每一条结果
                writer.writerow([prompt, result, response])
                # print(f"✅ Inference Complete!")


def main():
    infer = Infer()
    infer.infer()


if __name__ == '__main__':
    main()
