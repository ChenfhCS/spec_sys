import csv
import os.path

import zmq
import time
import torch
import base64
import argparse
import numpy as np
import pandas as pd
from torch import Tensor
from tqdm import tqdm

from datasets import load_dataset
from zmq.sugar.socket import Socket
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


# class Decoder:
#     def __init__(self, tokenizer) -> None:
#         self.tokenizer = tokenizer
#
#     def decode(self, t: torch.Tensor) -> str:
#         # assert t.dim == 2, "t must be 2d tensor"
#         return self.tokenizer.decode(t[0], skip_special_tokens=True)
#
#
# DECODER: Decoder = None


def sample(probs: torch.Tensor, num_samples: int = 1):
    """
    Samples indices from the given probability distribution. Ensures valid sampling.
    从给定的概率分布中采样。使用 torch.multinomial 来从概率分布中选取一个索引（token）
    """
    idx_next = torch.multinomial(probs, num_samples=num_samples)
    # Ensure the index is non-zero
    if probs[0, idx_next.item()] == 0:
        raise RuntimeError(f"Sampled token has zero probability: {idx_next.item()}")
    return idx_next, probs[0, idx_next.item()].item()


# copy from https://github.com/LeeSinLiang/microGPT/blob/ed40cf9780dbeb180adfe94c227d4aa97e69250e/gpt.py
def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    """
    对模型的输出 logits 进行过滤，确保在生成下一个 token 时只考虑 top-k 或 top-p 范围内的 token。
    top_k 选择前 k 个最可能的 tokens，top_p 选择累积概率超过 p 的 tokens
    Args:
        logits (torch.Tensorpe_): 2D tensor with shape (batch, vocab)
        top_k (int, optional): top_k. Defaults to 0.
        top_p (float, optional): top_p. Defaults to 0.0.

    Returns:
        torch.Tensor: a renormalized logits
    """
    if top_k > 0:
        filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        # print(logits.shape)
        # print(filter[:, [-1]].shape)
        logits[logits < filter[:, [-1]]] = float("-inf")
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0
        indices_to_remove = filter.scatter(1, sorted_indices, filter)
        logits[indices_to_remove] = float("-inf")
    return logits


def norm_logits(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    """
    对 logits 进行温度缩放（logits / temperature），然后应用 top-k 和 top-p 过滤，最后计算概率分布
    Args:
        logits (torch.Tensor): shape (1, vocab)
        temperature (float): temperature
        top_k (int): top_k
        top_p (float): top_p

    Returns:
        torch.Tensor: next token with shape as (batch, 1)
    """
    assert logits.dim() == 2
    logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=1)
    return probs


def max_fn(x):
    """
        norm(max (x, 0))
    """
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=1, keepdim=True)
    return x_max / x_max_sum


def process_message(message: dict) -> torch.Tensor:
    """将客户端发来的JSON数据解析成 Tensor"""
    shape = tuple(message["shape"])
    data = base64.b64decode(message["data"])
    dtype = message["dtype"]
    np_array = np.frombuffer(data, dtype=eval("np.{}".format(dtype))).reshape(shape)
    return torch.from_numpy(np_array)


def send_json(q, x, gamma, prefix_len, temperature, top_k, top_p, socket, signal=""):
    if q is not None and x is not None:
        for i in range(q.shape[1]):
            # 小模型生成的 logits 进行归一化处理
            q[:, i, :] = norm_logits(q[:, i, :], temperature, top_k, top_p)

        # 发送 q 和 x
        q_array = q.cpu().numpy()
        q_array_list = []
        for i in range(gamma):
            j = x[:, prefix_len + i]
            q_array_list.append(float(q_array[:, prefix_len + i - 1, j][0]))

        x_array = x.cpu().numpy()[:, prefix_len:]
        x_data = base64.b64encode(x_array).decode("ascii")
        message = {
            # "q_shape": list(q_array.shape),
            # "q_dtype": str(q_array.dtype),
            "q_array_list": q_array_list,
            "x_shape": list(x_array.shape),
            "x_dtype": str(x_array.dtype),
            "x_data": x_data,
            # "signal": "continue",
            "signal": signal,
        }
    else:
        message = {
            "q_array_list": [],
            "a_array_list": [],
            "signal": signal,
        }
    socket.send_json(message)


@torch.no_grad()
def speculative_sampling_v2(prefix: torch.Tensor, approx_model: torch.nn.Module, target_model: torch.nn.Module,
                            max_len: int, gamma: int = 4, temperature: float = 1,
                            top_k: int = 0, top_p: float = 0,
                            socket: Socket = None, writer_wrong_pos=None, tokenizer=None):
    """
    DeepMind version Speculative Sampling.
    Accelerating Large Language Model Decoding with Speculative Sampling
    https://arxiv.org/abs/2302.01318
    No KV Cache Optimization
    x 是当前的输入序列；prefix_len 是当前输入序列的长度；
    for _ in range(gamma) 循环表示小模型每次生成 gamma 个 token；
    q = approx_model(x).logits：使用小模型生成 logits（每个 token 的概率分布）；
    next_tok = sample(norm_logits(q[:, -1, :], temperature, top_k, top_p))：对最后一个 token 的 logits 进行处理并采样，得到下一个 token。
    x = torch.cat((x, next_tok), dim=1)：将新生成的 token 加入当前序列中。
    生成的新 token (next_tok) 被添加到 x 中，通过 torch.cat((x, next_tok), dim=1) 更新了 x。这时，x 包含了小模型生成的序列（含有已生成的 token）
    Returns:
        torch.Tensor: Generated tokens (batch, target_seqlen).
    """
    max_len = 200
    time_start = time.time()
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    # 发送最大生成长度 + 前缀
    prefix_array = prefix.cpu().numpy()
    data = base64.b64encode(prefix_array).decode("ascii")
    message = {
        "shape": list(prefix_array.shape),
        "dtype": str(prefix_array.dtype),
        "data": data,
        "signal": "continue",
        "max_len": max_len,
        "gamma": gamma,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "p_plus": 0.01
    }
    socket.send_json(message)

    # # 接收回应
    # message = client.zmq_comm.recv()
    # # print(message)
    # message = struct.unpack("!I", message[:4])[0]
    # print("111111111111111", message)

    assert prefix.shape[0] == 1, "Input batch size must be 1"

    # cumulative_acc = []
    # current_acc_token = 0
    # current_spec_token = 0

    receive_directly_num = 0

    confidence_mean_list = []
    while prefix.shape[1] < T:
        print("prefix.shape:", prefix.shape)
        prefix_len = prefix.shape[1]
        # q = M_q[prefix + x_0, x_1, ..., x_(gamma-2)]
        x = prefix
        confidence_list = []
        token_list = []
        for _ in range(gamma):
            # Approx model logits shape: (batch, seq, vocab)
            q = approx_model(x).logits
            next_tok, confidence = sample(norm_logits(q[:, -1, :], temperature, top_k, top_p))
            x = torch.cat((x, next_tok), dim=1)
            confidence_list.append(confidence)
            token_list.append(next_tok)

        confidence_mean = sum(confidence_list) / len(confidence_list)
        confidence_mean_list.append(confidence_mean)
        # print("confidence_mean:", confidence_mean)
        # print(confidence_mean, confidence_mean_list[-10:])

        # if len(confidence_mean_list) > 10:
        #     confidence_mean_last_10 = sum(confidence_mean_list[-10:]) / 10
        #     if confidence_mean_last_10 < 0.2:
        #         print("小模型生成的结果太差，完全由大模型指导输出！！")
        #         send_json(None, None, gamma, prefix_len,
        #                   temperature, top_k, top_p, socket, signal="always")
        #         received_message = socket.recv_json()
        #         new_token_tensor = process_message(received_message).to("cuda:0")
        #         prefix = torch.cat([prefix, new_token_tensor], dim=1)
        #         break
        if confidence_mean > 1000000.:
            # print("不发送给大模型诊断，小模型的输出直接作为生成式模型的输出！！")
            prefix = x
            prefix_array = prefix.cpu().numpy()[:, -gamma:]
            prefix_data = base64.b64encode(prefix_array).decode("ascii")
            message = {
                "prefix_shape": list(prefix_array.shape),
                "prefix_dtype": str(prefix_array.dtype),
                "prefix_data": prefix_data,
                "signal": "immediate",
            }
            socket.send_json(message)
            receive_directly_num += 1
        else:
            # print("发送给大模型诊断!!")
            send_json(q, x, gamma, prefix_len,
                      temperature, top_k, top_p, socket, signal="continue")
            # # Normalize logits for approx model
            # for i in range(q.shape[1]):
            #     # 小模型生成的 logits 进行归一化处理
            #     q[:, i, :] = norm_logits(q[:, i, :], temperature, top_k, top_p)
            #
            # # 发送 q 和 x
            # q_array = q.cpu().numpy()
            # q_array_list = []
            # for i in range(gamma):
            #     j = x[:, prefix_len + i]
            #     q_array_list.append(float(q_array[:, prefix_len + i - 1, j][0]))
            # # q_data = base64.b64encode(q_array).decode("ascii")
            # x_array = x.cpu().numpy()[:, prefix_len:]
            # x_data = base64.b64encode(x_array).decode("ascii")
            # message = {
            #     # "q_shape": list(q_array.shape),
            #     # "q_dtype": str(q_array.dtype),
            #     "q_array_list": q_array_list,
            #     "x_shape": list(x_array.shape),
            #     "x_dtype": str(x_array.dtype),
            #     "x_data": x_data,
            #     "signal": "continue",
            # }
            # socket.send_json(message)

            # 等待服务器回应
            received_message = socket.recv_json()
            if "q_n" in received_message:
                q_n = received_message["q_n"]
                message = {
                    "q_n": q[:, q_n, :].tolist(),
                    "signal": "continue",
                }
                socket.send_json(message)

                received_message = socket.recv_json()
            new_token_tensor = process_message(received_message).to("cuda:0")
            # large_model_p = received_message["large_model_p"]
            # small_model_q = received_message["small_model_q"]
            # r = received_message["r"]
            #
            # a, b = new_token_tensor.shape
            # # print("a b:", a, b)
            # if b != 9:
            #     writer_wrong_pos.writerow([b - 1, confidence_list[b - 1], large_model_p, small_model_q, r])
            # else:
            #     writer_wrong_pos.writerow([8, 1.0, large_model_p, small_model_q, r])

            prefix = torch.cat([prefix, new_token_tensor], dim=1)

    print(f"Current sequence length (L): {prefix.shape[1]}")
    time_end = time.time()
    print("spend time:", time_end - time_start)
    return prefix, time_end - time_start, receive_directly_num


def generate(input_text, approx_model_name, target_model_name, num_tokens=200, verbose=False):
    save_path = "llama-13b_JackFram_llama-68m"
    if not os.path.exists(os.path.join(save_path, "dolly_hhrlhf_300_plus")):
        os.makedirs(os.path.join(save_path, "dolly_hhrlhf_300_plus"))

    server_ip = "127.0.0.1"
    port = 6006
    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    socket.connect(f"tcp://{server_ip}:{port}")
    print("[Client] Connected to server:", f"tcp://{server_ip}:{port}")

    # NOTE() approx_model_name and target_model_name should use the same tokenizer!
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(approx_model_name)

    # global DECODER
    # DECODER = Decoder(tokenizer)

    print("begin loading models")
    small_model = AutoModelForCausalLM.from_pretrained(approx_model_name).to(torch_device)
    # large_model = AutoModelForCausalLM.from_pretrained(target_model_name).to(torch_device)
    print("finish loading models")
    small_model.eval()

    top_k = 10
    top_p = 0.9

    # path = "/root/autodl-tmp/wanjie/data_and_model/data/alespalla/chatbot_instruction_prompts"
    # dataset = load_dataset(path)
    # dataset_test_100 = dataset["test"].select(range(100))
    # print(dataset_test_100)

    df = pd.read_csv("get_large_model_PPL/data300_dolly_hhrlhf.csv")

    csvfile = open("{}/dolly_hhrlhf_300_plus/dolly_hhrlhf_0.01.csv".format(save_path),
                   "w", newline="", encoding="utf-8")
    writer = csv.writer(csvfile)
    writer.writerow(["prompt", "generated_text", "time", "receive_directly_num"])

    csvfile_wrong_pos = open(
        "{}/dolly_hhrlhf_300_plus/dolly_hhrlhf_wrong_pos_confidence_0.01.csv".format(save_path),
        "w", newline="", encoding="utf-8")
    writer_wrong_pos = csv.writer(csvfile_wrong_pos)
    writer_wrong_pos.writerow(["wrong_pos", "wrong_pos_confidence", "large_model_p", "small_model_q", "r"])

    total_time = 0.
    for one in df["prompt"]:
        start_time = time.time()
        input_ids = tokenizer.encode(one, return_tensors="pt").to(torch_device)
        output, time_spend, receive_directly_num = speculative_sampling_v2(
            input_ids, small_model, None,
            num_tokens, top_k=top_k, top_p=top_p,
            socket=socket, gamma=8,
            writer_wrong_pos=writer_wrong_pos,
            tokenizer=tokenizer)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        total_time += time_spend
        print(f"deepmind's speculative_sampling: {generated_text}")
        writer.writerow([one, generated_text, time.time() - start_time, receive_directly_num])
    print("total time:", total_time, total_time / 300)
    socket.send_json({"stop": "stop"})

    csvfile.close()
    if csvfile_wrong_pos is not None:
        csvfile_wrong_pos.close()


def parse_arguments():
    parser = argparse.ArgumentParser(description="args for sample.py")

    parser.add_argument("--input", type=str,
                        default="It seemed funny at that time by now I realize it was not funny at all.")
    parser.add_argument("--approx_model_name", type=str, default="/root/autodl-tmp/lrx/JackFram/llama-68m")
    parser.add_argument("--target_model_name", type=str, default="/root/autodl-tmp/lrx/huggyllama/llama-7b")
    parser.add_argument("--verbose", "-v", action="store_true", default=False, help="enable verbose mode")
    parser.add_argument("--confidence", type=float, help="")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    generate(args.input, args.approx_model_name, args.target_model_name, verbose=args.verbose)
