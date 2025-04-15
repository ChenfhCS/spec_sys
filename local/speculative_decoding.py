
import csv
import time
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from typing import Tuple
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


# tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

class Decoder:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def decode(self, t: torch.Tensor) -> str:
        # assert t.dim == 2, "t must be 2d tensor"
        return self.tokenizer.decode(t[0], skip_special_tokens=True)


DECODER: Decoder = None


def sample(probs: torch.Tensor, num_samples: int = 1):
    """
    Samples indices from the given probability distribution. Ensures valid sampling.
    从给定的概率分布中采样。使用 torch.multinomial 来从概率分布中选取一个索引（token）
    """
    idx_next = torch.multinomial(probs, num_samples=num_samples)
    # Ensure the index is non-zero
    # if probs[0, idx_next.item()] == 0:
    #     raise RuntimeError(f"Sampled token has zero probability: {idx_next.item()}")
    return idx_next


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
        logits[logits < filter[:, [-1]]] = float('-inf')
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0
        indices_to_remove = filter.scatter(1, sorted_indices, filter)
        logits[indices_to_remove] = float('-inf')
    return logits


def norm_logits(logits: torch.Tensor, temperature: float, top_k: float, top_p: float) -> torch.Tensor:
    """
    对 logits 进行温度缩放（logits / temperature），然后应用 top-k 和 top-p 过滤，最后计算概率分布
    Args:
        logits (torch.Tensor): shape (1, vocab)
        temperature (float): temperature
        top_k (float): top_k
        top_p (float): top_p

    Returns:
        torch.Tensor: next token with shape as (batch, 1)
    """
    assert logits.dim() == 2
    logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=1)
    return probs


def _debug_show_kvcache(past_key_values):
    if past_key_values is None:
        return
    for elem in past_key_values:
        k, v = elem
        print(f"kv cache: k shape {k.shape}, v shape {v.shape}")
        break


def trim_kv_cache(past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor]], q: torch.Tensor, end_pos: int):
    """
    trim the KV cache to the end_pos
    用于截取 key-value 缓存（KV 缓存），确保缓存的长度不会超过当前生成的 token 长度
    Args:
        past_key_values (Tuple): KV Cache
        end_pos (int): the position of the valid prefix

    Returns:
        Tuple: the trimmed KV Cache
    """
    past_key_values_trimmed = []
    for kv in past_key_values:
        k, v = kv
        # NOTE() the indexing is specific for bloom. This won't work for other models
        # For example llama k, v should be (batch, num_head, seq_len, hidden_dim)
        k = k[:, :, :end_pos]
        v = v[:, :end_pos, :]
        kv_trimmed = (k, v)
        past_key_values_trimmed.append(kv_trimmed)

    q = q[:, :end_pos, :]
    return past_key_values_trimmed, q


def forward_with_kvcache(model, input_ids, past_key_values, cached_q, temperature, top_k, top_p, use_debug=False):
    """
    模型的前向传播函数，支持缓存处理。如果缓存为空，会直接生成 logits；如果缓存已有，根据缓存进行推理，减少计算量
    """
    if past_key_values is None:
        assert cached_q is None
        # the first forward returns the prompt's logits
        outputs = model(input_ids)
        cached_q = outputs.logits
        for i in range(cached_q.shape[-2]):
            cached_q[:, i, :] = norm_logits(cached_q[:, i, :], temperature, top_k, top_p)
        last_q = cached_q[:, -1, :]
    else:
        # return the last token's logits
        cached_len = 0
        for kv in past_key_values:
            k, v = kv
            cached_len = k.shape[2]

        last_input_id = input_ids[:, cached_len:]
        if last_input_id.dim() == 1:
            last_input_id = torch.unsqueeze(last_input_id, 0)

        if use_debug:
            print(f"last_input_id shape {last_input_id.shape}")
            _debug_show_kvcache(past_key_values)

        outputs = model(last_input_id, past_key_values=past_key_values, use_cache=True)

        not_cached_q = outputs.logits
        if not_cached_q.dim() == 2:
            not_cached_q = torch.unsqueeze(not_cached_q, 0)

        for i in range(not_cached_q.shape[-2]):
            not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], temperature, top_k, top_p)

        if cached_q is not None:
            cached_q = torch.cat([cached_q, not_cached_q], dim=1)
        last_q = not_cached_q[:, -1, :]

    return last_q, outputs.past_key_values, cached_q


@torch.no_grad()
def autoregressive_sampling(x: torch.Tensor, model: torch.nn.Module, N: int,
                            temperature: float = 1, top_k: int = 0, top_p: float = 0):
    n = len(x)
    T = len(x) + N

    past_key_values = None
    with tqdm(total=N, desc="autoregressive sampling") as pbar:
        while n < T:
            # outputs = model(x)
            last_q, past_key_values, _ = forward_with_kvcache(model, x, past_key_values, None, temperature, top_k,
                                                              top_p)
            # logits = outputs.logits[::, -1, :]
            # past_key_values = outputs.past_key_values
            idx_next = sample(last_q)
            x = torch.cat((x, idx_next), dim=1)
            n += 1
            pbar.update(1)

    return x


def max_fn(x):
    """
        norm(max (x, 0))
    """
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=1, keepdim=True)
    return x_max / x_max_sum


def generate_with_kvcache(prefix: torch.Tensor,
                          gamma: int,
                          approx_model: torch.nn.Module,
                          temperature: float,
                          top_k: float,
                          top_p: float,
                          past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor]] = None,
                          cached_q=None,
                          use_debug=False) -> Tuple[
    torch.Tensor, torch.Tensor, Tuple[Tuple[torch.Tensor, torch.Tensor]]]:
    """ forward the model gamma times
    每次迭代中，approx_model（小模型）会使用输入的 x（当前的 prefix）生成预测的下一个 token，并将其附加到 x 上。这个过程不断进行 gamma 次（即每次生成 gamma 个 token）
    ·  approx_model(x).logits：小模型根据输入 x（当前的 prefix）生成的 logits（即对下一 token 的概率分布）。
    ·  target_model(x).logits：大模型根据更新后的输入 x（包含了小模型生成的 token）生成的 logits
    小模型（approx_model）与大模型（target_model）之间的交互通过q （概率分布）和 x（生成token序列）张量传递

    Args:
        prefix (torch.Tensor): the prefix
        gamma (int): how many times approx guesses
        approx_model (torch.nn.Module): an approx model
        temperature (float): temp for sampling
        top_k (float): top_k for sampling
        top_p (float): top_p for sampling
        past_key_values : valid kv cache
        cached_q: valid probability distribution of vocab on the all of the prefix tokens

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Tuple]: prefix+generated tokens, past key value cache, probability distribution of vocab on the all of the tokens
    """
    x = prefix

    for _ in range(gamma):
        q, past_key_values, cached_q = forward_with_kvcache(approx_model, x, past_key_values, cached_q, temperature,
                                                            top_k, top_p, use_debug)
        next_tok = sample(q)
        x = torch.cat((x, next_tok), dim=1)

    return x, past_key_values, cached_q


@torch.no_grad()
def speculative_sampling_v2(prefix: torch.Tensor, approx_model: torch.nn.Module, target_model: torch.nn.Module,
                            max_len: int, gamma: int = 4, temperature: float = 1,
                            top_k: int = 0, top_p: float = 0) -> torch.Tensor:
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

    Args:
        prefix (torch.Tensor): input sequence, (batch, prefix_seqlen), batch size must be 1.
        approx_model (torch.nn.Module): Approximation model (small model).
        target_model (torch.nn.Module): Target model (large model).
        max_len (int): Maximum length of generated tokens.
        gamma (int): Number of tokens guessed by the small model per iteration.
        temperature (float, optional): Temperature for sampling. Defaults to 1.
        top_k (int, optional): Top-k sampling parameter. Defaults to 0.
        top_p (float, optional): Top-p sampling parameter. Defaults to 0.

    Returns:
        torch.Tensor: Generated tokens (batch, target_seqlen).
    """
    max_len = 350
    time_start = time.time()
    seq_len = prefix.shape[1]
    T = seq_len + max_len

    assert prefix.shape[0] == 1, "Input batch size must be 1"

    # cumulative_acc = []
    # current_acc_token = 0
    # current_spec_token = 0

    with tqdm(total=T, desc="speculative sampling") as pbar:
        while prefix.shape[1] < T:
            # q = M_q[prefix + x_0, x_1, ..., x_(gamma-2)]
            x = prefix
            prefix_len = prefix.shape[1]
            for _ in range(gamma):
                # Approx model logits shape: (batch, seq, vocab)
                q = approx_model(x).logits
                next_tok = sample(norm_logits(q[:, -1, :], temperature, top_k, top_p))
                x = torch.cat((x, next_tok), dim=1)

            # Normalize logits for approx model
            for i in range(q.shape[1]):
                # 小模型生成的 logits 进行归一化处理
                q[:, i, :] = norm_logits(q[:, i, :], temperature, top_k, top_p)
            # p = M_p[prefix + x_0, ..., x_(gamma-1)]
            p = target_model(x).logits
            for i in range(p.shape[1]):
                # 使用目标模型生成 logits，并对 logits 进行归一化处理
                # x（即包含小模型生成的序列）被传递给了目标模型 target_model。
                # p = target_model(x).logits 通过目标模型生成对应的 logits。目标模型根据这个输入进行推理，输出该输入的 token 概率分布
                p[:, i, :] = norm_logits(p[:, i, :], temperature, top_k, top_p)

            # Determine the end position of the valid prefix
            is_all_accept = True
            n = prefix_len - 1

            # current_spec_token += gamma
            for i in range(gamma):
                r = torch.rand(1, device=p.device)
                j = x[:, prefix_len + i]
                if r < torch.min(torch.tensor([1.0], device=q.device),
                                 (p[:, prefix_len + i - 1, j] + 0.02) / q[:, prefix_len + i - 1, j]):
                    # Accept and update n
                    n += 1
                    # current_acc_token += 1
                    # if current_acc_token % 10 == 0:
                    #     cumulative_acc.append(round(current_acc_token / current_spec_token, 2))
                else:
                    # Reject and sample correction
                    t = sample(max_fn(p[:, n, :] - q[:, n, :]))
                    # t = sample(p[:, n, :])
                    is_all_accept = False
                    break

            prefix = x[:, :n + 1]

            # If all tokens are accepted, sample from the last target logits
            if is_all_accept:
                t = sample(p[:, -1, :])
            prefix = torch.cat((prefix, t), dim=1)

            # Update progress bar
            pbar.update(n - pbar.n)

            # Output progress every time n increases by 10
            # if (n - prefix_len) % 30 == 0:
            # print(f"Current sequence length (L): {prefix.shape[1]}")
            # print(f"Current prefix content: {prefix}")
    print(f"Current sequence length (L): {prefix.shape[1]},token:{n}")
    time_end = time.time()
    print("spend time:", time_end - time_start)
    return prefix, time_end - time_start


def generate(input_text, approx_model_name, target_model_name, num_tokens=200, verbose=False):
    # NOTE() approx_model_name and target_model_name should use the same tokenizer!

    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(approx_model_name)

    global DECODER
    DECODER = Decoder(tokenizer)

    print("begin loading models")
    small_model = AutoModelForCausalLM.from_pretrained(approx_model_name).to(torch_device)
    large_model = AutoModelForCausalLM.from_pretrained(target_model_name, torch_dtype=torch.float16).to(torch_device)
    print("finish loading models")
    small_model.eval()
    large_model.eval()

    top_k = 10
    top_p = 0.9

    df = pd.read_csv("get_large_model_PPL/data300_alpaca-cleaned.csv")
    csvfile = open("speculative_original/data300_alpaca-cleaned_plus_0.02_150_new_350.csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(csvfile)
    writer.writerow(["prompt", "generated_text", "time"])

    for one in df["prompt"][:150]:
        start_time = time.time()
        input_ids = tokenizer.encode(one, return_tensors="pt").to(torch_device)
        output, time_spend = speculative_sampling_v2(
            input_ids, small_model, large_model, num_tokens, top_k=top_k, top_p=top_p, gamma=8)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"deepmind's speculative_sampling: {generated_text}")
        writer.writerow([one, generated_text, time.time() - start_time])
        print("over!!")

    # input_ids = tokenizer.encode(input_text, return_tensors='pt').to(torch_device)
    # output, time_spend = speculative_sampling_v2(
    #     input_ids, small_model, large_model, num_tokens, top_k=top_k, top_p=top_p, gamma=8)
    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # print(f"deepmind's speculative_sampling: {generated_text}")

    # torch.manual_seed(123)
    # output = autoregressive_sampling(input_ids, large_model, num_tokens, top_k = top_k, top_p=top_p)
    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # print(f"large (target) model autoregressive_sampling: {generated_text}")

    # torch.manual_seed(123)
    # output = autoregressive_sampling(input_ids, small_model, num_tokens, top_k = top_k, top_p=top_p)
    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # print(f"small (approx) model autoregressive_sampling: {generated_text}")

    # torch.manual_seed(123)
    # output = speculative_sampling(input_ids, small_model, large_model, num_tokens, top_k = top_k, top_p=top_p, verbose = verbose)
    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # print(f"google's speculative_sampling: {generated_text}")


def parse_arguments():
    parser = argparse.ArgumentParser(description='args for sample.py')

    parser.add_argument('--input', type=str,
                        default="It seemed funny at that time by now I realize it was not funny at all.")
    parser.add_argument('--approx_model_name', type=str, default="/root/autodl-tmp/lrx/JackFram/llama-68m")
    parser.add_argument('--target_model_name', type=str,
                        default="/root/autodl-tmp/wanjie/data_and_model/model/huggyllama/llama-13b")
    parser.add_argument('--verbose', '-v', action='store_true', default=False, help='enable verbose mode')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    generate(args.input, args.approx_model_name, args.target_model_name, verbose=args.verbose)
