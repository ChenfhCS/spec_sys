import os
import csv
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer


def download_and_sample_dataset(dataset_path_or_name, local_dir, sample_count, split="train", seed=2025):
    """
    如果本地存在数据集，则加载本地；否则从Hub下载并缓存到指定目录。
    支持 dataset repo 或本地 JSON 数据。
    """
    if os.path.exists(local_dir):
        print(f"📂 本地数据集已存在，路径: {local_dir}")
        dataset = load_dataset(local_dir, split=split)
    else:
        print(f"🌐 本地未找到，正在从 Hub 下载数据集: {dataset_path_or_name}")
        dataset = load_dataset(dataset_path_or_name, cache_dir=local_dir, split=split)
    
    # 随机抽样
    dataset = dataset.shuffle(seed=seed).select(range(sample_count))
    return dataset


def save_dataset_as_csv(dataset, csv_path):
    """
    保存数据集为 CSV 文件（包含 prompt 和 response）
    """
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["prompt", "response"])
        for example in dataset:
            writer.writerow([example["prompt"], example["response"]])


def save_dataset_as_hf_format(dataset, output_path):
    """
    保存为 HuggingFace Dataset 格式
    """
    DatasetDict({"validation": dataset}).save_to_disk(output_path)


def process_dolly_dataset():
    """
    专门处理 Dolly 数据集（mosaicml/dolly_hhrlhf）
    """
    dataset_path = "mosaicml/dolly_hhrlhf"
    cache_dir = "/home/root/shared_tmp/dataset/dolly_hhrlhf"
    csv_output_path = os.path.join(cache_dir, "dolly_sample.csv")
    hf_output_path = os.path.join(cache_dir, "dolly_sample_dataset")
    sample_count = 300

    print("📥 下载并抽样 Dolly 数据集中...")
    sampled_dataset = download_and_sample_dataset(
        dataset_path_or_name=dataset_path,
        local_dir=cache_dir,
        sample_count=sample_count
    )

    print("💾 保存为 CSV...")
    save_dataset_as_csv(sampled_dataset, csv_output_path)

    print("💾 保存为 HuggingFace 数据集格式...")
    save_dataset_as_hf_format(sampled_dataset, hf_output_path)

    print("✅ Dolly 数据处理完成！")


if __name__ == '__main__':
    process_dolly_dataset()
