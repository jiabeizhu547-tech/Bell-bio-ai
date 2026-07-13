"""
下载蛋白质二级结构预测数据集。

使用 CB513 数据集（Cuff & Barton, 1999），
这是蛋白质二级结构预测领域最常用的基准数据集之一。

数据来源：https://github.com/kipoi/kipoi-model-zoo/
"""

import os
import urllib.request
import gzip
import shutil
from pathlib import Path

# ===== 配置 =====
DATA_DIR = Path(__file__).parent / "data"
CB513_URL = (
    "https://raw.githubusercontent.com/kipoi/kipoi-model-zoo/"
    "master/DeepBind/dataloader_files/"
    "cb513.fa.gz"  # 备选：直接从可靠镜像下载
)

# 备用方案：使用蛋白质序列 + DSSP 注释的经典格式
# 如果 CB513 下载失败，我们从另一个源获取


def download_cb513():
    """下载 CB513 数据集（FASTA 格式 + DSSP 注释）"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = DATA_DIR / "cb513.fa.gz"

    if output_path.exists():
        print(f"[OK] 数据已存在：{output_path}")
        return output_path

    print(f"[*] 正在下载 CB513 数据集...")
    try:
        urllib.request.urlretrieve(CB513_URL, output_path)
        print(f"[OK] 下载完成：{output_path}")
    except Exception as e:
        print(f"[FAIL] 下载失败：{e}")
        print("[!] 请手动下载数据集，或运行 generate_demo_data() 生成演示数据")
        return None

    return output_path


def generate_demo_data(n_sequences: int = 500):
    """
    生成演示用蛋白质二级结构数据。

    当无法下载真实数据时，生成合成的氨基酸序列和对应的
    二级结构标签（α-螺旋、β-折叠、无规卷曲）用于测试训练流程。
    """
    import numpy as np
    import random

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 20 种标准氨基酸
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    # 3 类二级结构：H=α-螺旋, E=β-折叠, C=无规卷曲（coil）
    ss_classes = ["H", "E", "C"]

    # 模拟真实分布：coil ≈ 45%, helix ≈ 35%, sheet ≈ 20%
    ss_weights = [0.35, 0.20, 0.45]

    sequences = []
    structures = []

    for i in range(n_sequences):
        length = random.randint(50, 300)  # 蛋白质长度 50-300 残基
        seq = "".join(random.choices(amino_acids, k=length))
        ss = "".join(random.choices(ss_classes, weights=ss_weights, k=length))
        sequences.append(f">seq_{i}\n{seq}")
        structures.append(f">seq_{i}\n{ss}")

    seq_file = DATA_DIR / "demo_sequences.fasta"
    ss_file = DATA_DIR / "demo_structures.fasta"

    with open(seq_file, "w") as f:
        f.write("\n".join(sequences))

    with open(ss_file, "w") as f:
        f.write("\n".join(structures))

    print(f"[OK] 生成 {n_sequences} 条演示序列 -> {seq_file}")
    print(f"[OK] 生成 {n_sequences} 条结构注释 -> {ss_file}")
    print("[!] 这是随机生成的数据，仅用于测试代码流程，模型不会有实际预测能力")

    return seq_file, ss_file


if __name__ == "__main__":
    result = download_cb513()
    if result is None:
        print("\n[*] 改为生成演示数据...")
        generate_demo_data()
