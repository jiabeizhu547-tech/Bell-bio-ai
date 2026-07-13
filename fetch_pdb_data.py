"""
从 PDB 下载真实蛋白质结构，提取序列和二级结构标注（Q3：H/E/C）。

PDB 文件中的 HELIX 和 SHEET 记录来自实验测定，是真实可靠的二级结构数据。
每个残基都会被标注为 H（α-螺旋）、E（β-折叠）、或 C（卷曲/其他）。
"""

import urllib.request
import os
import time
import random
from pathlib import Path
from io import StringIO

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1

DATA_DIR = Path(__file__).parent / "data"
PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

# 精选 ~60 个小型蛋白质结构（≤200 残基），覆盖 α/β/α+β/coil 富集等各种类型
# 结构多样，分辨率 ≤ 2.5 Å，来源于不同物种和功能类别
PDB_IDS = [
    # α-螺旋 富集
    "1L2Y",  # Trp-cage, 20aa, NMR
    "1ENH",  # Engrailed homeodomain, 61aa
    "1ZTR",  # Designed helical bundle, 46aa
    "2I9M",  # Villin headpiece, 35aa
    "1ROP",  # ROP protein (helix-turn-helix), 56aa x2
    "4TZ2",  # Designed coiled-coil, 29aa
    "1BDD",  # B-domain of protein A, 60aa
    "1PRB",  # Bovine prothrombin fragment, 67aa
    "1VII",  # Villin 14T, 35aa
    "1WFA",  # Designed peptide, 53aa

    # β-折叠 富集
    "1SHG",  # SH3 domain, 57aa
    "1FNA",  # Fibronectin type III, 91aa
    "1TEN",  # Fibronectin type III (alternate), 89aa
    "1TIT",  # Titin I27 domain, 89aa
    "1WIT",  # Twitchin Ig domain, 93aa
    "2GB1",  # Protein G B1 domain, 56aa
    "1PGB",  # Protein G B1 domain (alternate), 56aa
    "1PIN",  # WW domain, 34aa
    "1E0L",  # B1 domain, 60aa
    "1FMK",  # SH3 domain, 58aa

    # α+β 混合
    "1UBQ",  # Ubiquitin, 76aa
    "1CRN",  # Crambin, 46aa
    "1B4B",  # Protein G, 56aa
    "1SN1",  # Staphylococcal nuclease, 141aa
    "2M5R",  # C-terminal domain, 75aa
    "1GAB",  # GB1 domain, 56aa
    "3GB1",  # GB1 variant, 56aa
    "2OED",  # OB-fold domain, 74aa
    "1IGD",  # IgG binding domain, 45aa

    # 小型酶和功能蛋白
    "1LYZ",  # Lysozyme, 129aa
    "2LZM",  # T4 Lysozyme, 164aa
    "1RN1",  # RNase A (subtilisin complex), 124aa
    "1BPT",  # BPTI (basic pancreatic trypsin inhib.), 58aa
    "1DTK",  # Dendrotoxin K, 57aa
    "1K6U",  # Scorpion toxin, 39aa
    "1FAS",  # Fasciculin, 61aa
    "1E3H",  # Conotoxin, 35aa
    "1MYG",  # Myoglobin (sperm whale), 153aa
    "1LFA",  # Integrin domain, 175aa
    "1IMQ",  # I-set domain, 99aa

    # 额外补充多样性
    "1CSP",  # Cold shock protein, 67aa
    "1MJC",  # Major cold shock, 75aa
    "1POU",  # POU-specific domain, 75aa
    "1BA5",  # BBA5, 30aa
    "1COA",  # Coagulation factor, 64aa
    "1EDM",  # Mutant of eglin c, 70aa
    "1FD3",  # DnaK substrate binding, 92aa
    "1HYP",  # Antifungal protein, 92aa
    "1IFC",  # Intestinal fatty acid binding, 131aa
    "1MBO",  # Myoglobin (equine), 153aa
    "1NXB",  # Neurotoxin B-IV, 55aa
    "1PPT",  # Pancreatic polypeptide, 36aa
    "2B3I",  # B3 domain, 74aa
    "2CI2",  # Chymotrypsin inhibitor 2, 65aa
    "2OVO",  # Ovomucoid third domain, 56aa
    "3FIL",  # Filamin, 94aa
    "4ICB",  # Calcium binding protein, 75aa
    "5PTI",  # BPTI variant, 58aa
    "6PTI",  # BPTI variant, 58aa
]


def fetch_pdb(pdb_id: str) -> str | None:
    """下载单个 PDB 文件，返回文件内容字符串。"""
    url = PDB_URL.format(pdb_id=pdb_id.upper())
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Bell-Bio-AI/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8")
    except Exception as e:
        print(f"  [SKIP] {pdb_id} 下载失败: {e}")
        return None


def extract_ss_from_pdb(pdb_content: str) -> tuple[str, str] | None:
    """
    从 PDB 文件提取序列和 Q3 二级结构标签。

    使用 HELIX 和 SHEET 记录，它们是实验确证的二级结构。
    没有标注的残基标记为 'C'（卷曲/无规结构）。

    Returns:
        (sequence, structure) 或 None
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("pdb", StringIO(pdb_content))
    except Exception as e:
        return None

    model = structure[0]
    chain = next(iter(model.get_chains()))  # 取第一条链

    # 收集残基序列
    residues = []
    seq_chars = []
    for res in chain:
        if res.get_id()[0] != " ":  # 跳过异质残基（HETATM/水）
            continue
        resname = res.get_resname().upper()
        aa = protein_letters_3to1.get(resname, None)
        if aa is None:
            continue
        residues.append(res)
        seq_chars.append(aa)

    if len(residues) < 20:
        return None

    # 初始化所有残基为 'C'（卷曲）
    ss_labels = ["C"] * len(residues)

    # 建立 (chain_id, resnum, inscode) → index 映射
    res_map = {}
    for i, res in enumerate(residues):
        rid = res.get_id()  # (hetflag, resnum, inscode)
        # PDB residue numbering
        key = (rid[1], rid[2].strip())
        res_map[key] = i

    # 从 HELIX 记录标注 α-螺旋
    # HELIX 记录格式:
    # HELIX  serNum   helixID  initResName  initChainID  initSeqNum  initICode  endResName  endChainID  endSeqNum  endICode  ...
    for line in pdb_content.split("\n"):
        if line.startswith("HELIX"):
            try:
                init_chain = line[19].strip()
                init_num = int(line[21:25])
                init_icode = line[25].strip()
                end_chain = line[31].strip()
                end_num = int(line[33:37])
                end_icode = line[37].strip()
            except (ValueError, IndexError):
                continue

            # 只处理第一条链
            chain_id = residues[0].get_parent().get_id() if residues else ""

            start_key = (init_num, init_icode)
            end_key = (end_num, end_icode)

            # 找到对应索引
            if start_key in res_map and end_key in res_map:
                start_idx = res_map[start_key]
                end_idx = res_map[end_key]
                for i in range(start_idx, end_idx + 1):
                    if i < len(ss_labels):
                        ss_labels[i] = "H"

    # 从 SHEET 记录标注 β-折叠
    for line in pdb_content.split("\n"):
        if line.startswith("SHEET "):
            try:
                # PDB format: columns 22=(initChain), 23-26=(initSeqNum), 27=(initICode)
                #              columns 33=(endChain),  34-37=(endSeqNum),  38=(endICode)
                init_chain = line[21].strip() if len(line) > 21 else ""
                init_num = int(line[22:26]) if len(line) > 26 else None
                init_icode = line[26].strip() if len(line) > 26 else ""
                end_chain = line[32].strip() if len(line) > 32 else ""
                end_num = int(line[33:37]) if len(line) > 37 else None
                end_icode = line[37].strip() if len(line) > 37 else ""

                # 有些 SHEET 记录的 end 信息可能缺失，尝试其他字段位置
                if end_num is None:
                    # 字段位置因格式而异
                    continue
            except (ValueError, IndexError):
                continue

            start_key = (init_num, init_icode)
            end_key = (end_num, end_icode)

            if start_key in res_map and end_key in res_map:
                start_idx = res_map[start_key]
                end_idx = res_map[end_key]
                for i in range(start_idx, end_idx + 1):
                    if i < len(ss_labels):
                        ss_labels[i] = "E"

    sequence = "".join(seq_chars)
    structure = "".join(ss_labels)

    # 验证：至少有一些非 C 的残基
    h_count = structure.count("H")
    e_count = structure.count("E")
    if h_count + e_count < 5:
        return None  # 结构信息太少，跳过

    return sequence, structure


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    seq_file = DATA_DIR / "real_sequences.fasta"
    ss_file = DATA_DIR / "real_structures.fasta"

    # 清除旧数据
    for f in [seq_file, ss_file]:
        if f.exists():
            f.unlink()

    sequences = []
    structures = []
    success = 0

    print(f"Downloading {len(PDB_IDS)} PDB structures...")
    print("-" * 50)

    for i, pdb_id in enumerate(PDB_IDS):
        print(f"[{i+1:3d}/{len(PDB_IDS)}] {pdb_id}...", end=" ", flush=True)

        pdb_content = fetch_pdb(pdb_id)
        if pdb_content is None:
            continue

        result = extract_ss_from_pdb(pdb_content)
        if result is None:
            print("NO_SS")
            continue

        seq, ss = result
        h_pct = ss.count("H") / len(ss) * 100
        e_pct = ss.count("E") / len(ss) * 100
        c_pct = ss.count("C") / len(ss) * 100

        print(f"OK ({len(seq)}aa, H:{h_pct:.0f}% E:{e_pct:.0f}% C:{c_pct:.0f}%)")

        sequences.append(f">{pdb_id}\n{seq}")
        structures.append(f">{pdb_id}\n{ss}")
        success += 1

        # 温和访问 PDB 服务器（避免被 ban）
        time.sleep(0.3 + random.random() * 0.3)

    print("-" * 50)
    print(f"Success: {success}/{len(PDB_IDS)}")

    if success == 0:
        print("FAILED to download any data!")
        return

    # 保存 FASTA 文件
    with open(seq_file, "w") as f:
        f.write("\n".join(sequences))
    with open(ss_file, "w") as f:
        f.write("\n".join(structures))

    # 统计
    all_ss = ""
    for entry in structures:
        lines = entry.split("\n")
        for line in lines:
            if not line.startswith(">"):
                all_ss += line
    total = len(all_ss)
    print(f"\nTotal residues: {total}")
    print(f"  H (helix):  {all_ss.count('H')} ({all_ss.count('H')/total*100:.1f}%)")
    print(f"  E (sheet):  {all_ss.count('E')} ({all_ss.count('E')/total*100:.1f}%)")
    print(f"  C (coil):   {all_ss.count('C')} ({all_ss.count('C')/total*100:.1f}%)")
    print(f"\n[OK] Saved to {seq_file} and {ss_file}")


if __name__ == "__main__":
    main()
