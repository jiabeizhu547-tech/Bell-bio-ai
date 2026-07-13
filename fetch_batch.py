"""
从 PDB 数据库批量搜索并下载蛋白质结构，提取 Q3 二级结构标注。

使用 PDB REST API 搜索 + 下载，获取数百个高质量的蛋白质结构。
"""

import urllib.request
import urllib.parse
import json
import time
import random
from pathlib import Path
from io import StringIO

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1

DATA_DIR = Path(__file__).parent / "data"
PDB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
PDB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"


def search_pdb_ids(max_per_query: int = 500) -> list[str]:
    """
    通过 PDB REST API 搜索蛋白质结构。

    使用多种搜索策略组合，获取多样化的结构数据集。
    """
    all_ids = []

    # 策略1: X-ray，少链结构（排前面的是高分辨率的）
    # 策略2: NMR 结构
    queries = [
        ("X-ray (<=2 chains)", {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    {
                        "type": "terminal", "service": "text",
                        "parameters": {"attribute": "exptl.method", "operator": "exact_match", "value": "X-RAY DIFFRACTION"}
                    },
                    {
                        "type": "terminal", "service": "text",
                        "parameters": {"attribute": "rcsb_entry_info.polymer_entity_count", "operator": "less_or_equal", "value": 2}
                    },
                ],
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {"start": 0, "rows": max_per_query},
                "results_content_type": ["experimental"],
            },
        }),
        ("NMR (<=3 chains)", {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    {
                        "type": "terminal", "service": "text",
                        "parameters": {"attribute": "exptl.method", "operator": "exact_match", "value": "SOLUTION NMR"}
                    },
                    {
                        "type": "terminal", "service": "text",
                        "parameters": {"attribute": "rcsb_entry_info.polymer_entity_count", "operator": "less_or_equal", "value": 3}
                    },
                ],
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {"start": 0, "rows": max_per_query},
                "results_content_type": ["experimental"],
            },
        }),
    ]

    for name, query_payload in queries:
        print(f"Searching PDB: {name}...")
        data = json.dumps(query_payload).encode("utf-8")
        req = urllib.request.Request(
            PDB_SEARCH_URL,
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Bell-Bio-AI/1.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"  Failed: {e}")
            continue

        ids = []
        for entry in result.get("result_set", []):
            pid = entry.get("identifier", "")
            if pid and len(pid) == 4:
                ids.append(pid.upper())

        print(f"  Found {len(ids)} entries")
        all_ids.extend(ids)

    # 去重
    seen = set()
    unique = []
    for pid in all_ids:
        if pid not in seen:
            seen.add(pid)
            unique.append(pid)

    print(f"  Total unique: {len(unique)}")
    return unique


def fetch_pdb(pdb_id: str) -> str | None:
    """下载单个 PDB 文件。"""
    url = PDB_DOWNLOAD_URL.format(pdb_id=pdb_id.upper())
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Bell-Bio-AI/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8")
    except Exception as e:
        return None


def extract_ss(pdb_content: str) -> tuple[str, str] | None:
    """
    从 PDB 文件提取 (序列, Q3结构标签)。

    使用 HELIX/SHEET 记录标注 H/E，其余为 C。
    只取第一条蛋白链。
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("pdb", StringIO(pdb_content))
    except Exception:
        return None

    model = structure[0]

    # 收集第一条有效蛋白链
    residues = []
    seq_chars = []
    for chain in model.get_chains():
        for res in chain:
            if res.get_id()[0] != " ":
                continue
            aa = protein_letters_3to1.get(res.get_resname().upper(), None)
            if aa is None:
                continue
            residues.append(res)
            seq_chars.append(aa)
        # 只取第一条链
        break

    if len(residues) < 20:
        return None

    # 初始化全部为 C
    ss_labels = ["C"] * len(residues)

    # 建立残基编号 → 索引映射
    res_map = {}
    for i, res in enumerate(residues):
        rid = res.get_id()
        res_map[(rid[1], rid[2].strip())] = i

    # HELIX 记录 → H
    for line in pdb_content.split("\n"):
        if not line.startswith("HELIX"):
            continue
        try:
            init_num = int(line[21:25])
            init_icode = line[25].strip()
            end_num = int(line[33:37])
            end_icode = line[37].strip()
        except (ValueError, IndexError):
            continue

        start_key = (init_num, init_icode)
        end_key = (end_num, end_icode)
        if start_key in res_map and end_key in res_map:
            for i in range(res_map[start_key], res_map[end_key] + 1):
                ss_labels[i] = "H"

    # SHEET 记录 → E
    for line in pdb_content.split("\n"):
        if not line.startswith("SHEET "):
            continue
        try:
            init_num = int(line[22:26])
            init_icode = line[26].strip()
            end_num = int(line[33:37])
            end_icode = line[37].strip()
        except (ValueError, IndexError):
            continue

        start_key = (init_num, init_icode)
        end_key = (end_num, end_icode)
        if start_key in res_map and end_key in res_map:
            for i in range(res_map[start_key], res_map[end_key] + 1):
                ss_labels[i] = "E"

    sequence = "".join(seq_chars)
    structure = "".join(ss_labels)

    if structure.count("H") + structure.count("E") < 5:
        return None

    return sequence, structure


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 搜索 PDB 获取结构列表
    pdb_ids = search_pdb_ids(max_per_query=500)

    if not pdb_ids:
        print("Search returned no results, using backup list...")
        return

    # 去重：已有的 ID 不再下载
    existing_ids = set()
    seq_file = DATA_DIR / "real_sequences.fasta"
    if seq_file.exists():
        with open(seq_file) as f:
            for line in f:
                if line.startswith(">"):
                    existing_ids.add(line[1:].strip())

    new_ids = [p for p in pdb_ids if p not in existing_ids]
    print(f"  Existing: {len(existing_ids)}, New to fetch: {len(new_ids)}")

    sequences = []
    structures = []
    success = 0
    skip_count = 0

    print(f"\nDownloading up to {len(new_ids)} new structures...")
    print("-" * 50)

    for i, pdb_id in enumerate(new_ids):
        print(f"[{i+1:4d}/{len(new_ids)}] {pdb_id}...", end=" ", flush=True)

        pdb_content = fetch_pdb(pdb_id)
        if pdb_content is None:
            print("SKIP")
            skip_count += 1
            continue

        result = extract_ss(pdb_content)
        if result is None:
            print("NO_SS")
            skip_count += 1
            continue

        seq, ss = result
        h_pct = ss.count("H") / len(ss) * 100
        e_pct = ss.count("E") / len(ss) * 100
        c_pct = ss.count("C") / len(ss) * 100

        print(f"OK ({len(seq):4d}aa, H:{h_pct:3.0f}% E:{e_pct:3.0f}% C:{c_pct:3.0f}%)")

        sequences.append(f">{pdb_id}\n{seq}")
        structures.append(f">{pdb_id}\n{ss}")
        success += 1

        # 每 50 个保存一次（断点续传）
        if success % 50 == 0:
            _append_fasta(sequences[-50:], structures[-50:])

        # 温和访问
        time.sleep(0.2 + random.random() * 0.2)

    print("-" * 50)
    print(f"New success: {success}, Skipped: {skip_count}")

    # 追加剩余数据
    if sequences:
        remaining_start = (success // 50) * 50
        _append_fasta(sequences[remaining_start:], structures[remaining_start:])

    # 统计全部数据
    _print_stats()


def _append_fasta(seqs: list[str], structs: list[str]):
    """追加数据到 FASTA 文件。"""
    seq_file = DATA_DIR / "real_sequences.fasta"
    ss_file = DATA_DIR / "real_structures.fasta"

    with open(seq_file, "a") as f:
        f.write("\n" + "\n".join(seqs))
    with open(ss_file, "a") as f:
        f.write("\n" + "\n".join(structs))


def _print_stats():
    """打印整体统计。"""
    seq_file = DATA_DIR / "real_sequences.fasta"
    ss_file = DATA_DIR / "real_structures.fasta"

    count = 0
    all_ss = ""
    with open(ss_file) as f:
        content = f.read()
        for block in content.strip().split("\n>"):
            lines = block.strip().split("\n")
            for line in lines:
                if not line.startswith(">") and line.strip():
                    all_ss += line.strip()
                    count += 1

    total = len(all_ss)
    if total == 0:
        return

    print(f"\n{'='*50}")
    print(f"Total proteins: {count}")
    print(f"Total residues: {total}")
    print(f"  H (helix):  {all_ss.count('H'):6d} ({all_ss.count('H')/total*100:5.1f}%)")
    print(f"  E (sheet):  {all_ss.count('E'):6d} ({all_ss.count('E')/total*100:5.1f}%)")
    print(f"  C (coil):   {all_ss.count('C'):6d} ({all_ss.count('C')/total*100:5.1f}%)")
    print(f"Data files: {seq_file}, {ss_file}")


if __name__ == "__main__":
    main()
