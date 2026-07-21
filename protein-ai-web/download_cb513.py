"""
CB513 数据集下载工具
尝试多个源自动下载，如果都失败会给出手动下载指引
"""
import sys, os, ssl, urllib.request, io, gzip

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference", "data")
SEQ_FILE = os.path.join(DATA_DIR, "cb513_sequences.fasta")
SS_FILE = os.path.join(DATA_DIR, "cb513_structures.fasta")

SOURCES = [
    # DTU CBS (NetSurfP official) - HTTPS, need SSL bypass
    {
        "url": "https://download.services.cbs.dtu.dk/download/NetSurfP-2.0/datasets/CB513.tar.gz",
        "type": "tar_gz",
        "ssl": False,
    },
    # Backup: PSIPRED website
    {
        "url": "http://bioinfadmin.cs.ucl.ac.uk/downloads/psipred/CB513.nr",
        "type": "raw",
        "ssl": True,
    },
]


def extract_cb513_from_tar(data: bytes):
    """Extract sequence/structure from tar.gz"""
    import tarfile
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        names = []
        for member in tar.getmembers():
            if member.name.endswith(".fasta") or member.name.endswith(".seq"):
                f = tar.extractfile(member)
                if f:
                    content = f.read().decode("utf-8", errors="replace")
                    yield member.name, content


def parse_cb513_lines(text: str) -> tuple:
    """Parse CB513 text file into names, sequences, structures"""
    names, seqs, structures = [], [], []
    current_name = ""
    current_seq = []
    
    for line in text.strip().split("\\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(">"):
            if current_seq and len(current_seq) == 2:
                names.append(current_name)
                seqs.append(current_seq[0])
                structures.append(current_seq[1])
            elif current_seq and len(current_seq) == 1:
                names.append(current_name)
                seqs.append(current_seq[0])
                structures.append("C" * len(current_seq[0]))
            current_name = line[1:].split()[0]
            current_seq = []
        else:
            # Only keep valid SS chars
            cleaned = "".join(c for c in line.upper() if c in "HEC")
            if cleaned:
                current_seq.append(cleaned)
    
    if current_seq and len(current_seq) >= 1:
        names.append(current_name)
        seqs.append(current_seq[0] if len(current_seq) >= 1 else "")
        structures.append(current_seq[1] if len(current_seq) >= 2 else "C" * len(current_seq[0]))
    
    return names, seqs, structures


def download():
    """Download CB513 from multiple sources"""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    for source in SOURCES:
        url = source["url"]
        print(f"Trying: {url}")
        try:
            ctx = None if source.get("ssl", True) else ssl._create_unverified_context()
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            resp = urllib.request.urlopen(req, timeout=60, context=ctx)
            data = resp.read()
            print(f"  Downloaded: {len(data)} bytes")
            
            if source["type"] == "tar_gz":
                # Extract from tar.gz
                import tarfile
                try:
                    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
                        for name, content in extract_cb513_from_tar(data):
                            print(f"  Found: {name}")
                            n, s, ss = parse_cb513_lines(content)
                            if len(n) > 0:
                                names, seqs, structures = n, s, ss
                                # Save
                                with open(SEQ_FILE, "w", encoding="utf-8") as f:
                                    for nn, ss2 in zip(names, seqs):
                                        f.write(f">{nn}\\n{ss2}\\n")
                                with open(SS_FILE, "w", encoding="utf-8") as f:
                                    for nn, ss2 in zip(names, structures):
                                        f.write(f">{nn}\\n{ss2}\\n")
                                print(f"  Saved: {len(names)} sequences to {DATA_DIR}")
                                return True
                except tarfile.ReadError:
                    # Not a tar file, try parsing directly
                    n, s, ss = parse_cb513_lines(data.decode("utf-8", errors="replace"))
                    if len(n) > 0:
                        _save(n, s, ss)
                        return True
            else:
                n, s, ss = parse_cb513_lines(data.decode("utf-8", errors="replace"))
                if len(n) > 0:
                    _save(n, s, ss)
                    return True
                    
        except Exception as e:
            print(f"  Failed: {str(e)[:80]}")
    
    return False


def _save(names, seqs, structures):
    with open(SEQ_FILE, "w", encoding="utf-8") as f:
        for n, s in zip(names, seqs):
            f.write(f">{n}\\n{s}\\n")
    with open(SS_FILE, "w", encoding="utf-8") as f:
        for n, s in zip(names, structures):
            f.write(f">{n}\\n{s}\\n")
    print(f"Saved: {len(names)} sequences to {DATA_DIR}")


def manual_instructions():
    print("\\n" + "=" * 50)
    print("  自动下载失败。请手动下载 CB513 数据集：")
    print("=" * 50)
    print("\\n方法 1: 从 GitHub 下载")
    print("  1. 打开 https://github.com/jianlin-cheng/CB513")
    print("  2. 下载 cb513.fasta")
    print("  3. 另存为 inference/data/cb513_sequences.fasta")
    print("  4. 下载 cb513.ss (如果存在)")
    print("    或自行标注结构")
    print("\\n方法 2: 从 PSIPRED 下载")
    print("  1. 打开 http://bioinf.cs.ucl.ac.uk/psipred/")
    print("  2. 下载 CB513 数据集")
    print("  3. 解压到 inference/data/")
    print("\\n方法 3: 用其他工具生成")
    print("  pip install bio_embeddings")
    print("  bio_embeddings download cb513")
    print("\\n完成后文件应该放在:")
    print(f"  {SEQ_FILE}")
    print(f"  {SS_FILE}")
    print("\\n然后运行:")
    print("  python evaluate.py")
    print("=" * 50)


if __name__ == "__main__":
    if os.path.exists(SEQ_FILE) and os.path.exists(SS_FILE):
        with open(SEQ_FILE) as f:
            n = sum(1 for l in f if l.startswith(">"))
        print(f"CB513 already exists: {n} sequences")
        print(f"  {SEQ_FILE}")
        print(f"  {SS_FILE}")
    else:
        success = download()
        if not success:
            manual_instructions()
