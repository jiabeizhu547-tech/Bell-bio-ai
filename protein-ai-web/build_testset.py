
import ssl, urllib.request, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ctx = ssl._create_unverified_context()
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inference', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# Well-known PDB entries with reliable structures
PDB_LIST = ['1UBQ', '4HHB', '1HPT', '1AKI', '1CEX', '1LIT', '2GB1', '1BTA', '1ENH', '1CRN']

aa_map = {
    'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G',
    'HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N',
    'PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V',
    'TRP':'W','TYR':'Y'
}

saved = 0
for pdb_id in PDB_LIST:
    try:
        url = 'https://files.rcsb.org/download/' + pdb_id + '.pdb'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        resp = urllib.request.urlopen(req, timeout=15, context=ctx)
        lines = resp.read().decode(errors='replace').split('\n')
        
        seq_3letter = []
        helices = []
        sheets = []
        for line in lines:
            if line.startswith('SEQRES') and line[11:12] == 'A':
                seq_3letter.extend(line[19:70].split())
            elif line.startswith('HELIX'):
                s = int(line[21:25].strip())
                e = int(line[33:37].strip())
                helices.append((s, e))
            elif line.startswith('SHEET'):
                s = int(line[22:26].strip())
                e = int(line[33:37].strip())
                sheets.append((s, e))
        
        if not seq_3letter:
            continue
        
        seq = ''.join(aa_map.get(aa, 'X') for aa in seq_3letter)
        ss = ['C'] * len(seq)
        for start, end in helices:
            for i in range(start-1, min(end, len(ss))):
                ss[i] = 'H'
        for start, end in sheets:
            for i in range(start-1, min(end, len(ss))):
                ss[i] = 'E'
        ss_str = ''.join(ss)
        
        with open(os.path.join(DATA_DIR, 'cb513_sequences.fasta'), 'a', encoding='utf-8') as f:
            f.write('>' + pdb_id + '\n' + seq + '\n')
        with open(os.path.join(DATA_DIR, 'cb513_structures.fasta'), 'a', encoding='utf-8') as f:
            f.write('>' + pdb_id + '\n' + ss_str + '\n')
        
        h = ss_str.count('H')
        e = ss_str.count('E')
        c_val = ss_str.count('C')
        print(pdb_id + ': ' + str(len(seq)) + ' aa, H=' + str(h) + ' E=' + str(e) + ' C=' + str(c_val))
        saved += 1
    except Exception as ex:
        print(pdb_id + ': error - ' + str(ex)[:60])

print('\nSaved ' + str(saved) + ' sequences to ' + DATA_DIR)
