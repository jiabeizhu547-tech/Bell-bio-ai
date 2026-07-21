import ssl, urllib.request, os, sys
ctx = ssl._create_unverified_context()

# Clean
for f in ['cb513_sequences.fasta', 'cb513_structures.fasta']:
    p = os.path.join('inference', 'data', f)
    if os.path.exists(p):
        os.remove(p)

PDB_LIST = [
    '1CRN', '1UBQ', '4HHB', '1ENH', '1LIT', '1AKI', '1HPT', '2GB1',
    '1BTA', '1CEX', '1ROP', '1SHG', '1TGX', '1WHO', '1USM', '1QYS',
    '1PGA', '1OPD', '1NLS', '1MJC', '1LZI', '1K6Z', '1JWE', '1IBS',
    '1HIL', '1FNA', '1EMV', '1CSP', '1BGF', '1A6M'
]

aa_map = {
    'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G',
    'HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N',
    'PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V',
    'TRP':'W','TYR':'Y'
}

saved = 0
total_residues = 0
for pdb_id in PDB_LIST:
    try:
        url = 'https://files.rcsb.org/download/%s.pdb' % pdb_id
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        resp = urllib.request.urlopen(req, timeout=15, context=ctx)
        lines = resp.read().decode(errors='replace').split('\n')

        seq_3letter, helices, sheets = [], [], []
        for line in lines:
            if line.startswith('SEQRES') and line[11:12] == 'A':
                seq_3letter.extend(line[19:70].split())
            elif line.startswith('HELIX'):
                helices.append((int(line[21:25].strip()), int(line[33:37].strip())))
            elif line.startswith('SHEET'):
                sheets.append((int(line[22:26].strip()), int(line[33:37].strip())))

        if not seq_3letter:
            continue

        seq = ''.join(aa_map.get(aa, 'X') for aa in seq_3letter)
        ss = ['C'] * len(seq)
        for s, e in helices:
            for i in range(s - 1, min(e, len(ss))):
                ss[i] = 'H'
        for s, e in sheets:
            for i in range(s - 1, min(e, len(ss))):
                ss[i] = 'E'
        ss_str = ''.join(ss)

        with open('inference/data/cb513_sequences.fasta', 'a') as f:
            f.write('>%s\n%s\n' % (pdb_id, seq))
        with open('inference/data/cb513_structures.fasta', 'a') as f:
            f.write('>%s\n%s\n' % (pdb_id, ss_str))

        h, e, c = ss_str.count('H'), ss_str.count('E'), ss_str.count('C')
        print('%s: %daa H=%d E=%d C=%d' % (pdb_id, len(seq), h, e, c))
        saved += 1
        total_residues += len(seq)
    except Exception as ex:
        print('%s: error - %s' % (pdb_id, str(ex)[:60]))

print('\nSaved: %d sequences, %d residues' % (saved, total_residues))
