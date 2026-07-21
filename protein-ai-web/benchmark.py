import sys, os, json, random

_base = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _base)
DATA_DIR = os.path.join(_base, 'inference', 'data')

# Load truth labels
with open(os.path.join(DATA_DIR, 'cb513_structures.fasta')) as f:
    data = f.read()
parts = [p for p in data.strip().split('>') if p.strip()]
true_ss_all = []
for part in parts:
    lines = part.strip().split('\n')
    if len(lines) >= 2:
        true_ss_all.append(lines[-1])

true_all = ''.join(true_ss_all)
total = len(true_all)

# Baseline: always predict C
correct_c = sum(1 for t in true_all if t == 'C')
q3_c = correct_c / total * 100

# Baseline: always predict H  
correct_h = sum(1 for t in true_all if t == 'H')
q3_h = correct_h / total * 100

# Baseline: always predict E
correct_e = sum(1 for t in true_all if t == 'E')
q3_e = correct_e / total * 100

# Baseline: random uniform
random.seed(42)
random_pred = ''.join(random.choice('HEC') for _ in range(total))
correct_rand = sum(1 for p, t in zip(random_pred, true_all) if p == t)
q3_rand = correct_rand / total * 100

print('Trivial Baselines (30 seqs, %d residues):' % total)
print('  Always C:   %5.2f%%' % q3_c)
print('  Always H:   %5.2f%%' % q3_h)
print('  Always E:   %5.2f%%' % q3_e)
print('  Random:     %5.2f%%' % q3_rand)
print()
print('Class distribution in test set:')
print('  H=%d (%5.2f%%)' % (true_all.count('H'), true_all.count('H')/total*100))
print('  E=%d (%5.2f%%)' % (true_all.count('E'), true_all.count('E')/total*100))
print('  C=%d (%5.2f%%)' % (true_all.count('C'), true_all.count('C')/total*100))

# Combine with ablation results
with open(os.path.join(DATA_DIR, 'ablation_results_30seq.json')) as f:
    results = json.load(f)

results['always_C'] = {'q3': round(q3_c,2), 'H_f1': 0.0, 'E_f1': 0.0, 'C_f1': round(2*q3_c*100/(q3_c+100) if q3_c+100 > 0 else 0,2)}
results['always_H'] = {'q3': round(q3_h,2), 'H_f1': 0.0, 'E_f1': 0.0, 'C_f1': 0.0}
results['always_E'] = {'q3': round(q3_e,2), 'H_f1': 0.0, 'E_f1': 0.0, 'C_f1': 0.0}
results['random']   = {'q3': round(q3_rand,2), 'H_f1': 0.0, 'E_f1': 0.0, 'C_f1': 0.0}

with open(os.path.join(DATA_DIR, 'benchmark_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print('\nFull Comparison:')
print('%-16s %8s' % ('Method', 'Q3'))
print('-' * 28)
for name, r in sorted(results.items(), key=lambda x: x[1]['q3'], reverse=True):
    print('%-16s %7.2f%%' % (name, r['q3']))
