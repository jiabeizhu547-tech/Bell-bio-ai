import sys, os, json
_base = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _base)
_inf = os.path.join(_base, 'inference')
if _inf not in sys.path:
    sys.path.insert(0, _inf)
DATA_DIR = os.path.join(_base, 'inference', 'data')

from inference import predict_secondary_structure
import inference as inf_mod
import warnings
warnings.filterwarnings('ignore')

def load_data():
    names, seqs, ss = [], [], []
    with open(os.path.join(DATA_DIR, 'cb513_sequences.fasta')) as f:
        name, s = '', []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if s: names.append(name); seqs.append(''.join(s))
                name = line[1:].split()[0]; s = []
            elif line: s.append(line)
        if s: names.append(name); seqs.append(''.join(s))
    with open(os.path.join(DATA_DIR, 'cb513_structures.fasta')) as f:
        for line in f:
            line = line.strip()
            if not line.startswith('>'): ss.append(line)
    return names, seqs, ss

def evaluate(preds, labels):
    correct = sum(1 for p,l in zip(preds,labels) if p==l)
    q3 = correct/len(preds)*100 if preds else 0
    f1 = {}
    for cls in 'HEC':
        tp = sum(1 for p,l in zip(preds,labels) if p==cls and l==cls)
        fp = sum(1 for p,l in zip(preds,labels) if p==cls and l!=cls)
        fn = sum(1 for p,l in zip(preds,labels) if p!=cls and l==cls)
        pre = tp/(tp+fp) if (tp+fp)>0 else 0
        rec = tp/(tp+fn) if (tp+fn)>0 else 0
        f1[cls] = 2*pre*rec/(pre+rec)*100 if (pre+rec)>0 else 0
    return q3, f1

def run_config(name, w1, w2, smooth):
    inf_mod.ENSEMBLE_W1 = w1
    inf_mod.ENSEMBLE_W2 = w2
    inf_mod.SMOOTH_MIN_RUN = 3 if smooth else 999
    names, seqs, true_ss = load_data()
    preds_all, labels_all = '', ''
    for i, seq in enumerate(seqs):
        try:
            r = predict_secondary_structure(seq[:256])
            pl = min(len(r['structure']), len(true_ss[i]))
            preds_all += r['structure'][:pl]
            labels_all += true_ss[i][:pl]
        except:
            pass
    inf_mod.ENSEMBLE_W1 = 0.4; inf_mod.ENSEMBLE_W2 = 0.6; inf_mod.SMOOTH_MIN_RUN = 3
    q3, f1 = evaluate(preds_all, labels_all)
    return {'q3':round(q3,2),'H_f1':round(f1['H'],2),'E_f1':round(f1['E'],2),'C_f1':round(f1['C'],2)}

print('Ablation study (30 seqs, ~3500 residues)')
configs = [
    ('ensemble', 0.4, 0.6, True),
    ('v1_only', 1.0, 0.0, True),
    ('esm2_only', 0.0, 1.0, True),
    ('no_smooth', 0.4, 0.6, False),
]
results = {}
for name, w1, w2, smooth in configs:
    print(name + '...')
    r = run_config(name, w1, w2, smooth)
    results[name] = r
    print('  Q3=%.2f H_F1=%.2f E_F1=%.2f C_F1=%.2f' % (r['q3'],r['H_f1'],r['E_f1'],r['C_f1']))

with open(os.path.join(DATA_DIR, 'ablation_results_30seq.json'), 'w') as f:
    json.dump(results, f, indent=2)
print('Saved!')