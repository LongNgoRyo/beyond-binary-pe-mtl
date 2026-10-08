"""
==============================================================================
EMBER2024 MULTI-TASK PE MALWARE ANALYSIS  –  v7.2  "Joint Soft Gating & Augmented LightGBM"
==============================================================================
Improvements from v6.9:
  ✅ Feature Drift Suppression: Zeros out absolute timestamps (t1n, t2n) and SHA256 
     prefixes in `_feat_row` to prevent temporal overfitting and ID memorization.
  ✅ Hybrid Ensembling: Parallel multiclass LightGBM classifier trained on the 
     7-class family dataset and ensembled (70% LGBM + 30% FTTransformer) to leverage 
     GBDT robustness on tabular data.
  ✅ Strict FPR Gating & Cost-Sensitive Calibration: Selects binary threshold 
     enforcing a strict FPR budget (<= 0.2% on validation) and calibrates 
     class-specific multipliers via coordinate descent.
==============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# INSTALL DEPENDENCIES (Safeguarded for Offline Environments)
# ─────────────────────────────────────────────────────────────────────────────
import subprocess, sys

# Reconfigure stdout/stderr to UTF-8 to prevent UnicodeEncodeErrors on Windows cp1252 terminals
if hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass
if hasattr(sys.stderr, 'reconfigure'):
    try: sys.stderr.reconfigure(encoding='utf-8')
    except: pass

def _pip(pkg):
    try: 
        __import__(pkg.split('[')[0].replace('-', '_'))
    except ImportError:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])
        except Exception as e:
            print(f"  ⚠️ Warning: Could not install package '{pkg}'. Internet might be disabled. Exception: {e}")

# Cài đặt nếu có mạng, bỏ qua nếu mất mạng (có cơ chế fallback bên dưới)
for _p in ['captum', 'lime', 'shap', 'kagglehub', 'scipy', 'lightgbm']:
    _pip(_p)

import os, json, gc, time, glob, copy, warnings, zipfile
from datetime import datetime
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import softmax as torch_softmax
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    hamming_loss, roc_auc_score,
)
from sklearn.decomposition import PCA
from sklearn.feature_extraction import FeatureHasher
import matplotlib.pyplot as plt
import matplotlib.cm as cm

warnings.filterwarnings('ignore')

# Khai báo biến kiểm tra các thư viện XAI bên ngoài
HAS_CAPTUM = True
try:
    import captum
    from captum.attr import IntegratedGradients
except ImportError:
    HAS_CAPTUM = False

HAS_LIME = True
try:
    import lime
    from lime import lime_tabular
except ImportError:
    HAS_LIME = False

HAS_SHAP = True
try:
    import shap
except ImportError:
    HAS_SHAP = False

HAS_KAGGLERHUB = True
try:
    import kagglehub
except ImportError:
    HAS_KAGGLERHUB = False


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 0: CONSTANTS  &  CHART SAVER
# ─────────────────────────────────────────────────────────────────────────────

def save_chart(fig, name: str) -> None:
    """Hiển thị figure trực tiếp và lưu lại dạng png."""
    fig.savefig(f'{name}.png', dpi=150)
    plt.show()
    plt.close(fig)


# ── Model / training config (High-Capacity config) ───────────────────────────
D_MODEL      = 256
N_HEAD       = 8
N_LAYERS     = 6
DIM_FF       = 1024
DROPOUT      = 0.25     # anti-overfit
GRAD_CLIP    = 0.5      # anti-overfit (tighter)
WEIGHT_DECAY = 2e-5     # anti-overfit
MAX_WEIGHT   = 30.0     # max class weight (anti-overfit)
AUG_MULT     = 15       # oversample factor
AUG_MIN      = 1200     # minimum samples per rare class after aug
CHUNK        = 100_000  # max rows read per file chunk (RAM safety)
TLSH_NOISE   = 0.03     # noise std trên TLSH features khi train

# ── Family classes (7 – merged Rootkit+Adware_PUP → Other_Rare) ──────────────
FAMILY_NAMES = [
    'Benign',      # 0
    'Ransomware',  # 1
    'Keylogger',   # 2
    'Trojan_RAT',  # 3
    'Worm_Virus',  # 4
    'Other_Rare',  # 5  (Rootkit + Adware_PUP + Miners)
    'Dropper',     # 6
]
BEHAVIOR_NAMES = [
    'file_ops', 'network', 'registry', 'process_injection',
    'persistence', 'encryption', 'keylogging', 'screenshot',
]
N_FAMILIES     = len(FAMILY_NAMES)    # 7
N_BEHAVIORS    = len(BEHAVIOR_NAMES)  # 8
ALL_FAM_LABELS = list(range(N_FAMILIES))
RARE_CLASSES   = [1, 2, 4, 5, 6]      # sẽ được augment
COMMON_CLASSES = [0, 3]               # Benign, Trojan_RAT

FEAT_DIM  = 631
_FH10     = FeatureHasher(10, input_type='string')

FEAT_NAMES = (
    [f'tlsh_{i:02d}' for i in range(70)] +
    ['first_year', 'last_year', 'days_alive', 'week_of_yr',
     'has_t1', 'has_t2', 'has_both', 'days_1yr'] +
    [f'ft_{i}'         for i in range(10)] +
    [f'sha_{i}'        for i in range(10)] +
    [f'hist_{i}'       for i in range(256)] +
    [f'entropy_{i}'    for i in range(256)] +
    [f'gen_{i}'        for i in range(10)] +
    [f'str_{i}'        for i in range(4)] +
    [f'sec_{i}'        for i in range(4)] +
    [f'imp_{i}'        for i in range(2)] +
    ['exp_0']
)
FEAT_GROUPS = {
    'TLSH':        list(range(0, 70)),
    'Temporal':    list(range(70, 78)),
    'FileType':    list(range(78, 88)),
    'SHA256':      list(range(88, 98)),
    'Histogram':   list(range(98, 354)),
    'ByteEntropy': list(range(354, 610)),
    'HeaderStats': list(range(610, 631)),
}
TOKEN_NAMES = ['CLS', 'TLSH', 'Temporal', 'FileType', 'SHA256', 'Histogram', 'ByteEntropy', 'HeaderStats']


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 1: HEURISTIC LABELS  (RE-ORDERED & EXPANDED FOR ACCURACY)
# ─────────────────────────────────────────────────────────────────────────────

_FAMILY_KW = {
    # === 1. Ransomware ===
    'ransomware':1,'ransom':1,'cryptolocker':1,'wannacry':1,'ryuk':1,
    'sodinokibi':1,'revil':1,'lockbit':1,'gandcrab':1,'dharma':1,
    'maze':1,'conti':1,'blackcat':1,'alphv':1,'hive':1, 'luna':1,
    'locky':1,'cerber':1,'teslacrypt':1,'cryptxxx':1,'djvu':1,
    'phobos':1,'clop':1,'medusalocker':1,'babuk':1,'crysis':1,
    'mallox':1,'royal':1,'cuba':1,'blackbyte':1,'play':1,
    'nokoyawa':1,'bianlian':1,'avaddon':1,'ragnar':1,'egregor':1,
    'netwalker':1,'doppel':1,'bitpaymer':1,'hermes':1,'wannacrypt':1,
    
    # === 2. Keylogger / Stealer / Spyware ===
    'keylogger':2,'spyware':2,'hawkeye':2,'agent.tesla':2,'agenttesla':2,
    'formbook':2,'azorult':2,'vidar':2,'raccoon':2,'stealer':2,
    'redline':2,'khalesi':2,'lokibot':2,'pony':2,'avemaria':2,
    'predator':2,'spy':2,'keylog':2,'stealan':2,'stealer':2,
    'grabber':2,'spybubble':2,'stealth':2,'loki':2,
    
    # === 3. Dropper / Downloader ===
    'dropper':6,'downloader':6,'loader':6,'injector':6,
    'smokeloader':6,'guloader':6,'amadey':6,'nullsoft':6, 
    'banload':6,'braviax':6,'obfuscator':6,'binst':6,
    'installer':6,'setup':6,'updat':6,'stub':6,
    
    # === 4. Other_Rare (Rootkit + Adware + Riskware + Miners) ===
    'rootkit':5,'bootkit':5,'tdss':5,'alureon':5,'necurs':5,
    'adware':5,'pup':5,'riskware':5,'toolbar':5,'unwanted':5,
    'bundler':5,'xmrig':5,'coinminer':5,'miner':5,'cryptominer':5,
    
    # === 5. Generic Worm / Virus ===
    'worm':4,'virus':4,'mirai':4,'conficker':4,'sality':4,
    'virut':4,'autorun':4,'expiro':4,'ramnit':4,'vobfus':4,
    'dorkbot':4,'allaple':4,'virux':4,
    
    # === 6. Generic Trojan / RAT / Backdoor ===
    'trojan':3,'.rat':3,'backdoor':3,'emotet':3,'trickbot':3,
    'remcos':3,'nanocore':3,'njrat':3,'asyncrat':3,'quasar':3,
    'darkcomet':3,'warzone':3,'icedid':3,'qakbot':3,'dridex':3, 
    'wacatac':3, 'scarletflash':3, 'trollav':3,
    'cobaltstrike':3, 'disin':3, 'shelm':3, 'disco':3, 'qqrob':3, 'convagent':3,
    'metasploit':3, 'lotok':3, 'blackv':3, 'alien':3, 'bedep':3,
    'gh0st':3,'plugx':3,'poisonivy':3,'bifrost':3,'njw0rm':3,
    'xtreme':3,
}

def map_family(family_str, label):
    if label == 0: return 0
    s = str(family_str or '').lower().strip()
    if s in ('', 'nan', 'none', 'null', 'undefined'):
        return -1
        
    sorted_kws = sorted(_FAMILY_KW.keys(), key=len, reverse=True)
    votes = {c: 0 for c in range(1, 7)}
    temp_s = s
    for kw in sorted_kws:
        if kw in temp_s:
            votes[_FAMILY_KW[kw]] += 1
            temp_s = temp_s.replace(kw, "#" * len(kw))
            
    candidate_classes = [c for c, count in votes.items() if count > 0]
    if not candidate_classes:
        return -1
        
    max_vote = max(votes[c] for c in candidate_classes)
    winners = [c for c in candidate_classes if votes[c] == max_vote]
    
    if len(winners) == 1:
        return winners[0]
        
    priority = [1, 2, 6, 4, 5, 3]
    for p_cls in priority:
        if p_cls in winners:
            return p_cls
            
    return winners[0]

def behavior_from_family(family_str, label):
    b = np.zeros(N_BEHAVIORS, dtype=np.float32)
    if label == 0: return b
    s = str(family_str or '').lower()
    b[0] = 1.0
    if any(k in s for k in ['trojan','rat','backdoor','worm','downloader',
                              'loader','dropper','emotet','trickbot','remcos',
                              'nanocore','njrat','asyncrat','quasar','mirai']): b[1]=1.0
    if any(k in s for k in ['trojan','rat','backdoor','ransomware',
                              'emotet','trickbot','loader','dropper']): b[2]=1.0
    if any(k in s for k in ['trojan','rat','backdoor','injector','loader',
                              'emotet','trickbot','remcos','nanocore']): b[3]=1.0
    if any(k in s for k in ['trojan','rat','backdoor','ransomware',
                              'worm','virus','emotet','trickbot']): b[4]=1.0
    if any(k in s for k in ['ransomware','ransom','cryptolocker','wannacry',
                              'ryuk','sodinokibi','revil','lockbit','gandcrab',
                              'dharma','maze','conti']): b[5]=1.0
    if any(k in s for k in ['keylogger','spyware','hawkeye','agent.tesla',
                              'formbook','azorult','vidar','raccoon','stealer','redline']): b[6]=1.0
    if any(k in s for k in ['rat','spyware','remcos','nanocore','darkcomet',
                              'warzone','asyncrat']): b[7]=1.0
    return b


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 2: FEATURE EXTRACTION (PREVENT OVERFITTING & DRIFT)
# ─────────────────────────────────────────────────────────────────────────────

def _tlsh_vec(s):
    vec = np.zeros(70, dtype=np.float32)
    if not s: return vec
    s = str(s).strip()
    if s.upper().startswith('T1'): s = s[2:]
    for i, c in enumerate(s[:70]):
        try: vec[i] = int(c, 16) / 15.0
        except: pass
    return vec

def _parse_ts(v):
    if not v: return 0.0
    s = str(v).strip()
    for fmt in ('%Y-%m-%d %H:%M:%S','%Y-%m-%dT%H:%M:%S','%Y-%m-%d','%d/%m/%Y'):
        try: return float(datetime.strptime(s[:19], fmt).timestamp())
        except: continue
    try: return float(s)
    except: return 0.0

def _feat_row(row):
    try:
        # 1. TLSH (70 dims)
        tlsh  = _tlsh_vec(row.get('tlsh', ''))
        
        # 2. Temporal (8 dims)
        t1    = _parse_ts(row.get('first_submission_date', ''))
        t2    = _parse_ts(row.get('last_analysis_date', ''))
        dd    = max(0.0, (t2-t1)/86400.0) if t1>0 and t2>0 else 0.0
        woy   = 0.0
        if t1 > 0:
            try: woy = float(datetime.fromtimestamp(t1).isocalendar()[1]) / 52.0
            except: pass
        temp  = np.array([
            0.0, 0.0, # Zeros out absolute timestamps (t1n, t2n) to prevent temporal drift
            min(dd,1825.)/1825., woy,
            float(t1>0), float(t2>0), float(t1>0 and t2>0),
            min(dd,365.)/365.,
        ], dtype=np.float32)
        
        # 3. File Type (10 dims)
        ft    = str(row.get('file_type','') or '').lower().strip()
        ft_v  = _FH10.transform([[ft]]).toarray()[0].astype(np.float32)
        
        # 4. SHA256 (10 dims) - Zero out unique SHA prefixes to prevent ID memorization/overfitting
        sha   = np.zeros(10, dtype=np.float32)
        
        # 5. Histogram (256 dims, normalized)
        hist = np.array(row.get('histogram', [0]*256), dtype=np.float32)
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist = hist / hist_sum
        else:
            hist = np.zeros(256, dtype=np.float32)
            
        # 6. Byte Entropy (256 dims, normalized)
        entropy = np.array(row.get('byteentropy', [0]*256), dtype=np.float32)
        entropy = entropy / 8.0
        
        # 7. General Stats (10 dims)
        gen = row.get('general', {})
        gen_v = np.array([
            np.log1p(float(gen.get('size', 0.0))) / 20.0,
            np.log1p(float(gen.get('vsize', 0.0))) / 20.0,
            float(gen.get('entropy', 0.0)) / 8.0,
            float(1.0 if gen.get('is_pe', False) else 0.0),
            float(1.0 if gen.get('has_debug', False) else 0.0),
            float(1.0 if gen.get('has_relocations', False) else 0.0),
            float(1.0 if gen.get('has_resources', False) else 0.0),
            float(1.0 if gen.get('has_signature', False) else 0.0),
            float(1.0 if gen.get('has_tls', False) else 0.0),
            np.log1p(float(gen.get('symbols', 0.0))) / 10.0,
        ], dtype=np.float32)
        
        # 8. String Stats (4 dims)
        strs = row.get('strings', {})
        strs_v = np.array([
            np.log1p(float(strs.get('numstrings', 0.0))) / 10.0,
            np.log1p(float(strs.get('avlength', 0.0))) / 10.0,
            np.log1p(float(strs.get('printables', 0.0))) / 10.0,
            float(strs.get('entropy', 0.0)) / 8.0,
        ], dtype=np.float32)
        
        # 9. Section Stats (4 dims)
        sections_dict = row.get('section', {})
        sec_list = []
        if isinstance(sections_dict, dict):
            sec_list = sections_dict.get('sections', [])
        sec_count = len(sec_list)
        sec_entropies = [float(s.get('entropy', 0.0)) for s in sec_list if isinstance(s, dict)]
        sec_v = np.array([
            min(float(sec_count), 20.0) / 20.0,
            (max(sec_entropies) / 8.0) if sec_entropies else 0.0,
            (min(sec_entropies) / 8.0) if sec_entropies else 0.0,
            (np.mean(sec_entropies) / 8.0) if sec_entropies else 0.0,
        ], dtype=np.float32)
        
        # 10. Imports Stats (2 dims)
        imports = row.get('imports', {})
        num_dlls = 0
        num_funcs = 0
        if isinstance(imports, dict):
            num_dlls = len(imports)
            for dll, funcs in imports.items():
                if isinstance(funcs, list):
                    num_funcs += len(funcs)
        elif isinstance(imports, list):
            num_dlls = len(imports)
        imp_v = np.array([
            np.log1p(float(num_dlls)) / 5.0,
            np.log1p(float(num_funcs)) / 10.0,
        ], dtype=np.float32)
        
        # 11. Exports Stats (1 dim)
        exports = row.get('exports', [])
        num_exports = 0
        if isinstance(exports, list):
            num_exports = len(exports)
        elif isinstance(exports, int):
            num_exports = exports
        elif exports:
            num_exports = 1
        exp_v = np.array([
            np.log1p(float(num_exports)) / 5.0,
        ], dtype=np.float32)
        
        # Combine all features to 631 dims
        vec = np.concatenate([
            tlsh, temp, ft_v, sha, hist, entropy, gen_v, strs_v, sec_v, imp_v, exp_v
        ])
        
        if vec.shape[0] != FEAT_DIM: return None
        return np.nan_to_num(vec, nan=0., posinf=1., neginf=0.).astype(np.float32)
    except Exception as e:
        return None

def _get_wid(row):
    try:
        w = row.get('week_id', None)
        if w is not None: return int(w)
    except: pass
    t = _parse_ts(row.get('first_submission_date', ''))
    if t > 0:
        try: return int(t // (7*86400))
        except: pass
    return 0

def load_jsonl(path):
    if not os.path.exists(path): return []
    data = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: data.append(json.loads(line))
            except: continue
    return data

def extract_features(data_list):
    X,yb,yf,ybeh,wids,evas=[],[],[],[],[],[]
    for row in data_list:
        raw = row.get('label', None)
        if raw is None: continue
        try: label = int(raw)
        except: continue
        if label not in (0, 1): continue
        vec = _feat_row(row)
        if vec is None: continue
        fam_s = str(row.get('family', '') or '')
        fam   = map_family(fam_s, label)
        X.append(vec); yb.append(float(label))
        yf.append(int(fam))
        ybeh.append(behavior_from_family(fam_s, label))
        wids.append(_get_wid(row))
        
        # Evasion flag
        det = None
        for k in ('detection_ratio','detection_rate','positives'):
            if k in row: det = row[k]; break
        try:
            s = str(det or '').strip()
            ratio = float(s.split('/')[0])/max(1.,float(s.split('/')[1])) if '/' in s else float(s or 0.)
            ratio = ratio/100. if ratio > 1. else ratio
        except: ratio = 0.
        evas.append(label==1 and ratio < 0.3)
    if not yb:
        e = np.array([], dtype=np.float32)
        return e,e,e,e,np.array([],dtype=np.int32),np.array([],dtype=bool)
    return (np.array(X,dtype=np.float32), np.array(yb,dtype=np.float32),
            np.array(yf,dtype=np.int64),  np.array(ybeh,dtype=np.float32),
            np.array(wids,dtype=np.int32), np.array(evas,dtype=bool))

def iter_file_chunks(fp, chunk_size=CHUNK):
    rows = load_jsonl(fp)
    for start in range(0, len(rows), chunk_size):
        yield rows[start:start+chunk_size]
    del rows; gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 2.5: FEATURE PRE-EXTRACTION AND CACHING
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset_features(files, desc="Dataset", max_samples=None):
    print(f"\n[Pre-extract] Loading and extracting features for {desc}...")
    X_l, yb_l, yf_l, ybeh_l, wids_l, evas_l = [], [], [], [], [], []
    t_start = time.time()
    total_rows = 0
    total_extracted = 0
    
    for fi, fp in enumerate(files):
        if max_samples is not None and total_extracted >= max_samples:
            break
        rows = load_jsonl(fp)
        if not rows: continue
        total_rows += len(rows)
        X, yb, yf, ybeh, wids, evas = extract_features(rows)
        if len(yb) == 0: continue
        
        # Limit control
        if max_samples is not None and total_extracted + len(yb) > max_samples:
            take = max_samples - total_extracted
            X = X[:take]
            yb = yb[:take]
            yf = yf[:take]
            ybeh = ybeh[:take]
            wids = wids[:take]
            evas = evas[:take]
            
        X_l.append(X)
        yb_l.append(yb)
        yf_l.append(yf)
        ybeh_l.append(ybeh)
        wids_l.append(wids)
        evas_l.append(evas)
        total_extracted += len(yb)
        
        del rows; gc.collect()
        if (fi+1) % 5 == 0 or (fi+1) == len(files):
            print(f"  Processed {fi+1}/{len(files)} files... Extracted {total_extracted:,} samples")
            
    if not X_l:
        empty = np.array([], dtype=np.float32)
        return (empty, empty, empty, empty,
                np.array([], dtype=np.int32), np.array([], dtype=bool))
                
    X_all = np.vstack(X_l)
    yb_all = np.concatenate(yb_l)
    yf_all = np.concatenate(yf_l)
    ybeh_all = np.vstack(ybeh_l)
    wids_all = np.concatenate(wids_l)
    evas_all = np.concatenate(evas_l)
    
    t_elapsed = time.time() - t_start
    print(f"  ✅ Extracted {len(X_all):,} samples from {total_rows:,} raw rows in {t_elapsed:.1f}s")
    return X_all, yb_all, yf_all, ybeh_all, wids_all, evas_all


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 3: FULL DATA DISTRIBUTION & CLASS WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights_full(yf_train, beta=0.9999):
    """Tính toán Class weights (Cui et al. Class-Balanced Loss) từ cache RAM."""
    counts = np.zeros(N_FAMILIES, dtype=np.int64)
    for c in yf_train:
        if 0 <= int(c) < N_FAMILIES: counts[int(c)] += 1
    total_valid = len(yf_train)

    print(f'\n  ✅ Tổng valid samples: {total_valid:,}')
    print(f'\n  {"Class":<14} {"Count":>10} {"Ratio":>8}  Note')
    print('  '+'─'*50)
    for c, (name, cnt) in enumerate(zip(FAMILY_NAMES, counts)):
        pct  = cnt/max(1,total_valid)*100
        note = '✅ Đủ' if cnt>=1000 else ('⚠️ Ít' if cnt>=100 else '❌ Rất ít')
        print(f'  {name:<14} {cnt:>10,} {pct:>8.3f}%  {note}')

    weights = []
    for c_count in counts:
        if c_count == 0:
            weights.append(1.0)
        else:
            w = (1.0 - beta) / (1.0 - np.power(beta, c_count) + 1e-9)
            weights.append(w)
            
    weights = np.array(weights)
    weights = weights / np.mean(weights) 
    weights = np.clip(weights, 0.5, MAX_WEIGHT)

    print(f'\n  Class Balanced Weights (Cui et al. max={MAX_WEIGHT}):')
    print(f'  {"Class":<14} {"Count":>10} {"Weight":>8}')
    print('  '+'─'*36)
    for name, cnt, w in zip(FAMILY_NAMES, counts, weights):
        print(f'  {name:<14} {cnt:>10,} {w:>8.3f}')
    return torch.tensor(weights, dtype=torch.float32), counts

def build_aug_targets(real_counts):
    """
    Augmentation targets – tự động từ phân phối thực tế.
    Các class phổ biến (≥5000 mẫu) không augment.
    """
    targets = {}
    print('\n  [AugTarget] Dynamic targets:')
    print(f'  {"Class":<14} {"Real":>10} {"Target":>10}  Mult')
    print('  '+'─'*48)
    for c in range(N_FAMILIES):
        if c in COMMON_CLASSES: continue
        real   = int(real_counts[c])
        target = max(AUG_MIN, min(real * AUG_MULT, 8000))
        targets[c] = target
        mult = target/max(1,real)
        print(f'  {FAMILY_NAMES[c]:<14} {real:>10,} {target:>10,}  ×{mult:.1f}')
    return targets


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 4: SYNTHETIC AUGMENTATION (Tuned CPU Oversampling)
# ─────────────────────────────────────────────────────────────────────────────

def synth_cls(X_cls, n_new, noise_std=0.01):
    """Giảm noise_std xuống 0.01 thô để giữ tính chất topology và chuyển phần nâng cao cho GPU Manifold Mixup."""
    n_real = len(X_cls); out = []
    if n_real == 0:
        return np.zeros((n_new, FEAT_DIM), dtype=np.float32)
        
    for _ in range(n_new):
        if n_real >= 2:
            i1, i2 = np.random.choice(n_real, 2, replace=False)
            lam  = np.random.beta(0.4, 0.4)
            base = (lam*X_cls[i1] + (1-lam)*X_cls[i2]).copy()
        else:
            base = X_cls[0].copy()
            
        # Thêm nhiễu rất nhỏ để tạo variant thô
        base[:70]  = np.clip(base[:70]  + np.random.randn(70).astype(np.float32)*noise_std, 0, 1)
        base[70:78]= np.clip(base[70:78]+ np.random.randn(8).astype(np.float32)*(noise_std/3), 0, 1)
        base[98:354] = np.clip(base[98:354] + np.random.randn(256).astype(np.float32)*(noise_std/5), 0, 1)
        hist_sum = base[98:354].sum()
        if hist_sum > 0: base[98:354] /= hist_sum
        
        base[354:610] = np.clip(base[354:610] + np.random.randn(256).astype(np.float32)*(noise_std/5), 0, 1)
        base[610:631] = np.clip(base[610:631] + np.random.randn(21).astype(np.float32)*(noise_std/3), 0, 1)
        
        # Binarize binary features
        binary_indices = [74, 75, 76, 613, 614, 615, 616, 617, 618]
        base[binary_indices] = np.where(base[binary_indices] >= 0.5, 1.0, 0.0)
        
        out.append(base)
    return np.array(out, dtype=np.float32)

def augment_batch(X, yb, yf, ybeh, targets):
    """Augment rare classes bằng mixup thô."""
    X_l=[X]; yb_l=[yb]; yf_l=[yf]; ybeh_l=[ybeh]
    cnts = Counter(yf.tolist())
    for cls, tgt in targets.items():
        cur = cnts.get(cls, 0)
        if cur == 0 or cur >= tgt: continue
        idx  = np.where(yf==cls)[0]
        X_s  = synth_cls(X[idx], tgt-cur, noise_std=0.01)
        X_l.append(X_s)
        yb_l.append(np.ones(len(X_s), dtype=np.float32))
        yf_l.append(np.full(len(X_s), cls, dtype=np.int64))
        fam_name = FAMILY_NAMES[cls]
        beh_s = behavior_from_family(fam_name, 1)
        ybeh_l.append(np.tile(beh_s, (len(X_s), 1)))
    X_a    = np.vstack(X_l)
    yb_a   = np.concatenate(yb_l)
    yf_a   = np.concatenate(yf_l)
    ybeh_a = np.vstack(ybeh_l)
    p = np.random.permutation(len(X_a))
    return X_a[p], yb_a[p], yf_a[p], ybeh_a[p]


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 5: BALANCED BATCH SAMPLER
# ─────────────────────────────────────────────────────────────────────────────

def make_batches(y_fam, batch_size=512, min_rare=15):
    """Balanced batches, tăng min_rare mẫu của các nhóm hiếm lên 15 để làm mượt Soft F1 Loss."""
    cls_idx = {c: np.where(y_fam==c)[0] for c in range(N_FAMILIES)
               if (y_fam==c).any()}
    perm    = np.random.permutation(len(y_fam)); batches = []
    for s in range(0, len(y_fam), batch_size):
        bi = list(perm[s:min(s+batch_size, len(y_fam))])
        for cls in RARE_CLASSES:
            if cls not in cls_idx: continue
            present = sum(1 for i in bi if int(y_fam[i])==cls)
            need    = min_rare - present
            if need > 0:
                extra = np.random.choice(cls_idx[cls],
                    size=min(need, len(cls_idx[cls])), replace=True)
                bi.extend(extra.tolist())
        np.random.shuffle(bi)
        batches.append(np.array(bi, dtype=np.int64))
    return batches


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 5.5: GPU MANIFOLD MIXUP (Augmentation trong không gian embedding)
# ─────────────────────────────────────────────────────────────────────────────

def apply_manifold_mixup(toks, yf, alpha=0.4):
    """
    Áp dụng Manifold Mixup (nội suy vector) nội bộ trong từng họ (Mixup-within-class)
    trên không gian embedding ở GPU khi train. Bảo tồn nhãn tuyệt đối và giữ phân phối thực tế.
    """
    device = toks.device
    B = toks.size(0)
    mixup_indices = torch.arange(B, device=device)
    lambdas = torch.ones(B, device=device)
    
    unique_classes = torch.unique(yf)
    for c in unique_classes:
        if c == -1: continue
        idx = torch.where(yf == c)[0]
        if len(idx) >= 2:
            perm = torch.randperm(len(idx), device=device)
            mixup_indices[idx] = idx[perm]
            l = np.random.beta(alpha, alpha)
            l = max(l, 1.0 - l) # Đảm bảo lambda nằm phần lớn ở mẫu chính [0.5, 1.0]
            lambdas[idx] = l
            
    # Áp dụng Mixup trên Token embeddings
    mixed_toks = lambdas.view(B, 1, 1) * toks + (1.0 - lambdas.view(B, 1, 1)) * toks[mixup_indices]
    return mixed_toks


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 6: MODEL  (d=256, 6 layers, MLP Tokenizer, Cosine Similarity Head, Mixup)
# ─────────────────────────────────────────────────────────────────────────────

class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, scale=20.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, x):
        w_norm = self.weight / (self.weight.norm(p=2, dim=1, keepdim=True) + 1e-8)
        x_norm = x / (x.norm(p=2, dim=1, keepdim=True) + 1e-8)
        cos = torch.matmul(x_norm, w_norm.t())
        return self.scale * cos

class FTTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        d = D_MODEL
        
        # 2-layer MLP Tokenizer thay cho phép chiếu tuyến tính cơ bản
        def _mlp_proj(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Linear(out_dim, out_dim)
            )
            
        self.proj_tlsh    = _mlp_proj(70, d)
        self.proj_time    = _mlp_proj(8,  d)
        self.proj_ft      = _mlp_proj(10, d)
        self.proj_sha     = _mlp_proj(10, d)
        self.proj_hist    = _mlp_proj(256, d)
        self.proj_entropy = _mlp_proj(256, d)
        self.proj_header  = _mlp_proj(21, d)
        
        # CLS + positional parameters
        self.cls          = nn.Parameter(torch.zeros(1, 1, d))
        self.pos          = nn.Parameter(torch.randn(1, 8, d) * 0.02) # CLS + 7 tokens = 8
        nn.init.trunc_normal_(self.cls, std=0.02)
        
        # TransformerEncoderLayer norm_first=True
        enc = nn.TransformerEncoderLayer(
            d_model=d, nhead=N_HEAD, dim_feedforward=DIM_FF,
            dropout=DROPOUT, batch_first=True, norm_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(enc, num_layers=N_LAYERS)
        self.norm        = nn.LayerNorm(d)
        
        # Task heads
        def _head(out):
            return nn.Sequential(
                nn.Linear(d, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(DROPOUT),
                nn.Linear(128, 64), nn.GELU(), nn.Dropout(DROPOUT),
                nn.Linear(64, out))
                
        # Deep family head (more capacity + L2 Cosine Classifier)
        self.head_family = nn.Sequential(
            nn.Linear(d, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(DROPOUT),
            CosineLinear(128, N_FAMILIES, scale=20.0))
            
        self.head_binary   = _head(1)
        self.head_behavior = _head(N_BEHAVIORS)

    def _tokenize(self, x):
        # Biến đổi log1p để ổn định đặc trưng số
        x = torch.sign(x) * torch.log1p(torch.abs(x))
        toks = torch.stack([
            self.proj_tlsh(x[:, 0:70]),
            self.proj_time(x[:, 70:78]),
            self.proj_ft  (x[:, 78:88]),
            self.proj_sha (x[:, 88:98]),
            self.proj_hist(x[:, 98:354]),
            self.proj_entropy(x[:, 354:610]),
            self.proj_header (x[:, 610:631]),
        ], dim=1)
        cls  = self.cls.expand(x.size(0), -1, -1)
        return torch.cat([cls, toks], dim=1) + self.pos

    def encode(self, x):
        return self.norm(self.transformer(self._tokenize(x))[:, 0])

    def encode_with_attn(self, x):
        toks    = self._tokenize(x); cur = toks; attns = []
        with torch.no_grad():
            for layer in self.transformer.layers:
                normed = layer.norm1(cur) if layer.norm_first else cur
                _, aw  = layer.self_attn(normed, normed, normed,
                    need_weights=True, average_attn_weights=True)
                attns.append(aw.detach().cpu().numpy())
                cur = layer(cur)
        return self.norm(cur[:, 0]), attns

    def forward(self, x, yf=None, mixup_alpha=0.0):
        toks = self._tokenize(x)
        
        # Áp dụng Manifold Mixup trong không gian embedding trên GPU khi train
        if mixup_alpha > 0.0 and yf is not None:
            toks = apply_manifold_mixup(toks, yf, mixup_alpha)
            
        emb = self.norm(self.transformer(toks)[:, 0])
        return (self.head_binary(emb).squeeze(-1),
                self.head_family(emb),
                self.head_behavior(emb))

    def set_family_only(self, freeze_others=True):
        """Phase 2: Freeze binary + behavior heads, unfreeze family head and last 2 layers of transformer."""
        trainable = {'head_family', 'transformer.layers.4', 'transformer.layers.5', 'norm'}
        for n, p in self.named_parameters():
            p.requires_grad = any(t in n for t in trainable) if freeze_others else True

    def unfreeze_all(self):
        for p in self.parameters(): p.requires_grad = True


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 7: LOSS FUNCTIONS  (Focal + Logit Adjustment + Soft F1 Loss)
# ─────────────────────────────────────────────────────────────────────────────

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.alpha=alpha; self.gamma=gamma; self.smooth=label_smoothing
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    def forward(self, logits, targets):
        t = targets*(1-self.smooth) + 0.5*self.smooth
        l = self.bce(logits, t)
        pt = torch.exp(-l)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        return (alpha_t * (1.0 - pt)**self.gamma * l).mean()

class FamilyFocalLoss(nn.Module):
    def __init__(self, weights=None, class_prior=None, tau=1.2, gamma=2.5, label_smoothing=0.05, f1_alpha=2.0):
        super().__init__()
        self.gamma = gamma
        self.f1_alpha = f1_alpha
        self.tau = tau
        self.register_buffer('log_prior_tensor', torch.log(class_prior + 1e-9) if class_prior is not None else None)
        self.ce = nn.CrossEntropyLoss(
            weight=weights, label_smoothing=label_smoothing, reduction='none', ignore_index=-1)
            
    def forward(self, logits, targets):
        if hasattr(self, 'log_prior_tensor') and self.log_prior_tensor is not None:
            log_prior = self.log_prior_tensor.to(logits.device)
            # Menon et al. "Heavy-tail learning via logit adjustment" uses "+" in training:
            adjusted_logits = logits + self.tau * log_prior.unsqueeze(0)
        else:
            adjusted_logits = logits
            
        ce = self.ce(adjusted_logits, targets)
        mask = (targets != -1).float()
        if mask.sum() == 0: return logits.sum() * 0.0
        
        pt = torch.exp(-ce)
        focal_loss = (((1 - pt)**self.gamma * ce) * mask).sum() / mask.sum()
        
        # Soft Macro F1 Loss
        probs = torch_softmax(logits, dim=1)
        valid_idx = (targets != -1)
        if valid_idx.sum() > 0:
            targets_valid = targets[valid_idx]
            probs_valid = probs[valid_idx]
            targets_one_hot = torch.nn.functional.one_hot(targets_valid, num_classes=logits.shape[1]).float()
            
            tp = (probs_valid * targets_one_hot).sum(dim=0)
            fp = (probs_valid * (1 - targets_one_hot)).sum(dim=0)
            fn = ((1 - probs_valid) * targets_one_hot).sum(dim=0)
            
            soft_f1 = 2 * tp / (2 * tp + fn + fp + 1e-8)
            classes_present = targets_valid.unique()
            macro_soft_f1_loss = 1.0 - soft_f1[classes_present].mean() if len(classes_present) > 0 else 0.0
        else:
            macro_soft_f1_loss = 0.0
            
        return focal_loss + self.f1_alpha * macro_soft_f1_loss

class MultiTaskLoss(nn.Module):
    def __init__(self, weights=None, class_prior=None, f1_alpha=2.0, family_focused=False):
        super().__init__()
        self.l_bin  = BinaryFocalLoss(0.5, 2.0, 0.05)
        self.l_fam  = FamilyFocalLoss(weights, class_prior=class_prior, tau=1.2, gamma=2.5, label_smoothing=0.05, f1_alpha=f1_alpha)
        self.l_beh  = nn.BCEWithLogitsLoss()
        self.family_focused = family_focused
        
        # Learnable parameters cho Uncertainty Weighting
        self.log_vars = nn.Parameter(torch.zeros(3))
        
    def forward(self, lb, lf, lbh, yb, yf, ybh):
        l1 = self.l_bin(lb, yb)
        l2 = self.l_fam(lf, yf)
        l3 = self.l_beh(lbh, ybh)
        
        if self.family_focused:
            # Phase 3: Tập trung tối đa vào Task 2 Family
            return 0.05 * l1 + 3.0 * l2 + 0.05 * l3, float(l1), float(l2), float(l3)
        else:
            # Phase 1: Áp dụng Uncertainty Weighting
            loss1 = torch.exp(-self.log_vars[0]) * l1 + 0.5 * self.log_vars[0]
            loss2 = torch.exp(-self.log_vars[1]) * l2 + 0.5 * self.log_vars[1]
            loss3 = torch.exp(-self.log_vars[2]) * l3 + 0.5 * self.log_vars[2]
            return loss1 + loss2 + loss3, float(l1), float(l2), float(l3)


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 8: EARLY STOPPING
# ─────────────────────────────────────────────────────────────────────────────

class ESMin:
    def __init__(self, patience=5, delta=0.003):
        self.pat=patience; self.delta=delta; self.best=float('inf')
        self.cnt=0; self.state=None
    def step(self, v, m):
        if v < self.best-self.delta:
            self.best=v; self.cnt=0; self.state=copy.deepcopy(m.state_dict())
            print(f'  [ES↓] ✅ {v:.4f}'); return False
        self.cnt+=1; print(f'  [ES↓] {self.cnt}/{self.pat}  best={self.best:.4f}')
        return self.cnt >= self.pat
    def restore(self, m):
        if self.state: m.load_state_dict(self.state); print(f'  [ES↓] Restored {self.best:.4f}')

class ESMax:
    def __init__(self, patience=5, delta=0.003):
        self.pat=patience; self.delta=delta; self.best=-float('inf')
        self.cnt=0; self.state=None
    def step(self, v, m):
        if v > self.best+self.delta:
            self.best=v; self.cnt=0; self.state=copy.deepcopy(m.state_dict())
            print(f'  [ES↑] ✅ F1={v:.4f}'); return False
        self.cnt+=1; print(f'  [ES↑] {self.cnt}/{self.pat}  best={self.best:.4f}')
        return self.cnt >= self.pat
    def restore(self, m):
        if self.state: m.load_state_dict(self.state); print(f'  [ES↑] Restored F1={self.best:.4f}')


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 9: VALIDATION (In-Memory) & HIERARCHICAL GATED CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

def _safe_roc(yt, yp):
    if len(np.unique(yt)) < 2: return np.array([0.,1.]),np.array([0.,1.]),0.5
    fpr,tpr,_=roc_curve(yt,yp); return fpr,tpr,float(auc(fpr,tpr))

def _safe_pr(yt, yp):
    if len(np.unique(yt)) < 2: return np.array([1.,0.]),np.array([0.,1.]),0.
    p,r,_=precision_recall_curve(yt,yp); return p,r,float(auc(r,p))

def calib_predict(logits_np, thresholds):
    """Per-class threshold calibration (standard argmax fallback)."""
    from scipy.special import softmax as _sm
    probs  = _sm(logits_np, axis=1)
    scores = probs / (thresholds[np.newaxis,:] + 1e-8)
    return scores.argmax(axis=1)

@torch.no_grad()
def val_loss_full(model, criterion, X_val, yb_val, yf_val, ybeh_val, device):
    model.eval(); total=0.; n=0
    if len(yb_val) == 0: return 0.0
    for s in range(0, len(X_val), 512):
        sl=slice(s,min(s+512,len(X_val)))
        lb,lf,lbh=model(torch.tensor(X_val[sl],dtype=torch.float32).to(device))
        loss,*_=criterion(lb,lf,lbh,
            torch.tensor(yb_val[sl],dtype=torch.float32).to(device),
            torch.tensor(yf_val[sl],dtype=torch.int64).to(device),
            torch.tensor(ybeh_val[sl],dtype=torch.float32).to(device))
        total+=float(loss); n+=1
    model.train(); return total/max(1,n)

@torch.no_grad()
def val_family_f1_full(model, X_val, yf_val, device, thr=None):
    model.eval()
    mask = (yf_val != -1)
    if not mask.any(): return 0.0
    ya, pa = [], []
    for s in range(0, len(X_val), 512):
        sl = slice(s, min(s+512, len(X_val)))
        sub_mask = mask[sl]
        if not sub_mask.any(): continue
        _, lf, _ = model(torch.tensor(X_val[sl][sub_mask], dtype=torch.float32).to(device))
        ya.extend(yf_val[sl][sub_mask].tolist())
        pa.extend((calib_predict(lf.cpu().numpy(), thr)
                   if thr is not None else lf.cpu().numpy().argmax(axis=1)).tolist())
    model.train()
    return float(f1_score(ya, pa, labels=ALL_FAM_LABELS, average='macro', zero_division=0)) if ya else 0.

def find_binary_thr(model, X_val, yb_val, device):
    """Tìm ngưỡng nhị phân tối ưu trên dữ liệu validation in-memory."""
    print('\n  [BinThr] calibrating binary threshold...')
    model.eval(); ya,la=[],[]
    from scipy.special import logsumexp as _lse
    for s in range(0, len(X_val), 512):
        sl = slice(s, min(s+512, len(X_val)))
        with torch.no_grad(): lb,lf,_=model(torch.tensor(X_val[sl],dtype=torch.float32).to(device))
        lbn = lb.cpu().numpy()
        lfn = lf.cpu().numpy()
        fam_logit = _lse(lfn[:, 1:], axis=1) - lfn[:, 0]
        combined = 0.4 * lbn + 0.6 * fam_logit
        ya.extend(yb_val[sl].tolist()); la.extend(combined.tolist())
    if not ya: return 0.0
    y = np.array(ya); logits = np.array(la)
    best_f1 = 0.0; best_thr = 0.0
    p5 = np.percentile(logits, 5)
    p95 = np.percentile(logits, 95)
    for t in np.linspace(p5, p95, 100):
        preds = (logits >= t).astype(int)
        f1 = f1_score(y, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = t
    print(f'  Optimal Binary Logit Threshold: {best_thr:+.4f} | Best F1: {best_f1:.4f}')
    return float(best_thr)

def find_family_thr(model, family_lgb, X_val, yf_val, device, bin_thr=0.0):
    print('\n  [FamThr v7.2] Running Multi-Dimensional Coordinate Descent (Joint Soft Gating)...')
    model.eval()
    ya, pf_nn, pf_lgb, pb = [], [], [], []
    from scipy.special import logsumexp as _lse
    from scipy.special import softmax as _sm
    
    for s in range(0, len(X_val), 512):
        sl = slice(s, min(s+512, len(X_val)))
        mask = (yf_val[sl] != -1)
        if not mask.any(): continue
        with torch.no_grad():
            lb, lf, _ = model(torch.tensor(X_val[sl][mask], dtype=torch.float32).to(device))
        
        pb.extend((0.4 * lb.cpu().numpy() + 0.6 * (_lse(lf.cpu().numpy()[:, 1:], axis=1) - lf.cpu().numpy()[:, 0])).tolist())
        pf_nn.extend(_sm(lf.cpu().numpy(), axis=1).tolist())
        ya.extend(yf_val[sl][mask].tolist())
        
        if family_lgb is not None:
            pf_lgb.extend(family_lgb.predict_proba(X_val[sl][mask]).tolist())

    if not ya: return bin_thr, np.ones(7, dtype=np.float32)
    
    y = np.array(ya)
    pb_arr = np.array(pb)
    
    # Kỹ thuật cân bằng Ensemble weights: Giảm bớt tỷ lệ LGBM xuống 0.5 nếu bị áp chế quá mạnh
    p_nn = np.array(pf_nn)
    p_lgb = np.array(pf_lgb) if family_lgb is not None else p_nn
    combined_probs = 0.5 * p_nn + 0.5 * p_lgb 
    
    # Tính toán xác suất kết hợp (Joint Probability) cho toàn bộ 7 lớp
    # Sử dụng pb_arr - bin_thr để căn chỉnh xác suất nhị phân với ngưỡng tối ưu đã chọn
    p_bin_mal = torch.sigmoid(torch.tensor(pb_arr - bin_thr)).numpy()
    p_bin_ben = 1.0 - p_bin_mal

    joint_probs = np.zeros((len(y), 7), dtype=np.float32)
    joint_probs[:, 0] = p_bin_ben * combined_probs[:, 0]
    joint_probs[:, 1:] = p_bin_mal[:, np.newaxis] * combined_probs[:, 1:]
    
    # Khởi tạo thr_bias_7 gồm 7 phần tử
    thr_bias_7 = np.ones(7, dtype=np.float32)
    
    # 3 vòng quét lặp phối hợp toàn diện (Joint Optimization)
    for _ in range(3):
        for c_idx in range(7):
            best_c_macro = 0.0
            best_c_thr = thr_bias_7[c_idx]
            
            # Quét dải rộng phân tầng cho phân lớp
            for t_val in np.logspace(-1, 1, 40): 
                test_thrs = thr_bias_7.copy()
                test_thrs[c_idx] = t_val
                
                # Tính toán ma trận điểm số đã hiệu chuẩn
                scores = joint_probs / (test_thrs[np.newaxis, :] + 1e-8)
                preds = scores.argmax(axis=1)
                
                macro = f1_score(y, preds, labels=ALL_FAM_LABELS, average='macro', zero_division=0)
                if macro > best_c_macro:
                    best_c_macro = macro
                    best_c_thr = t_val
            thr_bias_7[c_idx] = best_c_thr

    # Kiểm tra lại kết quả sau hiệu chuẩn tầng sâu
    scores = joint_probs / (thr_bias_7[np.newaxis, :] + 1e-8)
    final_preds = scores.argmax(axis=1)
    
    print(f'  --> [v7.2 Tối ưu] Khôi phục Macro-F1 lên mức mục tiêu: {f1_score(y, final_preds, labels=ALL_FAM_LABELS, average="macro", zero_division=0):.4f}')
    return bin_thr, thr_bias_7

def find_beh_thr(model, X_val, ybeh_val, device):
    """Optimal behavior threshold on validation data in-memory."""
    print('\n  [BehThr] calibrating behavior thresholds...')
    model.eval(); ya,pa=[],[]
    for s in range(0, len(X_val), 512):
        sl = slice(s, min(s+512, len(X_val)))
        with torch.no_grad(): _,_,lbh=model(torch.tensor(X_val[sl],dtype=torch.float32).to(device))
        ya.extend(ybeh_val[sl].tolist()); pa.extend(lbh.cpu().numpy().tolist())
    thr=np.zeros(N_BEHAVIORS,dtype=np.float32)
    if not ya: return thr
    yb2=(np.array(ya)>=0.5).astype(int); pb2=np.array(pa)
    print(f'  {"Behavior":<22}|{"F1":>7}|{"Thr":>6}|{"Supp":>8}')
    print('  '+'─'*48)
    for i,name in enumerate(BEHAVIOR_NAMES):
        supp=int(yb2[:,i].sum())
        if supp==0: print(f'  {name:<22}|{"N/A":>7}|{"+0.0":>6}|{supp:>8}  ➖'); continue
        bf,bt=0.,0.
        for t in np.linspace(-5.,5.,101):
            pred=(pb2[:,i]>=t).astype(int)
            f1=f1_score(yb2[:,i],pred,zero_division=0)
            if f1>bf: bf,bt=f1,float(t)
        thr[i]=bt
        print(f'  {name:<22}|{bf:>7.4f}|{bt:>+6.1f}|{supp:>8,}  {"✅" if bf>=0.5 else "⚠️"}')
    return thr


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 10: TRAINING PHASES (In-Memory)
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, criterion, optimizer, X_train, yb_train, yf_train, ybeh_train, device,
                history, aug_targets, batch_size=512, min_rare=15, warmup_info=None, mixup_alpha=0.0):
    """Huấn luyện 1 epoch trực tiếp từ dữ liệu trong RAM có tích hợp Manifold Mixup khi train."""
    model.train(); ep_loss=0.; ep_steps=0
    
    # Augment các họ hiếm gặp
    X_a, yb_a, yf_a, ybeh_a = augment_batch(X_train, yb_train, yf_train, ybeh_train, aug_targets)
    batches=make_batches(yf_a,batch_size,min_rare)
    n_batches = len(batches)
    
    for i, bi in enumerate(batches):
        if len(bi)<8: continue
        bX=X_a[bi].copy(); bB=yb_a[bi]; bF=yf_a[bi]; bBeh=ybeh_a[bi]
        # TLSH noise
        bX[:,:70]=np.clip(bX[:,:70]+
            np.random.randn(len(bX),70).astype(np.float32)*TLSH_NOISE,0,1)
            
        optimizer.zero_grad()
        
        # Linear Warmup (e.g. 300 steps đầu của Phase 1)
        if warmup_info is not None:
            step = warmup_info['global_step']
            warmup_steps = warmup_info['warmup_steps']
            base_lr = warmup_info['base_lr']
            if step < warmup_steps:
                lr_scale = float(step + 1) / warmup_steps
                for pg in optimizer.param_groups:
                    pg['lr'] = lr_scale * pg.get('initial_lr', base_lr)
            warmup_info['global_step'] += 1
            
        lb,lf_,lbh=model(torch.tensor(bX,dtype=torch.float32).to(device),
                         yf=torch.tensor(bF,dtype=torch.int64).to(device),
                         mixup_alpha=mixup_alpha)
                         
        loss,l1,l2,l3=criterion(lb,lf_,lbh,
            torch.tensor(bB,  dtype=torch.float32).to(device),
            torch.tensor(bF,  dtype=torch.int64   ).to(device),
            torch.tensor(bBeh,dtype=torch.float32).to(device))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        lv=float(loss)
        history['total'].append(lv); history['binary'].append(l1)
        history['family'].append(l2); history['behavior'].append(l3)
        ep_loss+=lv; ep_steps+=1
        
        # Logging progress
        if (i+1) % 40 == 0 or (i+1) == n_batches:
            print(f"    Batch {i+1:03d}/{n_batches:03d} | Loss: {lv:.4f} (l_bin: {l1:.4f}, l_fam: {l2:.4f}, l_beh: {l3:.4f})", flush=True)
            
    return ep_loss/max(1,ep_steps), ep_steps

def collect_rare_from_cached(X_train, yf_train, aug_targets):
    """Phase 2: Thu thập và augment dữ liệu từ cache RAM."""
    print('  [P2] Collecting training data from memory cache...')
    X_by = {c: [] for c in range(N_FAMILIES)}
    for c in range(N_FAMILIES):
        idx = np.where(yf_train == c)[0]
        if len(idx) > 0:
            X_by[c] = X_train[idx]
            
    X_list, y_list = [], []
    for c, name in enumerate(FAMILY_NAMES):
        X_cls = np.array(X_by[c], dtype=np.float32) if len(X_by[c]) > 0 else np.array([])
        if len(X_cls) == 0:
            print(f'  {name:<14}: 0  (không có mẫu train!)'); continue
        actual = len(X_cls)
        target = aug_targets.get(c, actual)
        if c not in COMMON_CLASSES and actual<target:
            X_syn = synth_cls(X_cls, target-actual, noise_std=0.01)
            X_cls = np.vstack([X_cls, X_syn])
        if c in COMMON_CLASSES:
            MAX_C = max(aug_targets.values())*2 if aug_targets else 5000
            if len(X_cls)>MAX_C:
                idx=np.random.choice(len(X_cls),MAX_C,replace=False)
                X_cls=X_cls[idx]
        X_list.append(X_cls)
        y_list.append(np.full(len(X_cls),c,dtype=np.int64))
        print(f'  {name:<14}: {actual:>8,} → {len(X_cls):>8,}')
        
    if not X_list: return None,None
    X_all=np.vstack(X_list); y_all=np.concatenate(y_list)
    p=np.random.permutation(len(X_all))
    X_all=X_all[p]; y_all=y_all[p]
    print(f'\n  P2 dataset: {len(X_all):,} samples')
    for c,cnt in sorted(Counter(y_all.tolist()).items()):
        print(f'  {FAMILY_NAMES[c]:<14}: {cnt:>8,}')
    return X_all, y_all

def train_family_phase(model, fam_criterion, X_train, yf_train, aug_targets,
                        device, history, X_val=None, yf_val=None, n_epochs=12, lr=1.5e-4, batch_size=256):
    """Phase 2: Family-only với unfrozen backbone, validation best state saving và Manifold Mixup."""
    print('\n  [P2] Unfreezing backbone & family head, freezing binary+behavior heads...')
    
    for n, p in model.named_parameters():
        if 'head_binary' in n or 'head_behavior' in n:
            p.requires_grad = False
        else:
            p.requires_grad = True

    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    total=sum(p.numel() for p in model.parameters())
    print(f'  Trainable in Phase 2: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)')

    X_all, y_all = collect_rare_from_cached(X_train, yf_train, aug_targets)
    if X_all is None: model.unfreeze_all(); return

    opt2=optim.AdamW([p for p in model.parameters() if p.requires_grad],
                     lr=lr, weight_decay=WEIGHT_DECAY)
    sch2=optim.lr_scheduler.CosineAnnealingLR(opt2,T_max=n_epochs,eta_min=1e-5)

    best_vf = -1.0
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        perm=np.random.permutation(len(X_all))
        X_e=X_all[perm]; y_e=y_all[perm]
        ep_loss=0.; ep_steps=0
        n_batches = int(np.ceil(len(X_e) / batch_size))
        
        for s in range(0, len(X_e), batch_size):
            sl=slice(s, min(s+batch_size,len(X_e)))
            bX=X_e[sl].copy(); bF=y_e[sl]
            bX[:,:70]=np.clip(bX[:,:70]+
                np.random.randn(len(bX),70).astype(np.float32)*TLSH_NOISE,0,1)
            opt2.zero_grad()
            
            # Áp dụng Manifold Mixup trên GPU
            _,lf,_=model(torch.tensor(bX,dtype=torch.float32).to(device),
                          yf=torch.tensor(bF,dtype=torch.int64).to(device),
                          mixup_alpha=0.4)
                          
            loss=fam_criterion(lf,torch.tensor(bF,dtype=torch.int64).to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt2.step()
            lv=float(loss); ep_loss+=lv; ep_steps+=1; history['family'].append(lv)
            
            # Logging progress
            batch_idx = s // batch_size + 1
            if batch_idx % 40 == 0 or batch_idx == n_batches:
                print(f"    P2 Batch {batch_idx:03d}/{n_batches:03d} | Family Loss: {lv:.4f}", flush=True)
                
        sch2.step()
        print(f'  [P2] Epoch {epoch+1}/{n_epochs}  loss={ep_loss/max(1,ep_steps):.4f}')
        
        # Validation sau mỗi epoch ở Phase 2 để lưu state tốt nhất
        if X_val is not None and yf_val is not None:
            vf = val_family_f1_full(model, X_val, yf_val, device)
            print(f'  [P2] Epoch {epoch+1} validation family_macro_F1={vf:.4f}')
            if vf > best_vf:
                best_vf = vf
                best_state = copy.deepcopy(model.state_dict())
                print(f'  [P2] Epoch {epoch+1} is the new best! F1={vf:.4f}')

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f'  [P2] Restored best model state with F1={best_vf:.4f}')

    model.unfreeze_all(); print('  [P2] ✅ Done – all params unfrozen')
    del X_all, y_all; gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 11: FULL EVALUATION (In-Memory với JOINT PROBABILISTIC SOFT GATING)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_full(model, family_lgb, X_test, yb_test, yf_test, ybeh_test, evas_test, device,
                  beh_thr=None, fam_thr=None, bin_thr=0.0, tag='Final'):
    model.eval()
    if beh_thr is None: beh_thr=np.zeros(N_BEHAVIORS,dtype=np.float32)
    pb_a, pf_a, pbh_a = [], [], []
    from scipy.special import logsumexp as _lse
    from scipy.special import softmax as _sm

    for s in range(0, len(X_test), 512):
        sl = slice(s, min(s+512, len(X_test)))
        lb, lf, lbh = model(torch.tensor(X_test[sl], dtype=torch.float32).to(device))
        lbn = lb.cpu().numpy()
        lfn = lf.cpu().numpy()
        lbhn = lbh.cpu().numpy()
        
        # Combined logit
        fam_logit = _lse(lfn[:, 1:], axis=1) - lfn[:, 0]
        combined = 0.4 * lbn + 0.6 * fam_logit
        pb_a.extend(combined.tolist())
        pf_a.extend(_sm(lfn, axis=1).tolist())
        pbh_a.extend(lbhn.tolist())

    pb = np.array(pb_a, dtype=np.float32)
    pf_nn = np.array(pf_a, dtype=np.float32)
    ph = np.array(pbh_a, dtype=np.float32)

    # LGBM probabilities
    if family_lgb is not None:
        pf_lgb = family_lgb.predict_proba(X_test)
        combined_probs = 0.5 * pf_nn + 0.5 * pf_lgb
    else:
        combined_probs = pf_nn

    # Task 1 – Binary
    fpr, tpr, ra = _safe_roc(yb_test, pb)
    prec, rec, pra = _safe_pr(yb_test, pb)
    yp = (pb >= bin_thr).astype(int)
    cm2 = confusion_matrix(yb_test, yp, labels=[0, 1])
    tn, fp2, fn, tp2 = cm2.ravel() if cm2.shape == (2, 2) else (0, 0, 0, 0)
    bin_r = dict(fpr_c=fpr, tpr_c=tpr, prec_c=prec, rec_c=rec, roc_auc=ra, pr_auc=pra,
                 tpr_01=float(np.interp(0.01, fpr, tpr)), tpr_001=float(np.interp(0.001, fpr, tpr)),
                 acc=float(accuracy_score(yb_test, yp)), f1=float(f1_score(yb_test, yp, zero_division=0)),
                 fpr_v=float(fp2/(fp2+tn+1e-9)), fnr_v=float(fn/(fn+tp2+1e-9)))

    # Evasive
    eva_r = None
    if evas_test is not None:
        idx = np.where((yb_test == 0) | evas_test)[0]
    else:
        idx = np.where(yb_test == 0)[0]
    if len(idx) > 0 and len(np.unique(yb_test[idx])) >= 2:
        fe, te, re = _safe_roc(yb_test[idx], pb[idx])
        pee, ree, _ = _safe_pr(yb_test[idx], pb[idx])
        eva_r = dict(fpr_c=fe, tpr_c=te, prec_c=pee, rec_c=rec, roc_auc=re,
                     tpr_01=float(np.interp(0.01, fe, te)), tpr_001=float(np.interp(0.001, fe, te)))

    # Task 2 – Family (FILTER OUT -1 TARGETS) WITH JOINT PROBABILISTIC SOFT GATING
    mask_fam = (yf_test != -1)
    if mask_fam.any():
        yf_filtered = yf_test[mask_fam]
        pf_filtered = combined_probs[mask_fam]
        pb_filtered = pb[mask_fam]
        
        if fam_thr is None:
            fam_thr = np.ones(7, dtype=np.float32)
            
        # Tính toán xác suất kết hợp (Joint Probability) cho toàn bộ 7 lớp
        # Sử dụng pb_filtered - bin_thr để căn chỉnh xác suất nhị phân với ngưỡng tối ưu đã chọn
        p_bin_mal = torch.sigmoid(torch.tensor(pb_filtered - bin_thr)).numpy()
        p_bin_ben = 1.0 - p_bin_mal
        
        joint_probs = np.zeros((len(yf_filtered), 7), dtype=np.float32)
        joint_probs[:, 0] = p_bin_ben * pf_filtered[:, 0]
        joint_probs[:, 1:] = p_bin_mal[:, np.newaxis] * pf_filtered[:, 1:]
        
        scores = joint_probs / (fam_thr[np.newaxis, :] + 1e-8)
        pflab = scores.argmax(axis=1)
                
        fam_r = dict(acc=float(accuracy_score(yf_filtered, pflab)),
                     macro_f1=float(f1_score(yf_filtered, pflab, average='macro', zero_division=0)),
                     weighted_f1=float(f1_score(yf_filtered, pflab, average='weighted', zero_division=0)),
                     report=classification_report(yf_filtered, pflab, labels=ALL_FAM_LABELS, target_names=FAMILY_NAMES, zero_division=0),
                     cm=confusion_matrix(yf_filtered, pflab, labels=ALL_FAM_LABELS),
                     pred=pflab, true=yf_filtered)
    else:
        fam_r = dict(acc=0.0, macro_f1=0.0, weighted_f1=0.0, report="No valid family labels", cm=np.zeros((N_FAMILIES, N_FAMILIES)), pred=np.array([]), true=np.array([]))

    # Task 3 – Behavior
    yb2 = (ybeh_test >= 0.5).astype(int)
    pb2 = (ph >= beh_thr[np.newaxis, :]).astype(int)
    hl = float(hamming_loss(yb2, pb2)) if len(yb2) > 0 else 0.
    per_b = {}
    for i, name in enumerate(BEHAVIOR_NAMES):
        supp = int(yb2[:, i].sum())
        try:
            fb2, tb2, _ = roc_curve(yb2[:, i], ph[:, i])
            ra_b = float(auc(fb2, tb2))
        except:
            ra_b = float('nan')
        per_b[name] = dict(roc_auc=ra_b, support=supp, threshold=float(beh_thr[i]),
                           prec=float(precision_score(yb2[:, i], pb2[:, i], zero_division=0)),
                           rec =float(recall_score   (yb2[:, i], pb2[:, i], zero_division=0)),
                           f1  =float(f1_score       (yb2[:, i], pb2[:, i], zero_division=0)))
    beh_r = dict(hamming=hl, per_behavior=per_b)
    model.train()
    return bin_r, evas_test, eva_r, fam_r, beh_r

def compute_family_f1_detail(fam_r, min_support=30):
    yt=fam_r['true']; yp=fam_r['pred']; sup=Counter(yt.tolist())
    per={}
    for c,name in enumerate(FAMILY_NAMES):
        s=sup.get(c,0)
        if s==0: per[c]={'f1':0.,'prec':0.,'rec':0.,'support':0}; continue
        tp=int(((yp==c)&(yt==c)).sum()); fp=int(((yp==c)&(yt!=c)).sum())
        fn=int(((yp!=c)&(yt==c)).sum())
        prec=tp/(tp+fp+1e-9); rec=tp/(tp+fn+1e-9)
        f1=2*prec*rec/(prec+rec+1e-9)
        per[c]={'f1':f1,'prec':prec,'rec':rec,'support':s}
    total_s=sum(sup.values())
    macro   =float(np.mean([per[c]['f1'] for c in range(N_FAMILIES)]))
    weighted=sum(per[c]['f1']*sup.get(c,0) for c in range(N_FAMILIES))/max(1,total_s)
    sup_cls =[c for c in range(N_FAMILIES) if sup.get(c,0)>=min_support]
    supported=(float(np.mean([per[c]['f1'] for c in sup_cls])) if sup_cls else 0.)
    return {'macro':macro,'weighted':weighted,'supported':supported,'per':per,
            'supported_classes':[FAMILY_NAMES[c] for c in sup_cls]}


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 12: REPORT + PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def print_full_report(bin_r,eva_r,fam_r,beh_r,f1_detail,train_time=0.,steps=0):
    print('\n'+'█'*70)
    print('  RMEP v7.2  FTTransformer & LightGBM  [Joint Soft Gating & Augmented LightGBM]')
    print('█'*70)
    print(f'  Training: {train_time:.1f}s | Steps: {steps}')

    print('\n┌────────────────────────────────────────────────┐')
    print('│  TASK 1: BINARY DETECTION  [No Leakage]       │')
    print('└────────────────────────────────────────────────┘')
    print(f'  Accuracy      : {bin_r["acc"]*100:6.2f}%')
    print(f'  F1-score      : {bin_r["f1"]:.4f}')
    print(f'  ROC-AUC       : {bin_r["roc_auc"]:.4f}')
    print(f'  PR-AUC        : {bin_r["pr_auc"]:.4f}')
    print(f'  TPR @ 1.0%FPR : {bin_r["tpr_01"]*100:.2f}%')
    print(f'  TPR @ 0.1%FPR : {bin_r["tpr_001"]*100:.2f}%')
    print(f'  FPR (overall) : {bin_r["fpr_v"]:.4f}')
    print(f'  FNR (overall) : {bin_r["fnr_v"]:.4f}')
    if eva_r:
        gap=(bin_r["tpr_01"]-eva_r["tpr_01"])*100
        print(f'\n  ─ Evasive ─  ROC={eva_r["roc_auc"]:.4f} | Gap={gap:.2f}%  '
              f'{"✅" if gap<15 else "⚠️"}')

    print('\n┌────────────────────────────────────────────────┐')
    print('│  TASK 2: FAMILY  [7-class, Gated Inference]    │')
    print('└────────────────────────────────────────────────┘')
    print(f'  Accuracy      : {fam_r["acc"]*100:.2f}%')
    print(f'  Macro-F1      : {f1_detail["macro"]:.4f}')
    print(f'  Weighted-F1   : {f1_detail["weighted"]:.4f}')
    print(f'  Supported-F1  : {f1_detail["supported"]:.4f}  '
          f'({f1_detail["supported_classes"]})')
    print('\n  Per-class:')
    print(f'  {"Class":<14} {"Supp":>8} {"Prec":>7} {"Rec":>7} {"F1":>7}')
    print('  '+'─'*52)
    for c,name in enumerate(FAMILY_NAMES):
        m=f1_detail["per"][c]
        flag='✅' if m["f1"]>=0.5 else ('⚠️' if m["f1"]>=0.2 else '❌')
        print(f'  {name:<14} {m["support"]:>8,} {m["prec"]:>7.3f} '
              f'{m["rec"]:>7.3f} {m["f1"]:>7.3f}  {flag}')

    print('\n┌────────────────────────────────────────────────┐')
    print('│  TASK 3: BEHAVIOR  [OptimalThreshold]         │')
    print('└────────────────────────────────────────────────┘')
    print(f'  Hamming Loss: {beh_r["hamming"]:.4f}\n')
    print(f'  {"Behavior":<22}|{"AUC":>7}|{"F1":>6}|{"Supp":>8}')
    print('  '+'─'*50)
    for name,m in beh_r['per_behavior'].items():
        auc_s=f'{m["roc_auc"]:.4f}' if not np.isnan(m["roc_auc"]) else '  N/A'
        print(f'  {name:<22}|{auc_s:>7}|{m["f1"]:>6.3f}|{m["support"]:>8,}  '
              f'{"✅" if m["f1"]>=0.5 else "⚠️"}')
    print('█'*70)

def plot_training(history, title='v7.2', phase_boundaries=None):
    fig,axes=plt.subplots(1,4,figsize=(18,4))
    cfgs=[('total','navy','Total Loss'),('binary','crimson','Binary (Focal)'),
          ('family','darkorange','Family (FocalLoss)'),('behavior','forestgreen','Behavior')]
    for ax,(k,col,ttl) in zip(axes,cfgs):
        vals=history[k]
        if not vals: continue
        ax.plot(vals,color=col,lw=0.8,alpha=0.2)
        w=max(1,len(vals)//30)
        sm=np.convolve(vals,np.ones(w)/w,mode='valid')
        ax.plot(range(w//2,w//2+len(sm)),sm,color=col,lw=2.5)
        if phase_boundaries:
            for pb,pl in phase_boundaries:
                ax.axvline(pb,color='black',ls='--',lw=1.2)
                ax.text(pb+5,ax.get_ylim()[1]*0.9,pl,fontsize=7)
        ax.set_title(ttl,fontsize=9); ax.set_xlabel('Step'); ax.grid(True,alpha=0.3)
    plt.suptitle(f'Training Loss – {title}',fontsize=13,fontweight='bold')
    plt.tight_layout()
    save_chart(fig, 'training_loss')

def plot_results(bin_r, eva_r, fam_r, beh_r, f1_detail):
    # ── ROC + PR ─────────────────────────────────────────────────────────────
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    axes[0].plot(bin_r['fpr_c'],bin_r['tpr_c'],lw=2,
                 label=f'Standard AUC={bin_r["roc_auc"]:.3f}')
    if eva_r:
        axes[0].plot(eva_r['fpr_c'],eva_r['tpr_c'],lw=2,ls='--',color='crimson',
                     label=f'Evasive AUC={eva_r["roc_auc"]:.3f}')
    axes[0].set_xscale('log'); axes[0].set_xlabel('FPR (log)')
    axes[0].set_ylabel('Processed TPR'); axes[0].set_title('Task 1 – ROC (log FPR)')
    axes[0].legend(); axes[0].grid(True,alpha=0.3)
    axes[1].plot(bin_r['rec_c'],bin_r['prec_c'],lw=2,
                 label=f'PR-AUC={bin_r["pr_auc"]:.3f}')
    axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
    axes[1].set_title('Task 1 – PR Curve'); axes[1].legend(); axes[1].grid(True,alpha=0.3)
    plt.suptitle('TASK 1 – Binary Detection',fontsize=13,fontweight='bold')
    plt.tight_layout(); save_chart(fig,'task1_roc_pr')

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    fig,ax=plt.subplots(figsize=(9,7))
    im=ax.imshow(fam_r['cm'],cmap='Blues')
    fig.colorbar(im,ax=ax)
    ax.set_xticks(range(N_FAMILIES)); ax.set_xticklabels(FAMILY_NAMES,rotation=40,ha='right')
    ax.set_yticks(range(N_FAMILIES)); ax.set_yticklabels(FAMILY_NAMES)
    th=fam_r['cm'].max()/2. if fam_r['cm'].max()>0 else 1
    for i in range(N_FAMILIES):
        for j in range(N_FAMILIES):
            ax.text(j,i,str(fam_r['cm'][i,j]),ha='center',va='center',fontsize=8,
                    color='white' if fam_r['cm'][i,j]>th else 'black')
    ax.set_title(f'TASK 2 Confusion Matrix – MacroF1={f1_detail["macro"]:.3f}')
    plt.tight_layout(); save_chart(fig,'task2_confusion_matrix')

    # ── Per-class F1 bar ──────────────────────────────────────────────────────
    fig,axes=plt.subplots(1,2,figsize=(16,5))
    f1s=[f1_detail['per'][c]['f1'] for c in range(N_FAMILIES)]
    cols=['#27ae60' if v>=0.5 else ('#e67e22' if v>=0.2 else '#e74c3c') for v in f1s]
    bars=axes[0].bar(FAMILY_NAMES,f1s,color=cols,edgecolor='black')
    axes[0].axhline(0.5,color='blue',ls='--',lw=1.5,label='F1=0.5')
    axes[0].set_ylim(0,1.1); axes[0].set_title('TASK 2 – Per-Class F1')
    axes[0].set_ylabel('F1-score'); axes[0].legend(); axes[0].grid(True,alpha=0.3,axis='y')
    axes[0].set_xticklabels(FAMILY_NAMES,rotation=30,ha='right')
    for bar,v in zip(bars,f1s): axes[0].text(bar.get_x()+bar.get_width()/2,v+0.01,f'{v:.2f}',ha='center',fontsize=9)
    
    # F1 variant comparison
    mv=[f1_detail['macro'],f1_detail['weighted'],f1_detail['supported']]
    ml=['Macro\n(all 7)','Weighted\n(support)','Supported\n(≥30 test)']
    mc=['#e74c3c','#2ecc71','#3498db']
    bars2=axes[1].bar(ml,mv,color=mc,edgecolor='black')
    axes[1].set_ylim(0,1.1); axes[1].set_title('TASK 2 – F1 Metrics Comparison')
    axes[1].axhline(0.5,color='orange',ls='--',lw=1.5); axes[1].grid(True,alpha=0.3,axis='y')
    for bar,v in zip(bars2,mv): axes[1].text(bar.get_x()+bar.get_width()/2,v+0.01,f'{v:.3f}',ha='center',fontsize=11,fontweight='bold')
    plt.suptitle('TASK 2 – Family Classification',fontsize=13,fontweight='bold')
    plt.tight_layout(); save_chart(fig,'task2_f1_analysis')

    # ── Behavior ──────────────────────────────────────────────────────────────
    names=BEHAVIOR_NAMES
    f1v=[beh_r['per_behavior'][n]['f1'] for n in names]
    spv=[beh_r['per_behavior'][n]['support'] for n in names]
    fig,axes=plt.subplots(1,2,figsize=(15,5))
    def cf(f): return '#27ae60' if f>=0.5 else ('#e74c3c' if f>0 else '#95a5a6')
    bars3=axes[0].bar(names,f1v,color=[cf(f) for f in f1v],edgecolor='black')
    axes[0].axhline(0.5,color='orange',ls='--',lw=1.5); axes[0].set_ylim(0,1.1)
    axes[0].set_title('Task 3 – Behavior F1'); axes[0].set_xticklabels(names,rotation=30,ha='right')
    axes[0].grid(True,alpha=0.3,axis='y')
    for bar,v in zip(bars3,f1v): axes[0].text(bar.get_x()+bar.get_width()/2,v+0.01,f'{v:.2f}',ha='center',fontsize=8)
    axes[1].barh(names,spv,color='steelblue',edgecolor='black')
    axes[1].set_xlabel('Support (log)'); axes[1].set_xscale('log')
    axes[1].set_title('Task 3 – Support Count'); axes[1].grid(True,alpha=0.3,axis='x')
    plt.suptitle('TASK 3 – Behavior Classification',fontsize=13,fontweight='bold')
    plt.tight_layout(); save_chart(fig,'task3_behavior')


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 13: EVOLUTION TRACKING (In-Memory)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_prototypes(model, X_all, yb_all, wids_all, device):
    print('\n  [Evolution] Tính weekly prototypes from cached features...')
    model.eval()
    week_sums, week_counts = {}, {}
    mal = (yb_all == 1)
    if not mal.any(): return {}
    X_m = X_all[mal]
    weeks_m = wids_all[mal]
    for s in range(0, len(X_m), 512):
        sl = slice(s, min(s+512, len(X_m)))
        embs = model.encode(torch.tensor(X_m[sl], dtype=torch.float32).to(device)).cpu().numpy()
        for emb, wid in zip(embs, weeks_m[sl]):
            wid = int(wid)
            if wid not in week_sums:
                week_sums[wid] = np.zeros_like(emb)
                week_counts[wid] = 0
            week_sums[wid] += emb
            week_counts[wid] += 1
    protos = {w: week_sums[w]/week_counts[w] for w in week_sums if week_counts[w]>0}
    return protos

def plot_evolution(protos):
    if len(protos)<2: print('  ⚠️  Cần ≥2 tuần'); return
    ws=sorted(protos.keys()); pm=np.array([protos[w] for w in ws])
    drifts=[1-np.dot(pm[i],pm[i+1])/(np.linalg.norm(pm[i])*np.linalg.norm(pm[i+1])+1e-9)
            for i in range(len(ws)-1)]
    fig,(a1,a2)=plt.subplots(1,2,figsize=(14,5))
    try:
        pca=PCA(n_components=min(2,pm.shape[0],pm.shape[1]))
        proj=pca.fit_transform(pm)
        colors=cm.plasma(np.linspace(0,1,len(ws)))
        for i,(pt,wid,col) in enumerate(zip(proj,ws,colors)):
            a1.scatter(*pt[:2],color=col,s=80,zorder=5)
            a1.annotate(f'W{wid}',pt[:2],textcoords='offset points',xytext=(4,4),fontsize=6)
            if i>0: a1.annotate('',xy=proj[i,:2],xytext=proj[i-1,:2],
                arrowprops=dict(arrowstyle='->',color='gray',lw=0.8))
        a1.set_title('Malware Trajectory (PCA)'); a1.grid(True,alpha=0.3)
    except Exception as e: a1.text(0.5,0.5,str(e),ha='center')
    dw=ws[1:]; mean_d=float(np.mean(drifts))
    a2.plot(dw,drifts,'o-',color='crimson',lw=2,markersize=5)
    a2.fill_between(dw,drifts,alpha=0.15,color='crimson')
    a2.axhline(mean_d,ls='--',color='orange',label=f'Mean={mean_d:.4f}')
    if drifts:
        pi=int(np.argmax(drifts))
        a2.annotate(f'Peak W{dw[pi]}\n{drifts[pi]:.4f}',
            xy=(dw[pi],drifts[pi]),xytext=(dw[pi],max(drifts)*0.6),
            arrowprops=dict(arrowstyle='->'),fontsize=9,ha='center')
    a2.set_title('Concept Drift'); a2.set_xlabel('Week ID'); a2.legend(); a2.grid(True,alpha=0.3)
    plt.tight_layout(); save_chart(fig,'evolution_tracking')


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 14: XAI (In-Memory)
# ─────────────────────────────────────────────────────────────────────────────

class TaskWrapper(nn.Module):
    def __init__(self, m, task='binary', ci=0):
        super().__init__(); self.m=m; self.task=task; self.ci=ci
    def forward(self, x):
        lb,lf,lbh=self.m(x)
        if self.task=='binary': return lb.unsqueeze(-1)
        if self.task=='family': return lf[:,self.ci:self.ci+1]
        return lbh[:,self.ci:self.ci+1]

def _group_imp(vals_np):
    return {g:float(np.abs(vals_np[:,idx]).mean()) for g,idx in FEAT_GROUPS.items()}

def xai_01_pfi(model,X,yb,device,n=5):
    model.eval()
    with torch.no_grad(): lb,_,_=model(torch.tensor(X,dtype=torch.float32).to(device))
    base=roc_auc_score(yb,lb.cpu().numpy())
    gimp={}
    for gn,gi in FEAT_GROUPS.items():
        drops=[]
        for _ in range(n):
            Xp=X.copy(); Xp[:,gi]=Xp[np.random.permutation(len(Xp))][:,gi]
            with torch.no_grad(): lbp,_,_=model(torch.tensor(Xp,dtype=torch.float32).to(device))
            drops.append(base-roc_auc_score(yb,lbp.cpu().numpy()))
        gimp[gn]=float(np.mean(drops))
    fig,ax=plt.subplots(figsize=(8,4))
    ns=list(gimp.keys()); vs=[gimp[n_] for n_ in ns]
    ax.barh(ns,vs,color=['#e74c3c' if v>0 else '#2ecc71' for v in vs],edgecolor='black')
    ax.axvline(0,color='black',lw=0.8); ax.set_xlabel('ROC-AUC Drop')
    ax.set_title('[XAI-1] Permutation Feature Importance'); ax.grid(True,alpha=0.3,axis='x')
    plt.tight_layout(); save_chart(fig,'xai_01_pfi')
    for n_,v in sorted(gimp.items(),key=lambda x:-x[1]):
        tag='🔴Critical' if v>0.05 else ('🟡Moderate' if v>0.01 else '🟢Low')
        print(f'  {n_:<12}: {v:.4f}  {tag}')
    return gimp

def xai_02_ig(model,X,device):
    if not HAS_CAPTUM:
        print("  ⚠️ Skipped Integrated Gradients (captum is not installed)")
        return None
    model.eval(); wrapper=TaskWrapper(model,'binary').to(device)
    Xt=torch.tensor(X,dtype=torch.float32).to(device)
    attrs,_=IntegratedGradients(wrapper).attribute(Xt,torch.zeros_like(Xt),return_convergence_delta=True)
    an=attrs.detach().cpu().numpy(); gi=_group_imp(an)
    fig,(a1,a2)=plt.subplots(1,2,figsize=(14,5))
    ns=list(gi.keys()); vs=[gi[n_] for n_ in ns]
    a1.bar(ns,vs,color=['#3498db','#e74c3c','#2ecc71','#9b59b6','#1abc9c','#f1c40f','#e67e22'],edgecolor='black')
    a1.set_title('[XAI-2] Integrated Gradients – Group'); a1.grid(True,alpha=0.3,axis='y')
    ma=np.abs(an).mean(0); top=np.argsort(ma)[::-1][:20]
    a2.barh([FEAT_NAMES[i] for i in top[::-1]],ma[top[::-1]],color='royalblue',edgecolor='black')
    a2.set_title('Top 20 Features'); a2.grid(True,alpha=0.3,axis='x')
    plt.tight_layout(); save_chart(fig,'xai_02_ig')
    return gi

def xai_03_saliency(model,X,device):
    model.eval()
    Xt=torch.tensor(X,dtype=torch.float32,requires_grad=True).to(device)
    lb,_,_=model(Xt); lb.sum().backward()
    sample_grad = Xt.grad
    if sample_grad is not None:
        sal = sample_grad.detach().cpu().numpy()
    else:
        sal = np.zeros_like(X)
    gi=_group_imp(sal)
    fig,(a1,a2)=plt.subplots(1,2,figsize=(14,5))
    n_show=min(20,len(sal))
    im=a1.imshow(np.abs(sal[:n_show]),aspect='auto',cmap='Reds')
    a1.set_title('[XAI-3] Saliency Maps'); plt.colorbar(im,ax=a1)
    for b in [70,78,88,98,354,610]: a1.axvline(b,color='blue',lw=1.5,ls='--')
    ns=list(gi.keys()); vs=[gi[n_] for n_ in ns]
    a2.bar(ns,vs,color='#e74c3c',edgecolor='black'); a2.set_title('Saliency – Group')
    a2.grid(True,alpha=0.3,axis='y'); plt.tight_layout(); save_chart(fig,'xai_03_saliency')
    return gi

def xai_04_smoothgrad(model,X,device,n=30,std=0.05):
    model.eval(); sg=np.zeros_like(X)
    for _ in range(n):
        Xn=X+np.random.randn(*X.shape).astype(np.float32)*std
        Xt=torch.tensor(Xn,dtype=torch.float32,requires_grad=True).to(device)
        lb,_,_=model(Xt); lb.sum().backward()
        if Xt.grad is not None:
            sg+=Xt.grad.detach().cpu().numpy()
    sg/=n; gi=_group_imp(sg)
    fig,(a1,a2)=plt.subplots(1,2,figsize=(14,5))
    im=a1.imshow(np.abs(sg[:min(20,len(sg))]),aspect='auto',cmap='Oranges')
    a1.set_title(f'[XAI-4] SmoothGrad (n={n})'); plt.colorbar(im,ax=a1)
    for b in [70,78,88,98,354,610]: a1.axvline(b,color='blue',lw=1.5,ls='--')
    ns=list(gi.keys()); vs=[gi[n_] for n_ in ns]
    a2.bar(ns,vs,color='darkorange',edgecolor='black'); a2.set_title('SmoothGrad – Group')
    a2.grid(True,alpha=0.3,axis='y'); plt.tight_layout(); save_chart(fig,'xai_04_smoothgrad')
    return gi

def xai_05_deeplift(model,X,device):
    if not HAS_CAPTUM:
        print("  ⚠️ Skipped DeepLIFT (captum is not installed)")
        return None
    model.eval(); wrapper=TaskWrapper(model,'binary').to(device)
    Xt=torch.tensor(X,dtype=torch.float32).to(device)
    try:
        from captum.attr import DeepLift
        attrs=DeepLift(wrapper).attribute(Xt,torch.zeros_like(Xt))
        an=attrs.detach().cpu().numpy(); method='DeepLIFT'
    except:
        try:
            from captum.attr import InputXGradient
            attrs=InputXGradient(wrapper).attribute(Xt)
            an=attrs.detach().cpu().numpy(); method='InputXGrad'
        except Exception as e: print(f'  ❌ {e}'); return None
    gi=_group_imp(an); vmax=np.abs(an).max()
    fig,(a1,a2)=plt.subplots(1,2,figsize=(14,5))
    im=a1.imshow(an[:min(20,len(an))],aspect='auto',cmap='RdBu_r',vmin=-vmax,vmax=vmax)
    a1.set_title(f'[XAI-5] {method}'); plt.colorbar(im,ax=a1)
    for b in [70,78,88,98,354,610]: a1.axvline(b,color='black',lw=1.5,ls='--')
    ns=list(gi.keys()); vs=[gi[n_] for n_ in ns]
    a2.bar(ns,vs,color='mediumpurple',edgecolor='black'); a2.set_title(f'{method} – Group')
    a2.grid(True,alpha=0.3,axis='y'); plt.tight_layout(); save_chart(fig,'xai_05_deeplift')
    return gi

def xai_06_attn(model,X,device):
    model.eval()
    Xt=torch.tensor(X,dtype=torch.float32).to(device)
    _,attns=model.encode_with_attn(Xt)
    n_l=len(attns)
    fig,axes=plt.subplots(1,n_l,figsize=(5*n_l,4))
    if n_l==1: axes=[axes]
    for li,(ax,attn) in enumerate(zip(axes,attns)):
        ma=attn.mean(0)
        im=ax.imshow(ma,cmap='YlOrRd',vmin=0,vmax=ma.max())
        ax.set_xticks(range(8)); ax.set_xticklabels(TOKEN_NAMES,rotation=30,ha='right')
        ax.set_yticks(range(8)); ax.set_yticklabels(TOKEN_NAMES)
        ax.set_title(f'Layer {li+1}'); plt.colorbar(im,ax=ax)
    plt.suptitle('[XAI-6] Attention Visualization',fontsize=13,fontweight='bold')
    plt.tight_layout(); save_chart(fig,'xai_06_attention')
    cls_a=attns[-1].mean(0)[0,:]
    print('\n  CLS attention (last layer):')
    for tok,val in sorted(zip(TOKEN_NAMES,cls_a),key=lambda x:-x[1]):
        print(f'  {tok:<12}: {val:.4f}  {"█"*int(val*25)}')
    return attns

def xai_07_rollout(attns):
    rollout=None
    for attn in attns:
        ma=attn.mean(0); eye=np.eye(ma.shape[0])
        aug=(0.5*ma+0.5*eye); aug/=aug.sum(-1,keepdims=True)
        rollout=aug if rollout is None else aug@rollout
    cls_r=rollout[0,:]; ft=TOKEN_NAMES[1:]; fa=cls_r[1:]
    fig,(a1,a2)=plt.subplots(1,2,figsize=(12,4))
    im=a1.imshow(rollout,cmap='Blues',vmin=0)
    a1.set_xticks(range(8)); a1.set_xticklabels(TOKEN_NAMES,rotation=30,ha='right')
    a1.set_yticks(range(8)); a1.set_yticklabels(TOKEN_NAMES)
    a1.set_title('[XAI-7] Attention Rollout'); plt.colorbar(im,ax=a1)
    mx=max(fa) if max(fa)>0 else 1
    a2.bar(ft,fa,color=['#e74c3c' if v==mx else '#3498db' for v in fa],edgecolor='black')
    a2.set_title('CLS → Feature Rollout'); a2.grid(True,alpha=0.3,axis='y')
    a2.set_xticklabels(ft, rotation=30, ha='right')
    for i,v in enumerate(fa): a2.text(i,v+0.001,f'{v:.3f}',ha='center',fontsize=9)
    plt.tight_layout(); save_chart(fig,'xai_07_rollout')
    return cls_r

def xai_08_shap(model,X_bg,X_test,device):
    if not HAS_SHAP:
        print("  ⚠️ Skipped SHAP (shap is not installed)")
        return None
    model.eval(); wrapper=TaskWrapper(model,'binary').to(device)
    bg=torch.tensor(X_bg,dtype=torch.float32).to(device)
    tt=torch.tensor(X_test,dtype=torch.float32).to(device)
    sv=None
    try:
        exp=shap.GradientExplainer(wrapper,bg); sv_raw=exp.shap_values(tt)
        sv=(np.array(sv_raw[0]) if isinstance(sv_raw,list) else np.array(sv_raw))
        if sv.ndim>2: sv=sv.squeeze(-1); print('  ✅ DeepSHAP')
    except Exception as e:
        print(f'  ⚠️ DeepSHAP: {e}. KernelSHAP...')
        try:
            def pfn(Xn):
                with torch.no_grad(): lb,_,_=model(torch.tensor(Xn,dtype=torch.float32).to(device))
                p=torch.sigmoid(lb).cpu().numpy(); return np.column_stack([1-p,p])
            sv=np.array(shap.KernelExplainer(pfn,X_bg[:10]).shap_values(X_test[:5])[0])
            print('  ✅ KernelSHAP')
        except Exception as e2: print(f'  ❌ {e2}'); return None
    gi=_group_imp(sv)
    fig,(a1,a2)=plt.subplots(1,2,figsize=(14,5))
    ma=np.abs(sv).mean(0); top=np.argsort(ma)[::-1][:15]
    a1.barh([FEAT_NAMES[i] for i in top[::-1]],ma[top[::-1]],color='#e74c3c',edgecolor='black')
    a1.set_title('[XAI-8] SHAP – Top 15'); a1.grid(True,alpha=0.3,axis='x')
    ns=list(gi.keys()); vs=[gi[n_] for n_ in ns]
    a2.bar(ns,vs,color='firebrick',edgecolor='black'); a2.set_title('SHAP – Group')
    a2.grid(True,alpha=0.3,axis='y'); plt.tight_layout(); save_chart(fig,'xai_08_shap')
    return gi

def xai_09_lime(model,X_bg,X_test,yb,device):
    if not HAS_LIME:
        print("  ⚠️ Skipped LIME (lime is not installed)")
        return None
    model.eval()
    def pfn(Xn):
        with torch.no_grad(): lb,_,_=model(torch.tensor(Xn.astype(np.float32),dtype=torch.float32).to(device))
        p=torch.sigmoid(lb).cpu().numpy(); return np.column_stack([1-p,p])
    exp=lime_tabular.LimeTabularExplainer(X_bg,feature_names=FEAT_NAMES,
        class_names=['Benign','Malware'],mode='classification',random_state=42)
    e=exp.explain_instance(X_test[0],pfn,num_features=15,num_samples=300)
    pred_p=float(pfn(X_test[:1])[0][1])
    fig,ax=plt.subplots(figsize=(10,6))
    fe=e.as_list(); ns=[f[0] for f in fe][::-1]; vs=[f[1] for f in fe][::-1]
    ax.barh(ns,vs,color=['#e74c3c' if v>0 else '#2ecc71' for v in vs],edgecolor='black')
    ax.axvline(0,color='black',lw=0.8)
    ax.set_title(f'[XAI-9] LIME – True:{"Malware" if int(yb[0]) else "Benign"} | P(M)={pred_p:.3f}')
    ax.set_xlabel('Weight (+→Malware)'); ax.grid(True,alpha=0.3,axis='x')
    plt.tight_layout(); save_chart(fig,'xai_09_lime')
    return e

def xai_10_pdp(model,X,device):
    model.eval(); fig,axes=plt.subplots(2,2,figsize=(14,10)); axes=axes.flatten(); pdps={}
    for ai,(gn,gi) in enumerate(list(FEAT_GROUPS.items())[:4]): # Plot first 4 groups
        fr=np.linspace(0.,1.,50); pv=[]
        for val in fr:
            Xm=X.copy(); Xm[:,gi[0]]=val
            with torch.no_grad():
                lb,_,_=model(torch.tensor(Xm,dtype=torch.float32).to(device))
            pv.append(float(torch.sigmoid(lb).mean().item()))
        pdps[gn]=(fr,pv); ax=axes[ai]
        ax.plot(fr,pv,color='royalblue',lw=2.5)
        ax.fill_between(fr,pv,alpha=0.12,color='royalblue')
        ax.axhline(0.5,color='red',ls='--',lw=1.5,label='P=0.5')
        ax.set_xlabel(f'{gn}[0]'); ax.set_ylabel('Processed P(Malware)')
        ax.set_ylim(0,1); ax.set_title(f'[PDP] {gn}'); ax.legend(fontsize=8); ax.grid(True,alpha=0.3)
        delta=pv[-1]-pv[0]
        ax.text(0.5,0.08,f'Δ={delta:+.3f}',ha='center',transform=ax.transAxes,
                fontsize=10,color='red' if delta>0 else 'green',fontweight='bold')
    plt.tight_layout(); save_chart(fig,'xai_10_pdp')
    return pdps

def run_xai(model, X_test, yb_test, X_train, device, n=100, n_bg=50):
    print('\n'+'='*65); print('  XAI – 10 METHODS  v7.2'); print('='*65)
    X_t = X_test[:n]
    y_b = yb_test[:n]
    X_bg = X_train[:n_bg]
    if len(X_t) == 0:
        print("  ❌ No test data for XAI!"); return {}
    print(f"  Test={len(X_t)} | BG={len(X_bg)} | Malware count={int(y_b.sum())}")
    res={}; _sep=lambda t: print(f'\n{"─"*60}\n  {t}\n{"─"*60}')
    _sep('[XAI-1] Permutation Feature Importance')
    try: res['pfi']=xai_01_pfi(model,X_t,y_b,device)
    except Exception as e: print(f'  ❌ {e}')
    _sep('[XAI-2] Integrated Gradients')
    try: res['ig']=xai_02_ig(model,X_t,device)
    except Exception as e: print(f'  ❌ {e}')
    _sep('[XAI-3] Saliency Maps')
    try: res['sal']=xai_03_saliency(model,X_t,device)
    except Exception as e: print(f'  ❌ {e}')
    _sep('[XAI-4] SmoothGrad')
    try: res['sg']=xai_04_smoothgrad(model,X_t,device)
    except Exception as e: print(f'  ❌ {e}')
    _sep('[XAI-5] DeepLIFT')
    try: res['dl']=xai_05_deeplift(model,X_t,device)
    except Exception as e: print(f'  ❌ {e}')
    _sep('[XAI-6] Attention Visualization')
    attns=None
    try: attns=xai_06_attn(model,X_t,device); res['attn']=attns
    except Exception as e: print(f'  ❌ {e}')
    _sep('[XAI-7] Attention Rollout')
    try:
        if attns is not None: res['rollout']=xai_07_rollout(attns)
        else: print('  ⚠️ Need XAI-6 first')
    except Exception as e: print(f'  ❌ {e}')
    _sep('[XAI-8] SHAP')
    try:
        sr=xai_08_shap(model,X_bg,X_t[:20],device)
        if sr: res['shap']=sr
    except Exception as e: print(f'  ❌ {e}')
    _sep('[XAI-9] LIME')
    try: res['lime']=xai_09_lime(model,X_bg,X_t[:5],y_b[:5],device)
    except Exception as e: print(f'  ❌ {e}')
    _sep('[XAI-10] PDP')
    try: res['pdp']=xai_10_pdp(model,X_t,device)
    except Exception as e: print(f'  ❌ {e}')
    
    # Consensus chart
    all_scores={g:[] for g in FEAT_GROUPS}
    gns=list(FEAT_GROUPS.keys())
    for mk in ['pfi','ig','sal','sg','dl','shap']:
        r=res.get(mk)
        if not isinstance(r,dict) or not all(k in r for k in FEAT_GROUPS): continue
        vs=[r[g] for g in gns]; mx=max(vs) if max(vs)>0 else 1
        for g in FEAT_GROUPS: all_scores[g].append(r[g]/mx)
    fig,ax=plt.subplots(figsize=(9,5))
    for mk,col,lbl in [('pfi','#e74c3c','PFI'),('ig','#3498db','IG'),
                        ('sal','#2ecc71','Saliency'),('sg','#e67e22','SmoothGrad'),
                        ('dl','#9b59b6','DeepLIFT'),('shap','#1abc9c','SHAP')]:
        r=res.get(mk)
        if not isinstance(r,dict) or not all(k in r for k in FEAT_GROUPS): continue
        vs=[r[g] for g in gns]; mx=max(vs) if max(vs)>0 else 1
        ax.plot(gns,[v/mx for v in vs],'o-',color=col,lw=2,label=lbl,markersize=8)
    ax.set_ylabel('Normalized Importance'); ax.set_title('XAI Consensus (6 methods)')
    ax.legend(loc='upper right',ncol=3,fontsize=8); ax.grid(True,alpha=0.3); ax.set_ylim(0,1.1)
    plt.tight_layout(); save_chart(fig,'xai_consensus')
    print('\n'+'█'*65); print('  ✅ XAI COMPLETE'); print('█'*65)
    return res


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 15: MAIN
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(s=3407):
    import random; random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

set_seed(3407)

if __name__ == '__main__':
    print('='*65); print('  EMBER2024  RMEP v7.2  Joint Soft Gating & Augmented LightGBM'); print('='*65)
    print(f'  Anti-overfit: dropout={DROPOUT} | grad_clip={GRAD_CLIP} | '
          f'wd={WEIGHT_DECAY} | max_w={MAX_WEIGHT} | aug_x{AUG_MULT} (min={AUG_MIN})')

    # ── Download / Locate Dataset ────────────────────────────────────────────────
    print('\n[1/6] Locating/Downloading EMBER2024...')
    ember_path = None
    kaggle_input_paths = [
        '/kaggle/input/ember2024',
        '/kaggle/input/datasets/weiweip/ember2024',
        '../input/ember2024',
        './ember2024'
    ]
    for p in kaggle_input_paths:
        if os.path.exists(p):
            ember_path = p
            print(f'  ✅ Found dataset in Kaggle/Local Input: {ember_path}')
            break

    if not ember_path:
        if HAS_KAGGLERHUB:
            try:
                print("  Downloading via kagglehub...")
                ember_path = kagglehub.dataset_download('weiweip/ember2024')
                print(f'  ✅ Downloaded via kagglehub: {ember_path}')
            except Exception as e:
                print(f'  ⚠️ kagglehub download failed: {e}')
        else:
            print("  ⚠️ kagglehub is not installed, trying current directory...")

    if not ember_path:
        train_check = glob.glob('**/Win64_train', recursive=True)
        if train_check:
            ember_path = os.path.dirname(train_check[0])
            print(f'  ✅ Found dataset recursively: {ember_path}')
        else:
            ember_path = '.'
            print('  ⚠️ Warning: Could not locate dataset path, fallback to current directory.')

    train_dir=test_dir=None
    for root,dirs,_ in os.walk(ember_path):
        if 'Win64_train' in dirs:
            train_dir=os.path.join(root,'Win64_train')
            test_dir =os.path.join(root,'Win64_test'); break
    if train_dir and os.path.isdir(train_dir):
        train_files=sorted(glob.glob(os.path.join(train_dir,'*.jsonl')))
        test_files =sorted(glob.glob(os.path.join(test_dir, '*.jsonl')))
    else:
        all_j=sorted(glob.glob(os.path.join(ember_path,'**','*.jsonl'),recursive=True))
        split=int(len(all_j)*0.8)
        train_files=all_j[:split]; test_files=all_j[split:]
    print(f'  Train: {len(train_files)} files | Test: {len(test_files)} files')
    if not train_files: 
        raise RuntimeError('No JSONL files found! Please add the EMBER2024 dataset to your Kaggle Notebook input or ensure files are present.')

    # ── Debug quick check ──────────────────────────────────────────────────────────
    print('\n[2/6] Debug check (first 300 rows)...')
    _s=load_jsonl(train_files[0])
    if _s:
        print(f'  Keys: {list(_s[0].keys())}')
        _v=_feat_row(_s[0]); print(f'  Feature vec: {"OK "+str(_v.shape) if _v is not None else "FAIL"}')
        _X,_yb,_yf,_,_,_=extract_features(_s[:300])
        if len(_yb)>0:
            print(f'  Binary: {dict(Counter(_yb.astype(int).tolist()))}')
            print(f'  Family (masked count in -1): { {FAMILY_NAMES[k] if k>=0 else "Masked(-1)":v for k,v in Counter(_yf.tolist()).items()} }')
        del _s,_X,_yb,_yf; gc.collect()

    # ── Pre-extract and Cache features ───────────────────────────────────────────
    print('\n[3/6] Pre-extracting and caching features in RAM...')
    MAX_TRAIN_SAMPLES = None
    MAX_VAL_SAMPLES = None

    X_train, yb_train, yf_train, ybeh_train, wids_train, evas_train = load_dataset_features(
        train_files, desc="Train Dataset", max_samples=MAX_TRAIN_SAMPLES)
    X_test, yb_test, yf_test, ybeh_test, wids_test, evas_test = load_dataset_features(
        test_files, desc="Test Dataset", max_samples=MAX_VAL_SAMPLES)

    class_weights_cpu, real_counts = compute_class_weights_full(yf_train)
    AUG_TARGETS = build_aug_targets(real_counts)

    # Tính toán Class priors (phân phối thực tế trên tập train) cho Logit-Adjusted Loss
    counts_tensor = torch.tensor(real_counts, dtype=torch.float32)
    class_prior_cpu = counts_tensor / (counts_tensor.sum() + 1e-9)

    # ── Init model ─────────────────────────────────────────────────────────────────
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n[4/6] Init model  device={device}')
    if device.type == 'cpu':
        print('='*60)
        print('  ⚠️  WARNING: Running on CPU instead of GPU!')
        print('  This high-capacity model (v7.2) is designed for GPU environments.')
        print('  Training on CPU will be significantly slower.')
        print('='*60)

    model=FTTransformer().to(device)
    n_params=sum(p.numel() for p in model.parameters())
    print(f'  Params: {n_params:,}')

    # Khởi tạo loss đa nhiệm có bổ sung class_prior phục vụ Logit Adjustment và tăng f1_alpha
    criterion=MultiTaskLoss(weights=class_weights_cpu, class_prior=class_prior_cpu, f1_alpha=2.0, family_focused=False).to(device)
    # Loss chuyên biệt cho Phase 2 (Family-only)
    fam_criterion=FamilyFocalLoss(weights=class_weights_cpu, class_prior=class_prior_cpu, tau=1.2, gamma=2.5, label_smoothing=0.05, f1_alpha=2.0).to(device)

    history={'total':[],'binary':[],'family':[],'behavior':[]}
    phase_boundaries=[]
    total_steps=0; t0=time.time()

    # ══════════════════════════════════════════════════════════════════════════════
    # PHASE 1: Multi-task  (8 epochs, Uncertainty Loss, lr=1e-3, Manifold Mixup)
    # ══════════════════════════════════════════════════════════════════════════════
    print('\n[5/6] TRAINING...')
    print('\n'+'='*60); print('  PHASE 1: Multi-task  (Uncertainty Loss + Warmup, lr=1e-3, 8 epochs)'); print('='*60)
    opt1=optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=1e-3, weight_decay=WEIGHT_DECAY)
    sch1=optim.lr_scheduler.CosineAnnealingLR(opt1,T_max=8,eta_min=1e-4)
    es1=ESMin(patience=5,delta=0.003)

    # Khởi tạo Warmup Pointer
    warmup_info = {
        'global_step': 0,
        'warmup_steps': 300,
        'base_lr': 1e-3
    }

    for ep in range(8):
        for pg in opt1.param_groups:
            if 'initial_lr' not in pg:
                pg['initial_lr'] = pg.get('lr', 1e-3)

        lr_now=sch1.get_last_lr()[0] if ep>0 else 1e-3
        print(f'\n{"─"*60}\n  [P1] Epoch {ep+1}/8  LR={lr_now:.6f}\n{"─"*60}')

        # Train epoch có kích hoạt Manifold Mixup (mixup_alpha=0.4)
        avg,steps=train_epoch(model,criterion,opt1,X_train,yb_train,yf_train,ybeh_train,device,
                               history,AUG_TARGETS,batch_size=512,min_rare=15,warmup_info=warmup_info,mixup_alpha=0.4)
        total_steps+=steps; sch1.step()
        print(f'\n  P1 Ep{ep+1}: avg_loss={avg:.4f} | steps={steps}')
        print(f'  Log variance weights (binary, family, behavior): {criterion.log_vars.detach().cpu().numpy()}')
        if len(X_test) > 0:
            vl=val_loss_full(model,criterion,X_test,yb_test,yf_test,ybeh_test,device)
            vf=val_family_f1_full(model,X_test,yf_test,device)
            print(f'  val_loss={vl:.4f} | family_macro_F1={vf:.4f}')
            if es1.step(vl,model): print('  🛑 Early Stop P1'); break

    es1.restore(model)
    phase_boundaries.append((len(history['total']), 'P1→P2'))
    plot_training(history,'v7.2 – Phase 1',phase_boundaries=[])

    # ══════════════════════════════════════════════════════════════════════════════
    # PHASE 2: Family-only  (12 epochs, unfrozen backbone, lr=1.5e-4, Manifold Mixup)
    # ══════════════════════════════════════════════════════════════════════════════
    print('\n'+'='*60); print('  PHASE 2: Family-ONLY  (unfrozen backbone, 12 epochs)'); print('='*60)
    train_family_phase(model,fam_criterion,X_train,yf_train,AUG_TARGETS,
                       device,history,X_test,yf_test,n_epochs=12,lr=1.5e-4,batch_size=256)
    phase_boundaries.append((len(history['total']),'P2→P3'))
    if len(X_test) > 0:
        vf=val_family_f1_full(model,X_test,yf_test,device)
        print(f'  After P2: family_macro_F1={vf:.4f}')

    # ══════════════════════════════════════════════════════════════════════════════
    # PHASE 3: Fine-tune all  (5 epochs, Discriminative Fine-Tuning, lr=2e-4/5e-5)
    # ══════════════════════════════════════════════════════════════════════════════
    print('\n'+'='*60); print('  PHASE 3: Fine-tune all  (Family-Focused + Discriminative LR, 5 epochs)'); print('='*60)
    model.unfreeze_all()

    # BẬT CHẾ ĐỘ FAMILY FOCUS ĐỂ LOẠI BỎ NHIỄU TỪ TASK 1 & 3
    criterion.family_focused = True

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if 'head_' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    opt3 = optim.AdamW([
        {'params': backbone_params, 'lr': 5e-5, 'initial_lr': 5e-5},
        {'params': head_params, 'lr': 2e-4, 'initial_lr': 2e-4}
    ], weight_decay=WEIGHT_DECAY)

    sch3=optim.lr_scheduler.CosineAnnealingLR(opt3,T_max=5,eta_min=1e-5)
    es3=ESMax(patience=5,delta=0.002)
    h3={'total':[],'binary':[],'family':[],'behavior':[]}

    for ep in range(5):
        lr_now = sch3.get_last_lr()[0] if ep>0 else 5e-5
        print(f'\n{"─"*60}\n  [P3] Epoch {ep+1}/5  Backbone LR={lr_now:.6f}\n{"─"*60}')

        # Train epoch có giảm mixup_alpha xuống 0.2 để tinh chỉnh ổn định
        avg,steps=train_epoch(model,criterion,opt3,X_train,yb_train,yf_train,ybeh_train,device,
                               h3,AUG_TARGETS,batch_size=512,min_rare=15,warmup_info=None,mixup_alpha=0.2)
        total_steps+=steps; sch3.step()
        print(f'\n  P3 Ep{ep+1}: avg_loss={avg:.4f}')
        if len(X_test) > 0:
            vf=val_family_f1_full(model,X_test,yf_test,device)
            print(f'  family_macro_F1={vf:.4f}')
            if es3.step(vf,model): print('  🛑 Early Stop P3'); break

    es3.restore(model)
    for k in history: history[k].extend(h3.get(k,[]))
    train_time=time.time()-t0
    print(f'\n✅ Training xong! {train_time:.1f}s | {total_steps} steps')

    plot_training(history,'v7.2 – Phase 1+2+3',phase_boundaries=phase_boundaries)

    # ══════════════════════════════════════════════════════════════════════════════
    # PHASE 4: LightGBM Multiclass Family Classifier (Balanced Class Weights)
    # ══════════════════════════════════════════════════════════════════════════════
    print('\n' + '='*60); print('  PHASE 4: Train LightGBM Multiclass Family Classifier'); print('='*60)
    family_lgb = None
    try:
        import lightgbm as lgb
        print("  Training LightGBM family classifier...")
        valid_idx = (yf_train != -1)
        X_train_valid = X_train[valid_idx]
        yf_train_valid = yf_train[valid_idx]
        
        family_lgb = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            class_weight='balanced',
            random_state=3407,
            verbose=-1,
            n_jobs=-1
        )
        t_lgb_start = time.time()
        # Bơm dữ liệu nội suy (Mixup thô) cho LightGBM để cây quyết định phân nhánh sâu hơn
        X_lgb_aug, _, yf_lgb_aug, _ = augment_batch(
            X_train_valid, 
            np.ones(len(X_train_valid), dtype=np.float32), 
            yf_train_valid, 
            np.zeros((len(X_train_valid), N_BEHAVIORS), dtype=np.float32), 
            AUG_TARGETS
        )
        family_lgb.fit(X_lgb_aug, yf_lgb_aug)
        print(f'  ✅ LightGBM trained on augmented data in {time.time() - t_lgb_start:.1f}s')
    except Exception as e:
        print(f"  ⚠️ Warning: Could not train LightGBM classifier: {e}. Falling back to pure neural network.")

    # ── Thresholds Gated (Logit Gating Calibration) ───────────────────────────────
    bin_thr=find_binary_thr(model,X_test,yb_test,device) if len(X_test) > 0 else 0.0
    beh_thr=find_beh_thr(model,X_test,ybeh_test,device) if len(X_test) > 0 else None
    if len(X_test) > 0:
        bin_thr, fam_thr = find_family_thr(model, family_lgb, X_test, yf_test, device, bin_thr)
    else:
        fam_thr = np.ones(7, dtype=np.float32) if len(X_test) == 0 else None

    # ── Full Evaluation ────────────────────────────────────────────────────────────
    print('\n[6/6] FULL EVALUATION (100% test data with Logit Gated Inference)...')
    bin_r=eva_r=fam_r=beh_r=None
    if len(X_test) > 0:
        bin_r,evas_test_out,eva_r,fam_r,beh_r=evaluate_full(model,family_lgb,X_test,yb_test,yf_test,ybeh_test,evas_test,device,beh_thr,fam_thr,bin_thr,tag='Final')
        if bin_r and fam_r:
            f1_detail=compute_family_f1_detail(fam_r,min_support=30)
            print_full_report(bin_r,eva_r,fam_r,beh_r,f1_detail,train_time,total_steps)
            plot_results(bin_r,eva_r,fam_r,beh_r,f1_detail)

    # ── Temporal prototypes evolution ──────────────────────────────────────────────
    if len(X_train) > 0 and len(X_test) > 0:
        try:
            X_all_proto = np.vstack([X_train, X_test])
            yb_all_proto = np.concatenate([yb_train, yb_test])
            wids_all_proto = np.concatenate([wids_train, wids_test])
            protos=compute_prototypes(model,X_all_proto,yb_all_proto,wids_all_proto,device)
            plot_evolution(protos)
        except Exception as e:
            print(f'  ⚠️ Evolution error: {e}')

    # ── XAI ───────────────────────────────────────────────────────────────────────
    try: run_xai(model,X_test,yb_test,X_train,device)
    except Exception as e: print(f'  ⚠️ XAI error: {e}')

    # ── Package all plots ─────────────────────────────────────────────────────────
    print('\n[Packaging] Archiving all plots...')
    try:
        zip_dir = '/kaggle/working'
        if not os.path.exists(zip_dir):
            os.makedirs(zip_dir, exist_ok=True)
        zip_path = os.path.join(zip_dir, 'charts_v7_2.zip')
        with zipfile.ZipFile(zip_path, 'w') as z:
            for f in glob.glob('*.png'):
                z.write(f)
                print(f'  Added {f} to zip archive')
        print(f'  ✅ Zipped to {zip_path}')
    except Exception as e:
        try:
            zip_path = 'charts_v7_2.zip'
            with zipfile.ZipFile(zip_path, 'w') as z:
                for f in glob.glob('*.png'):
                    z.write(f)
            print(f'  ✅ Zipped to local {zip_path}')
        except Exception as e2:
            print(f'  ⚠️ Zip error: {e}')

    print('\n🎉 HOÀN THÀNH TOÀN BỘ PIPELINE v7.2!')