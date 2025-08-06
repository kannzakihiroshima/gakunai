# sweep_and_test_6methods.py – multi‑distance, CSV OD, max‑track gallery
# -*- coding: utf-8 -*-
"""
・exp1 で各距離関数の最良しきい値を決定
・その閾値を exp2 / exp3 に適用し F1 と OD 表を評価
・OD 表は
    └ all‑labels
    └ label1‑6 別
  を ground‑truth / predicted の両方で作成し csv 保存
"""

# --- dummy modules (最上部) ---------------------------
import sys, types
sys.modules['tensorflow'] = types.ModuleType("tensorflow")      # POT が見に行ってもOK
tb = types.ModuleType("tensorboard")
tb.SummaryWriter = lambda *a, **k: None
sys.modules['tensorboard'] = tb
sys.modules['torch.utils.tensorboard'] = tb
# -------------------------------------------------------

# ────────── Imports ──────────
import os, re, csv, numpy as np, torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from torchreid.reid.utils import FeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
import ot
import cv2      # ← まだ無ければ追加

# ────────── Config ──────────
BASE_DIR   = r"C:/Users/sugie/PycharmProjects/StrongSORT-YOLO/data_rename"
OUT_DIR    = "od_csv_data_rename_0pad"                              # 生成 CSV 出力先
TH_RANGE   = np.linspace(0, 0.60, 31)              # sweep 範囲
MODEL_NAME = "osnet_ain_x1_0"
MODEL_PATH = r"C:/Users/sugie/PycharmProjects/pythonProject10/MARS/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter (1).pth"
TRACK_TXT_PATTERN = "exp{exp}_point{loc}.txt"
PERSON_LABEL_MAP = {"ichimanda":1,"ichiraku":2,"idogawa":3,
                    "miyazaki":4,"noguchi":5,"sakai":6}
IMG_EXTS = (".jpg",".jpeg",".png",".bmp",".tif",".tiff")

os.makedirs(OUT_DIR, exist_ok=True)
META, MISSING_TIME = {}, []          # global metadata

# ────────── Helper ──────────
def parse_loc_ord(fname):
    m=re.search(r'point(\d+)',fname)
    if not m: return None,None
    d=m.group(1);  return int(d[0]), int(d[1:] or 0)

def build_track_idx(exp_root):
    exp_num=re.match(r'exp(\d+)', os.path.basename(exp_root)).group(1)
    idx={}
    for loc in range(1,7):
        txt=os.path.join(exp_root,TRACK_TXT_PATTERN.format(exp=exp_num,loc=loc))
        if not os.path.isfile(txt): continue
        pid_frames={}
        for ln in open(txt,'r',encoding='utf-8'):
            g,obj,pid=map(int,ln.split()[:3])
            if obj!=0: continue
            pid_frames.setdefault(pid,[]).append(g)
        for pid,frames in pid_frames.items():
            for ord_,g in enumerate(sorted(frames)):
                idx[(loc,pid,ord_)]=g
    return idx

def path_label(path):
    parts=path.split(os.sep)
    try:
        i=parts.index("crops")
        if parts[i-1].endswith("_annotation") and parts[i+1]=="person":
            return PERSON_LABEL_MAP.get(parts[i+2],0)
    except ValueError: pass
    return 0

# ────────── Data loader ──────────
class ImgDS(Dataset):
    def __init__(self,paths,tfm): self.paths,self.tfm=paths,tfm
    def __len__(self): return len(self.paths)
    def __getitem__(self,i):
        return self.tfm(Image.open(self.paths[i]).convert("RGB")), self.paths[i]
def collate(b): return default_collate(b)

class ExpLoader:
    """ラベル 1‑6 は最多枚数トラックを gallery、そのほか query。"""
    def __init__(self,base,exp,min_len=1):
        self.root=os.path.join(base,exp)
        self.anno=os.path.join(self.root,f"{exp}_annotation","crops","person")
        self.min=min_len
    def _anno_tracks(self):
        gal,qry=[],[]
        if not os.path.isdir(self.anno): return gal,qry
        for pdir in os.listdir(self.anno):
            p=os.path.join(self.anno,pdir)
            if not os.path.isdir(p): continue
            tracks=[]
            for tk in os.listdir(p):
                tkdir=os.path.join(p,tk)
                if not os.path.isdir(tkdir): continue
                imgs=[os.path.join(tkdir,f) for f in os.listdir(tkdir)
                      if f.lower().endswith(IMG_EXTS)]
                if len(imgs)>=self.min: tracks.append(imgs)
            if not tracks: continue
            # 最も枚数の多いトラックを gallery
            gal_track=max(tracks,key=len)
            gal.append(gal_track)
            for t in tracks:
                if t is not gal_track: qry.append(t)
        return gal,qry
    def _label0(self):
        qry=[]
        for d in os.listdir(self.root):
            if not d.startswith(f"{os.path.basename(self.root)}_point"): continue
            base=os.path.join(self.root,d,"crops","person",f"{d}_other")
            if not os.path.isdir(base): continue
            for tk in os.listdir(base):
                tkdir=os.path.join(base,tk)
                if not os.path.isdir(tkdir): continue
                imgs=[os.path.join(tkdir,f) for f in os.listdir(tkdir)
                      if f.lower().endswith(IMG_EXTS)]
                if len(imgs)>=self.min: qry.append(imgs)
        return qry
    def load(self):
        g,q=self._anno_tracks()
        return q+self._label0(), g
def letterbox_resize(rgb, out_wh=(256, 128)):
    """
    最大辺を out_wh に合わせリサイズし、余白を黒でパディングして返す。
      rgb : numpy H×W×3 (RGB)
      out_wh : (out_h, out_w) → OSNet は (256,128)
    """
    oh, ow = out_wh
    h, w   = rgb.shape[:2]
    scale  = min(oh / h, ow / w)      # 全体が収まる方向で拡大/縮小
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_top    = (oh - nh) // 2
    pad_bottom = oh - nh - pad_top
    pad_left   = (ow - nw) // 2
    pad_right  = ow - nw - pad_left

    mean_color = tuple(map(int, img.mean(axis=(0, 1))))  # (R, G, B)

    return cv2.copyMakeBorder(
        img, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(0,0,0)
    )
# ────────── Embedding ──────────
def embed_paths(paths, extractor, device, cache, idx):
    key=tuple(paths)
    if key in cache: return cache[key]
    for p in paths:
        if p in META: continue
        loc,ord_=parse_loc_ord(os.path.basename(p))
        pid=int(os.path.basename(os.path.dirname(p)))
        t=idx.get((loc,pid,ord_))
        if t is None: MISSING_TIME.append(p)
        META[p]=dict(loc=loc,ordinal=ord_,time=t,pid=pid,
                     label=path_label(p),pred=None)

    # 新:
    def to_letterboxed_tensor(pil_img):
        rgb = np.array(pil_img)  # PIL → numpy RGB
        rgb = letterbox_resize(rgb, (256, 128))
        return transforms.ToTensor()(rgb)  # 0–1 float C×H×W

    tfm = transforms.Compose([
        transforms.Lambda(to_letterboxed_tensor),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # tfm=transforms.Compose([
    #     transforms.Resize((256,128)), transforms.ToTensor(),
    #     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    loader=DataLoader(ImgDS(paths,tfm),64,False,num_workers=0,collate_fn=collate)
    with torch.no_grad():
        feats=np.vstack([extractor(imgs.to(device)).cpu().numpy()
                         for imgs,_ in loader])
    cache[key]=feats
    return feats

# ────────── 汎用距離関数 ──────────
def _norm(v): return np.linalg.norm(v,axis=1,keepdims=True)
def d_mean(Q,G):  return 1-cosine_similarity(Q.mean(0,keepdims=True),
                                             G.mean(0,keepdims=True))[0,0]
def d_pair(Q,G):  return (1-cosine_similarity(Q,G)).mean()
def d_pair_w(Q,G):D=1-cosine_similarity(Q,G); W=_norm(Q)@_norm(G).T; W/=W.sum(); return (D*W).sum()
def d_emd(Q,G):   C=1-cosine_similarity(G,Q); C=np.maximum(C,0); a=np.ones(G.shape[0])/G.shape[0]; b=np.ones(Q.shape[0])/Q.shape[0]; return ot.emd2(a,b,C)
# def d_emd_w(Q,G): C=1-cosine_similarity(G,Q); C=np.maximum(C,0); a=_norm(G).ravel(); a/=a.sum(); b=_norm(Q).ravel(); b/=b.sum(); return ot.emd2(a,b,C)

# DIST = dict(mean=d_mean, pair=d_pair, pair_w=d_pair_w, emd=d_emd, emd_w=d_emd_w)
DIST = dict(mean=d_mean, pair=d_pair, pair_w=d_pair_w, emd=d_emd)

# ────────── F1 用 ──────────
def f1_metric(th, dist_fn, q_tks, g_tks, ext, dev, cache, idx):
    g_embs=[embed_paths(t,ext,dev,cache,idx) for t in g_tks]
    g_lbls=[META[t[0]]['label'] for t in g_tks]
    y_true,y_pred=[],[]
    for tk in q_tks:
        Q=embed_paths(tk,ext,dev,cache,idx)
        d=[dist_fn(Q,G) for G in g_embs]
        pred=g_lbls[int(np.argmin(d))] if min(d)<=th else 0
        y_true.append(META[tk[0]]['label']); y_pred.append(pred)
    return f1_score(y_true,y_pred,average='macro',zero_division=0)

def sweep(loader, ext, dev):
    idx=build_track_idx(loader.root); cache={}
    best={}
    for name,dist in DIST.items():
        top_f,top_th=-1,None
        for th in TH_RANGE:
            f1=f1_metric(th,dist,*loader.load(),ext,dev,cache,idx)
            if f1>top_f: top_f,top_th=f1,th
        best[name]=(top_th,top_f)
    return best

# ────────── 分類 (任意距離) ──────────
def classify(tracks, gallery_embs, gallery_lbls, dist_fn, th, ext, dev, cache, idx):
    for tk in tracks:
        Q=embed_paths(tk,ext,dev,cache,idx)
        d=[dist_fn(Q,G) for G in gallery_embs]
        pred=gallery_lbls[int(np.argmin(d))] if min(d)<=th else 0
        for p in tk: META[p]['pred']=pred

# ────────── OD 作成 & CSV 保存 ──────────
def to_csv(mat, path):
    with open(path,'w',newline='') as f:
        writer=csv.writer(f)
        writer.writerows(mat)

def od_table(tracks, key, label_filter=None):
    od=np.zeros((6,6),int)
    reps=[]
    for tk in tracks:
        info=[META[p] for p in tk if META[p]['time'] is not None]
        if not info: continue
        lbl=info[0][key]
        if lbl==0: continue
        if label_filter and lbl!=label_filter: continue
        reps.append((lbl, min(i['time'] for i in info), META[tk[0]]['loc']))
    from collections import defaultdict
    seq=defaultdict(list)
    for lbl,t,loc in reps:
        if 1<=loc<=6: seq[lbl].append((t,loc))
    for lst in seq.values():
        lst.sort()
        for (_,prev),(_,curr) in zip(lst,lst[1:]):
            if prev!=curr: od[prev-1,curr-1]+=1
    return od

# ────────── 1 EXP × 1 DIST を評価 ──────────
def evaluate_exp(exp, dist_name, th, ext, dev):
    print(f"\n===== {exp}  |  {dist_name} (th={th:.2f}) =====")
    loader=ExpLoader(BASE_DIR,exp)
    q_tks,g_tks=loader.load()
    idx=build_track_idx(loader.root); cache={}
    # gallery
    g_embs=[embed_paths(t,ext,dev,cache,idx) for t in g_tks]
    g_lbls=[META[t[0]]['label'] for t in g_tks]
    # classify
    classify(q_tks+g_tks,g_embs,g_lbls,DIST[dist_name],th,ext,dev,cache,idx)
    tracks=q_tks+g_tks

    # ----- OD GT & PRED -----
    od_gt   = od_table(tracks,'label')
    od_pred = od_table(tracks,'pred')

    np.set_printoptions(linewidth=120)
    print("GT  (all)");   print(od_gt)
    print("PRED(all)");   print(od_pred)

    # ---- CSV 保存 ----
    base=f"{exp}_{dist_name}"
    to_csv(od_gt,   os.path.join(OUT_DIR, f"{base}_gt_all.csv"))
    to_csv(od_pred, os.path.join(OUT_DIR, f"{base}_pred_all.csv"))
    for lbl in range(1,7):
        to_csv(od_table(tracks,'label',lbl),
               os.path.join(OUT_DIR, f"{base}_gt_label{lbl}.csv"))
        to_csv(od_table(tracks,'pred',lbl),
               os.path.join(OUT_DIR, f"{base}_pred_label{lbl}.csv"))

    # ----- F1 (query setのみ) -----
    f1 = f1_metric(th,DIST[dist_name], q_tks, g_tks, ext, dev, cache, idx)
    print(f"Query‑set F1: {f1:.3%}")

def exp_stats(exp_name: str):
    """exp 内のトラック数・サンプル数をラベル別に集計"""
    loader = ExpLoader(BASE_DIR, exp_name)
    query_tks, gallery_tks = loader.load()
    tracks = query_tks + gallery_tks

    track_cnt   = {lbl: 0 for lbl in range(7)}   # 0=その他, 1-6=人ID
    sample_cnt  = {lbl: 0 for lbl in range(7)}

    for tk in tracks:
        lbl = path_label(tk[0])          # トラックのラベル（先頭画像で代表）
        track_cnt[lbl]  += 1
        sample_cnt[lbl] += len(tk)

    return track_cnt, sample_cnt


def print_stats(exp_name, track_cnt, sample_cnt):
    print(f"\n── {exp_name} ──")
    print(f"{'Label':>5} | {'Tracks':>6} | {'Samples':>7}")
    print("-" * 25)
    for lbl in range(7):
        print(f"{lbl:>5} | {track_cnt[lbl]:>6} | {sample_cnt[lbl]:>7}")
    total_tracks  = sum(track_cnt.values())
    total_samples = sum(sample_cnt.values())
    print("-" * 25)
    print(f"{'ALL':>5} | {total_tracks:>6} | {total_samples:>7}")

# ────────── main ──────────
if __name__=="__main__":
    device="cuda" if torch.cuda.is_available() else "cpu"
    extractor=FeatureExtractor(model_name=MODEL_NAME,
                               model_path=MODEL_PATH, device=device)

    # 1. sweep on exp1
    print("── Sweep (exp1) ──")
    best=sweep(ExpLoader(BASE_DIR,"exp1"), extractor, device)
    for n,(th,f1) in best.items():
        print(f"{n:<6s} best_th={th:.2f} F1={f1:.3%}")

    # 2. apply to exp1/2/3
    for exp in ("exp1","exp2","exp3"):
        for dist in DIST.keys():
            th=best[dist][0]
            evaluate_exp(exp, dist, th, extractor, device)

    if MISSING_TIME:
        print("\n[INFO] 未取得 global_frame:", len(MISSING_TIME))

    # 3. stats for exp1/2/3
    for exp in ("exp1", "exp2", "exp3"):
        t_cnt, s_cnt = exp_stats(exp)
        print_stats(exp, t_cnt, s_cnt)
