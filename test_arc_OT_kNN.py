
# -*- coding: utf-8 -*-
"""
ArcFace + k-NN（tracklet多数決）で人物ラベル 0–6 を推定
────────────────────────────────────────────────────────────
主な変更:
  • tid を 物理パス全体から生成し一意性を保証
      tid = os.path.relpath(os.path.dirname(tk[0]), BASE_DIR).replace(os.sep, '_')
"""

import os, re, csv, random
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from pytorch_metric_learning.losses import ArcFaceLoss
from torchreid.reid.utils import FeatureExtractor

# ───────── Configuration ─────────
BASE_DIR   = r"C:/Users/sugie/PycharmProjects/StrongSORT-YOLO/clear_data_0.5_green_raw"
OUT_DIR    = "od_csv_knn_1_epoch10"
MODEL_PATH = (r"C:/Users/sugie/PycharmProjects/pythonProject10/MARS/"
              "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter (1).pth")
IMG_EXTS   = (".jpg",".jpeg",".png",".bmp",".tif",".tiff")

EPOCHS       = 10
BATCH_SIZE   = 64
LR_HEAD      = 1e-4
RATIO_LABEL0 = 5

random.seed(2025)
np.random.seed(2025)
torch.manual_seed(2025)

# ───────── Transforms ─────────
def make_transform(train: bool):
    aug = [
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.RandomResizedCrop((256,128), scale=(0.8,1.0), ratio=(0.75,1.33)),
    ]
    basic = [
        transforms.Resize((256,128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ]
    return transforms.Compose(aug+basic if train else basic)

TFM_TRAIN = make_transform(True)
TFM_TEST  = make_transform(False)

# ───────── Utility ─────────
PERSON_LABEL_MAP = {
    "ichimanda":1, "ichiraku":2, "idogawa":3,
    "miyazaki":4,  "noguchi":5,  "sakai":6
}
def path_label(p:str)->int:
    parts=p.split(os.sep)
    try:
        i=parts.index("crops")
        if parts[i-1].endswith("_annotation") and parts[i+1]=="person":
            return PERSON_LABEL_MAP.get(parts[i+2],0)
    except ValueError:
        pass
    return 0

def parse_loc_ord(fname:str):
    m=re.search(r'point(\d+)',fname)
    if not m: return None,None
    d=m.group(1)
    return int(d[0]), int(d[1:] or 0)

def build_track_idx(exp_root:str):
    TRACK_TXT_PATTERN="exp{exp}_point{loc}.txt"
    exp_num=re.match(r'exp(\d+)',os.path.basename(exp_root)).group(1)
    idx={}
    for loc in range(1,7):
        txt=os.path.join(exp_root,
                         TRACK_TXT_PATTERN.format(exp=exp_num,loc=loc))
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

# ───────── Dataset ─────────
class ImgDS(Dataset):
    def __init__(self,paths,labels,tfm):
        self.paths,self.labels,self.tfm=paths,labels,tfm
    def __len__(self): return len(self.paths)
    def __getitem__(self,i):
        return self.tfm(Image.open(self.paths[i]).convert("RGB")), self.labels[i]

def collate(batch):
    imgs,lbls=zip(*batch)
    return torch.stack(imgs,0), torch.tensor(lbls)

# ───────── Net128_osnet head ─────────
class Net128_osnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(512,400)
        self.fc2=nn.Linear(400,300)
        self.bn1=nn.BatchNorm1d(300)
        self.fc3=nn.Linear(300,200)
        self.drop=nn.Dropout(0.2)
        self.fc4=nn.Linear(200,128)
    def forward(self,x):
        x=F.relu(self.fc1(x)); x=F.relu(self.fc2(x))
        x=self.bn1(x); x=F.relu(self.fc3(x)); x=self.drop(x)
        x=F.relu(self.fc4(x))
        return F.normalize(x,p=2,dim=1)

# ───────── ExpLoader ─────────
class ExpLoader:
    def __init__(self,base,exp,min_len=1):
        self.root=os.path.join(base,exp)
        self.anno=os.path.join(self.root,f"{exp}_annotation","crops","person")
        self.minlen=min_len
    def _anno_tracks(self):
        gal,qry=[],[]
        if not os.path.isdir(self.anno): return gal,qry
        for pid in os.listdir(self.anno):
            pdir=os.path.join(self.anno,pid)
            if not os.path.isdir(pdir): continue
            tracks=[]
            for tk in os.listdir(pdir):
                tdir=os.path.join(pdir,tk)
                if not os.path.isdir(tdir): continue
                imgs=[os.path.join(tdir,f) for f in os.listdir(tdir)
                      if f.lower().endswith(IMG_EXTS)]
                if len(imgs)>=self.minlen: tracks.append(imgs)
            if not tracks: continue
            gal_track=max(tracks,key=len)
            gal.append(gal_track)
            qry.extend(t for t in tracks if t is not gal_track)
        return gal,qry
    def _label0_tracks(self):
        tracks=[]
        for d in os.listdir(self.root):
            if not d.startswith(os.path.basename(self.root)+"_point"): continue
            base=os.path.join(self.root,d,"crops","person",f"{d}_other")
            if not os.path.isdir(base): continue
            for tk in os.listdir(base):
                tdir=os.path.join(base,tk)
                if not os.path.isdir(tdir): continue
                imgs=[os.path.join(tdir,f) for f in os.listdir(tdir)
                      if f.lower().endswith(IMG_EXTS)]
                if len(imgs)>=self.minlen: tracks.append(imgs)
        return tracks
    def gallery_tracks(self):
        gal,_=self._anno_tracks(); return gal
    def all_tracks(self):
        gal,qry=self._anno_tracks(); label0=self._label0_tracks()
        return gal+qry+label0


# ───────── WeightedArcFaceLoss ─────────
class WeightedArcFaceLoss(ArcFaceLoss):
    def __init__(self,num_classes,embedding_size,class_weights=None,
                 margin=28.6,scale=64,**kw):
        super().__init__(num_classes=num_classes,
                         embedding_size=embedding_size,
                         margin=margin,scale=scale,**kw)
        if class_weights is not None and not isinstance(class_weights,torch.Tensor):
            class_weights=torch.tensor(class_weights,dtype=torch.float32)
        self.cross_entropy=nn.CrossEntropyLoss(weight=class_weights,
                                               reduction="none")

# ───── FeatureExtractor ─────
def get_feature_extractor(device):
    fe=FeatureExtractor(model_name="osnet_ain_x1_0",
                        model_path=MODEL_PATH,device=device,verbose=False)
    fe.model.eval()
    for p in fe.model.parameters(): p.requires_grad_(False)
    return fe

# ───── build TrainLoader ─────────
def build_train_loader(exp,other_exps):
    gal_tracks=ExpLoader(BASE_DIR,exp).gallery_tracks()
    g_paths=[p for tk in gal_tracks for p in tk]
    g_labels=[path_label(p) for p in g_paths]

    n0=int(len(g_paths)*RATIO_LABEL0)
    pool=[]
    for e in other_exps: pool+=ExpLoader(BASE_DIR,e)._label0_tracks()
    pool_flat=[img for tk in pool for img in tk]
    random.shuffle(pool_flat)
    label0=pool_flat[:n0]

    paths  = g_paths + label0
    labels = g_labels + [0]*len(label0)

    cnt=Counter(labels)
    print("  [Train samples] "+" ".join(f"label{l}:{cnt[l]}" for l in sorted(cnt)))
    ds=ImgDS(paths,labels,TFM_TRAIN)
    return DataLoader(ds,BATCH_SIZE,shuffle=True,drop_last=True,
                      num_workers=0,collate_fn=collate), cnt

# ───── train head + ArcFace ─────
def train_arcface(exp,device,fe):
    others=[e for e in ("exp1","exp2","exp3") if e!=exp]
    loader,cnt=build_train_loader(exp,others)
    num_classes=max(cnt.keys())+1
    cw=np.zeros(num_classes,dtype=np.float32)
    for l,c in cnt.items(): cw[l]=1.0/(c+1e-6)

    head=Net128_osnet().to(device)
    loss_fn=WeightedArcFaceLoss(num_classes=num_classes,embedding_size=128,
                                class_weights=cw).to(device)
    opt=optim.Adam(list(head.parameters())+list(loss_fn.parameters()), lr=LR_HEAD)

    fe.model.eval()
    for ep in range(1,EPOCHS+1):
        losses=[]
        for imgs,lbls in loader:
            imgs,lbls=imgs.to(device),lbls.to(device)
            with torch.no_grad(): feat512=fe.model(imgs)
            emb=head(feat512)
            loss=loss_fn(emb,lbls).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"    Epoch {ep:02d} loss={np.mean(losses):.4f}")
    head.eval(); loss_fn.eval()
    return head

# ───── embed & register ─────
META, MISSING_TIME = {}, []
def embed_and_register(fe,head,tk,idx):
    ds=ImgDS(tk,[0]*len(tk),TFM_TEST)
    loader=DataLoader(ds,BATCH_SIZE,False,num_workers=0,
                      collate_fn=lambda b: torch.stack([x for x,_ in b],0))
    embs=[]
    with torch.no_grad():
        for imgs in loader:
            embs.append(head(fe.model(imgs.to(next(head.parameters()).device))).cpu())
    emb=torch.cat(embs,0)

    fname=os.path.basename(tk[0]); loc,ord_=parse_loc_ord(fname)
    pid=int(os.path.basename(os.path.dirname(tk[0])))
    time=idx.get((loc,pid,ord_),0)
    if time==0: MISSING_TIME.append(tk[0])

    META[tk[0]]={'loc':loc,'time':time,'label':path_label(tk[0])}
    return emb

# ───── downsample for k-NN ─────
def downsample_label0_indices(labels,max_ratio=5,seed=2025):
    np.random.seed(seed)
    idx0     = np.where(labels==0)[0]
    idx_non0 = np.where(labels!=0)[0]
    max_other=np.bincount(labels[idx_non0]).max()
    n0_allowed=int(max_other*max_ratio)
    if len(idx0)>n0_allowed:
        idx0=np.random.choice(idx0,n0_allowed,replace=False)
    return np.concatenate([idx0,idx_non0])

# ─────── OD table ───────
def od_matrix(tracks,key):
    od=np.zeros((6,6),int)
    seq=defaultdict(list)
    for t in tracks:
        lbl,loc,time=t[key],t['loc'],t['time']
        if lbl==0 or loc==0 or time==0: continue
        seq[lbl].append((time,loc))
    for lst in seq.values():
        lst.sort()
        for (_,p),(_,c) in zip(lst,lst[1:]):
            if p!=c: od[p-1,c-1]+=1
    return od

def to_csv(mat,path):
    with open(path,'w',newline='') as f:
        csv.writer(f).writerows(mat)

# ─────────────── Main ──────────────
if __name__=="__main__":
    os.makedirs(OUT_DIR,exist_ok=True)
    device="cuda" if torch.cuda.is_available() else "cpu"
    fe=get_feature_extractor(device)

    for exp in ("exp1","exp2","exp3"):
        print(f"\n===== {exp} =====")
        head=train_arcface(exp,device,fe)

        loader=ExpLoader(BASE_DIR,exp)
        idx=build_track_idx(loader.root)
        gal_tracks=loader.gallery_tracks()
        all_tracks=loader.all_tracks()

        # embed gallery (初回呼出で META 更新)
        for tk in gal_tracks: _=embed_and_register(fe,head,tk,idx)

        # ─── collect embeddings ───
        embs, labels, tids, metas = [],[],[],[]
        for tk in all_tracks:
            emb=embed_and_register(fe,head,tk,idx)
            lbl=META[tk[0]]['label']
            # --- UNIQUE tid ---------------------------------------------------
            tid=os.path.relpath(os.path.dirname(tk[0]), BASE_DIR) \
                        .replace(os.sep,'_')           ### CHANGED ###
            # ------------------------------------------------------------------
            for vec in emb.numpy():
                embs.append(vec); labels.append(lbl); tids.append(tid)
            metas += [{'tracklet_id':tid,'loc':META[tk[0]]['loc'],
                       'time':META[tk[0]]['time'],'label':lbl}] * emb.shape[0]

        arr_emb=np.stack(embs,0); labels_arr=np.array(labels)

        knn_idx=downsample_label0_indices(labels_arr,max_ratio=5,seed=2025)
        X_knn,y_knn = arr_emb[knn_idx],labels_arr[knn_idx]

        knn=KNeighborsClassifier(n_neighbors=5,metric="cosine")
        knn.fit(X_knn,y_knn)

        preds=knn.predict(arr_emb)

        # ─── majority vote per (unique) tracklet ───
        track_preds,y_true,y_pred=[],[],[]
        grouping=defaultdict(list)
        for i,tid in enumerate(tids): grouping[tid].append(i)

        for tid,idxs in grouping.items():
            sub_pred=preds[idxs]
            cnt_sub=Counter(sub_pred)
            maxf=max(cnt_sub.values())
            maj=min(l for l,f in cnt_sub.items() if f==maxf)
            meta0=metas[idxs[0]]
            track_preds.append({'tid':tid,'loc':meta0['loc'],
                                'time':meta0['time'],
                                'label':meta0['label'],'pred':maj})
            y_true.append(meta0['label']); y_pred.append(maj)

        # ─── evaluation / save ───
        f1=f1_score(y_true,y_pred,average='macro',zero_division=0)
        print(f"  Macro-F1: {f1:.3%}")

        od_gt=od_matrix(track_preds,'label')
        od_pr=od_matrix(track_preds,'pred')
        to_csv(od_gt,os.path.join(OUT_DIR,f"{exp}_gt_all.csv"))
        to_csv(od_pr,os.path.join(OUT_DIR,f"{exp}_pred_all.csv"))
        for l in range(1,7):
            to_csv(od_matrix([t for t in track_preds if t['label']==l],'label'),
                   os.path.join(OUT_DIR,f"{exp}_gt_label{l}.csv"))
            to_csv(od_matrix([t for t in track_preds if t['label']==l],'pred'),
                   os.path.join(OUT_DIR,f"{exp}_pred_label{l}.csv"))

        meta_file=os.path.join(OUT_DIR,f"{exp}_tracklets_meta.csv")
        with open(meta_file,'w',newline='') as f:
            w=csv.writer(f)
            w.writerow(['tracklet_id','loc','time','true_label','pred_label'])
            for t in track_preds:
                w.writerow([t['tid'],t['loc'],t['time'],t['label'],t['pred']])
        print(f"  saved tracklets meta -> {meta_file}")

    if MISSING_TIME:
        print(f"\n[WARN] Missing time entries: {len(MISSING_TIME)}")
    print("\nDone, output in", OUT_DIR)
