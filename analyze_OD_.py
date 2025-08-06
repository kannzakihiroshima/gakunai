#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OD 行列 (GT / Pred) の誤差分析 & 可視化

前提:
    既存スクリプトで生成した od_csv/
        exp1_mean_gt_all.csv
        exp1_mean_pred_all.csv
        ... （任意数）
を対象に、同名ペアごとに
    ・各種誤差指標を計算
    ・差分行列 CSV 出力
    ・ヒートマップ画像出力
最終的に summary_metrics.csv を作成する。
"""

import os, re, glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ────── 設定 ──────
OD_DIR   = "od_csv_knn_1_epoch10"          # 既存 CSV の保存先
OUT_DIR  = "od_analysis_learning_knn"     # 新規生成先
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------
# 1. ユーティリティ
# -------------------------------------------------
def load_mat(path) -> np.ndarray:
    """CSV → int64 行列 (6×6 固定想定)"""
    return np.loadtxt(path, delimiter=",", dtype=np.int64)

def save_csv(mat: np.ndarray, path: str):
    np.savetxt(path, mat, fmt="%d", delimiter=",")

def diff_heatmap(diff: np.ndarray, title: str, save_path: str):
    """PR - GT を青白赤ヒートマップで保存"""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(diff, cmap="bwr", vmin=-5, vmax=5)
    # セル数値を描画
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            ax.text(j, i, int(diff[i, j]), ha="center", va="center", color="black")
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xlabel("Destination"); ax.set_ylabel("Origin")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Pred - GT")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def compute_metrics(gt: np.ndarray, pr: np.ndarray) -> dict:
    err  = pr - gt
    mae  = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    rel_mae = mae / (gt.mean() + 1e-9)      # 0 除回避
    trip_match = np.sum(np.minimum(gt, pr)) / (gt.sum() + 1e-9)
    # 相関（要フラット化、定数行列はナンセンスなので try/except）
    try:
        corr = np.corrcoef(gt.ravel(), pr.ravel())[0, 1]
    except Exception:
        corr = np.nan
    return dict(MAE=mae, RMSE=rmse, RelMAE=rel_mae,
                TripMatch=trip_match, Pearson=corr)

# -------------------------------------------------
# 2. メインループ
# -------------------------------------------------
records = []

for gt_path in glob.glob(os.path.join(OD_DIR, "*_gt_all.csv")):
    base = os.path.basename(gt_path)
    pred_path = gt_path.replace("_gt_all.csv", "_pred_all.csv")
    if not os.path.isfile(pred_path):
        print(f"[WARN] 対応する pred が見つかりません: {base}")
        continue

    tag = re.sub(r"_gt_all\.csv$", "", base)   # 例: exp1_mean
    print(f"[INFO] {tag}")

    gt  = load_mat(gt_path)
    pr  = load_mat(pred_path)
    diff = pr - gt
    abs_diff = np.abs(diff)

    # 2-1. 指標計算
    metrics = compute_metrics(gt, pr)
    metrics.update(tag=tag)
    records.append(metrics)

    # 2-2. 差分行列 CSV
    save_csv(diff,     os.path.join(OUT_DIR, f"{tag}_diff.csv"))
    save_csv(abs_diff, os.path.join(OUT_DIR, f"{tag}_abs_diff.csv"))

    # 2-3. ヒートマップ PNG
    diff_heatmap(diff,
                 title=f"{tag}: Pred - GT",
                 save_path=os.path.join(OUT_DIR, f"{tag}_diff_heat.png"))

# -------------------------------------------------
# 3. サマリ保存
# -------------------------------------------------
if records:
    df = pd.DataFrame(records)
    df = df[["tag", "MAE", "RMSE", "RelMAE", "TripMatch", "Pearson"]]
    df.to_csv(os.path.join(OUT_DIR, "summary_metrics.csv"), index=False)
    print(f"[DONE] 結果を {OUT_DIR}/ に保存しました")
else:
    print("[INFO] 対象ファイルが見つかりませんでした")
