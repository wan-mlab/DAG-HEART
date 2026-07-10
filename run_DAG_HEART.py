#!/usr/bin/env python

import os
import sys
import argparse
from pathlib import Path

base_dir = Path(f"/home/.theano/{os.getpid()}")
base_dir.mkdir(parents=True, exist_ok=True)

prev = os.environ.get("THEANO_FLAGS", "").strip()
addon = f"base_compiledir={base_dir}"
os.environ["THEANO_FLAGS"] = prev + ("," if prev else "") + addon

SEED = 35
HIDDENLAYERS = [100]
PATIENCE = 10
ALPHA = 0.1
DECAY = 0.0
POS_NEG = 1.0

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run BRCA DAG + MKL + KPCA + CCSA/TLDA experiment."
    )

    parser.add_argument("--input-dir", type=str, default=os.path.join(str(Path.home()), "Inputs"))
    parser.add_argument("--output-root", type=str, default=os.path.join(str(Path.home()), "DAG_HEART"))

    parser.add_argument(
        "--endpoint",
        type=str,
        default="PFI",
        choices=["OS", "DSS", "DFI", "PFI"],
    )

    parser.add_argument("--years", type=int, default=4)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--n-repeats", type=int, default=10)

    parser.add_argument(
        "--lr-list",
        type=float,
        nargs="+",
        default=[1e-3, 3e-3, 1e-2],
    )

    parser.add_argument(
        "--dr-list",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5],
    )

    parser.add_argument("--seed-base", type=int, default=45)
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[100])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--decay", type=float, default=0.0)
    parser.add_argument("--pos-neg", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)

    return parser.parse_args()

ARGS = parse_args()

input_path = Path(ARGS.input_dir).resolve()
folder_path = Path(ARGS.output_root).resolve()
cf_dir_path = ARGS.endpoint
endpoint = ARGS.endpoint
ynum = int(ARGS.years)

folds = int(ARGS.folds)
n_repeats = int(ARGS.n_repeats)
lr_list = list(ARGS.lr_list)
dr_list = list(ARGS.dr_list)

SEED = int(ARGS.seed_base)
HIDDENLAYERS = [int(x) for x in ARGS.hidden_layers]
PATIENCE = int(ARGS.patience)
ALPHA = float(ARGS.alpha)
DECAY = float(ARGS.decay)
POS_NEG = float(ARGS.pos_neg)

if input_path.exists() and any(p.suffix == ".py" for p in input_path.glob("*.py")):
    sys.path.insert(0, str(input_path))
else:
    print(f"[WARN] No .py modules found in {input_path}; sys.path insert skipped.")

year_path = f"{ynum}yr"

out_dir = folder_path / year_path / cf_dir_path
out_dir.mkdir(parents=True, exist_ok=True)
os.chdir(out_dir)

fig_root = out_dir / "figures"
fig_root.mkdir(parents=True, exist_ok=True)

FIG = {
    "fig7": fig_root / "Figure7",
    "fig_kpca": fig_root / "Figure_KPCA",
    "fig_notears": fig_root / "Figure_NOTEARS",
    "fig_pairs": fig_root / "Figure_CCSA_pairs",
    "fig_scores": fig_root / "Figure_Scores_KM",
    "fig_mkl": fig_root / "Figure_MKL",
}
for p in FIG.values():
    p.mkdir(parents=True, exist_ok=True)

print(f"Working dir: {out_dir}")
print(f"Custom input directory: {input_path}")
print(f"Endpoint: {endpoint}")
print(f"Years: {ynum}")
print(f"Folds: {folds}")
print(f"Repeats: {n_repeats}")
print(f"Learning rates: {lr_list}")
print(f"Dropout rates: {dr_list}")
print(f"Seed base: {SEED}")
print(f"Hidden layers: {HIDDENLAYERS}")
print(f"Alpha: {ALPHA}")
print(f"Decay: {DECAY}")
print(f"Pos/neg ratio: {POS_NEG}")
print(f"Patience: {PATIENCE}")

import random
import time
import inspect
import glob
import pickle
from collections import OrderedDict

import numpy as np
import pandas as pd
import tensorflow as tf

from keras import Input, Model
from keras.layers import Activation, Dense, Dropout, Lambda

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    VarianceThreshold,
    SelectFromModel,
    mutual_info_classif,
)
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    pairwise_distances,
    silhouette_score,
)
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.calibration import calibration_curve

from scipy import optimize
from scipy.linalg import expm
from scipy.special import logit, expit

import Initialization_kernel

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    from lifelines import CoxPHFitter
    HAVE_LIFELINES = True
except Exception:
    HAVE_LIFELINES = False
    print("[WARN] lifelines is not installed. Cox summary will be skipped.")

try:
    from notears import linear as notears_linear
    HAVE_NOTEARS = True
except Exception:
    HAVE_NOTEARS = False
    print("NOTEARS is not installed.")

devices = tf.config.experimental.list_physical_devices("GPU")
if devices:
    print("GPU is detected.")
    tf.config.experimental.set_memory_growth(devices[0], True)
    print("GPU dynamic memory allocation is activated.")
else:
    print("GPU can't used for it.")

def concordance_index_simple(T, risk, E):
    T = np.asarray(T, dtype=float).reshape(-1)
    risk = np.asarray(risk, dtype=float).reshape(-1)
    E = np.asarray(E, dtype=int).reshape(-1)

    n = len(T)
    assert len(risk) == n and len(E) == n

    concordant = 0.0
    ties = 0.0
    comparable = 0.0

    for i in range(n):
        if E[i] != 1:
            continue
        for j in range(n):
            if T[i] < T[j]:
                comparable += 1.0
                if risk[i] > risk[j]:
                    concordant += 1.0
                elif risk[i] == risk[j]:
                    ties += 1.0

    if comparable == 0:
        return np.nan
    return (concordant + 0.5 * ties) / comparable

def _ensure_block_order(X_blocks, blocks_order=None):
    if blocks_order is not None:
        return OrderedDict([(k, X_blocks[k]) for k in blocks_order])
    return X_blocks if isinstance(X_blocks, OrderedDict) else OrderedDict(X_blocks)

def _zscore_blocks_fit(X_blocks):
    scalers = {k: StandardScaler(with_mean=True, with_std=True).fit(X) for k, X in X_blocks.items()}
    Xs = {k: scalers[k].transform(X_blocks[k]) for k in X_blocks}
    return Xs, scalers

def _zscore_blocks_transform(X_blocks, scalers):
    return {k: scalers[k].transform(X_blocks[k]) for k in X_blocks}

def _pca_blocks_fit(Xs, pca_dims=None, random_state=0):
    if not isinstance(Xs, (dict, OrderedDict)) or len(Xs) == 0:
        raise ValueError("Xs must be a non-empty dict/OrderedDict of blocks.")

    pcas, Zs, starts, q_dims = {}, {}, {}, {}
    q_total = 0
    Z_cols = []
    n_ref = None

    for k, X in Xs.items():
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Block {k} must be 2D array, got shape {X.shape}")
        n, p = X.shape
        if n_ref is None:
            n_ref = n
        elif n != n_ref:
            raise ValueError(f"[Sample mismatch] block={k} has n={n} but expected n={n_ref}")

        if pca_dims is None:
            pcas[k] = None
            Z = X
        else:
            qk = int(pca_dims.get(k, p))
            qk = max(1, min(qk, p, max(1, n - 1)))

            solver = "randomized" if qk >= 2 else "auto"
            pc = PCA(n_components=qk, svd_solver=solver, random_state=random_state)
            Z = pc.fit_transform(X)
            pcas[k] = pc

        Zs[k] = Z
        starts[k] = q_total
        q_dims[k] = Z.shape[1]
        q_total += Z.shape[1]
        Z_cols.append(Z)

    Z_all = np.concatenate(Z_cols, axis=1)
    return Zs, pcas, starts, Z_all, q_dims

def _pca_blocks_transform(Xs, pcas, starts, q_dims):
    if set(Xs.keys()) != set(pcas.keys()):
        raise ValueError(f"[PCA keys mismatch] Xs={list(Xs.keys())}, pcas={list(pcas.keys())}")

    Zs = {}
    Z_cols = []
    n_ref = None

    for k, X in Xs.items():
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Block {k} must be 2D array, got shape {X.shape}")
        n = X.shape[0]
        if n_ref is None:
            n_ref = n
        elif n != n_ref:
            raise ValueError(f"[Sample mismatch] block={k} has n={n} but expected n={n_ref}")

        pc = pcas.get(k, None)
        Z = X if pc is None else pc.transform(X)

        if k not in q_dims:
            raise KeyError(f"q_dims missing key: {k}")
        if Z.shape[1] != q_dims[k]:
            raise ValueError(f"[PCA dim mismatch] block={k}, got {Z.shape[1]} expected {q_dims[k]}")

        Zs[k] = Z
        Z_cols.append(Z)

    Z_all = np.concatenate(Z_cols, axis=1)
    return Zs, Z_all

def _threshold_weights(W, w_threshold):
    if w_threshold is None:
        return W
    W = W.copy()
    W[np.abs(W) < float(w_threshold)] = 0.0
    return W

def _build_latent_mask_from_A_block(starts, q_dims, blocks_order, A_block):
    if A_block is None:
        raise ValueError("A_block is None; mask build should not be called.")

    if blocks_order is None or len(blocks_order) == 0:
        raise ValueError("blocks_order must be a non-empty list.")

    missing = [b for b in blocks_order if (b not in starts or b not in q_dims)]
    if missing:
        raise KeyError(f"blocks_order contains blocks missing in starts/q_dims: {missing}")

    A = np.asarray(A_block, dtype=int)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A_block must be square 2D, got {A.shape}")
    if len(blocks_order) != A.shape[0]:
        raise ValueError("blocks_order length must match A_block shape.")

    q_total = int(sum(int(q_dims[b]) for b in blocks_order))
    M = np.zeros((q_total, q_total), dtype=np.float64)

    for i, s in enumerate(blocks_order):
        s_slice = slice(starts[s], starts[s] + q_dims[s])
        for j, t in enumerate(blocks_order):
            t_slice = slice(starts[t], starts[t] + q_dims[t])
            if A[i, j] != 0:
                M[s_slice, t_slice] = 1.0

    np.fill_diagonal(M, 0.0)
    return M

def _notears_linear_masked_l2(
    Z, W_mask,
    lambda1=0.1, max_iter=100, h_tol=1e-8, rho_max=1e16,
    w_threshold=None, random_state=0,
):
    Z = np.asarray(Z, dtype=np.float64)
    n, d = Z.shape

    M = np.asarray(W_mask, dtype=np.float64)
    if M.shape != (d, d):
        raise ValueError(f"W_mask must be {(d,d)}, got {M.shape}")
    M = M.copy()
    np.fill_diagonal(M, 0.0)

    rng = np.random.default_rng(int(random_state))

    def _loss(W):
        R = Z - Z @ W
        return 0.5 / n * np.sum(R * R)

    def _grad_loss(W):
        R = Z - Z @ W
        return -(1.0 / n) * (Z.T @ R)

    def _h(W):
        E = expm(W * W)
        return float(np.trace(E) - d), E

    def _grad_h(W, E):
        return (E.T) * (2.0 * W)

    w_init = np.zeros(2 * d * d, dtype=np.float64)
    eps = 1e-3
    init_mat = rng.normal(0.0, eps, size=(d, d)) * M
    w_init[: d * d] = np.maximum(init_mat.reshape(-1), 0.0)
    w_init[d * d :] = np.maximum(-init_mat.reshape(-1), 0.0)

    bounds = [(0.0, None)] * (2 * d * d)
    rho, nu = 1.0, 0.0

    def _vec_to_W(w):
        w = np.asarray(w, dtype=np.float64)
        w_pos = w[: d * d].reshape(d, d)
        w_neg = w[d * d :].reshape(d, d)
        W_raw = w_pos - w_neg
        W = W_raw * M
        np.fill_diagonal(W, 0.0)
        return W, W_raw

    def _func(w):
        W, _ = _vec_to_W(w)
        loss = _loss(W)
        hval, E = _h(W)
        obj = loss + 0.5 * rho * (hval ** 2) + nu * hval + float(lambda1) * np.sum(w)

        G_loss = _grad_loss(W)
        G_h = _grad_h(W, E)
        G_W = G_loss + (rho * hval + nu) * G_h

        G_W_raw = G_W * M

        g_pos = G_W_raw.reshape(-1) + float(lambda1)
        g_neg = (-G_W_raw.reshape(-1)) + float(lambda1)
        grad = np.concatenate([g_pos, g_neg], axis=0)
        return obj, grad, hval

    w = w_init
    for _ in range(int(max_iter)):
        sol = optimize.minimize(
            fun=lambda vv: _func(vv)[0],
            x0=w,
            jac=lambda vv: _func(vv)[1],
            method="L-BFGS-B",
            bounds=bounds,
        )
        w = sol.x
        hval = _func(w)[2]

        if hval <= float(h_tol if h_tol is not None else 1e-8):
            break

        nu += rho * hval
        rho = min(rho * 10.0, float(rho_max if rho_max is not None else 1e16))

    W_est, _ = _vec_to_W(w)
    W_est = _threshold_weights(W_est, w_threshold)
    return W_est

def _call_notears_linear(
    Z_all,
    notears_lambda=0.1,
    max_iter=100,
    loss_type="l2",
    h_tol=None,
    rho_max=None,
    w_threshold=None,
    transpose_W=False,
    W_mask=None,
    random_state=0,
):
    Z_all = np.asarray(Z_all, dtype=np.float64, order="C")
    d = Z_all.shape[1]

    if W_mask is not None:
        W_mask = np.asarray(W_mask, dtype=np.float64)
        if W_mask.shape != (d, d):
            raise ValueError(f"W_mask must be {(d,d)} for latent dim d={d}, got {W_mask.shape}")
        W_mask = W_mask.copy()
        np.fill_diagonal(W_mask, 0.0)

    if HAVE_NOTEARS:
        try:
            kwargs = dict(lambda1=notears_lambda, loss_type=loss_type, max_iter=max_iter)
            if h_tol is not None:
                kwargs["h_tol"] = h_tol
            if rho_max is not None:
                kwargs["rho_max"] = rho_max
            if w_threshold is not None:
                kwargs["w_threshold"] = w_threshold

            if W_mask is None:
                Wz = notears_linear.notears_linear(Z_all, **kwargs)
                Wz = np.asarray(Wz, dtype=np.float64)
            else:
                Wz = None
                for mask_key in ["w_mask", "W_mask", "mask", "M"]:
                    try_kwargs = dict(kwargs)
                    try_kwargs[mask_key] = W_mask
                    try:
                        Wz = notears_linear.notears_linear(Z_all, **try_kwargs)
                        Wz = np.asarray(Wz, dtype=np.float64)
                        break
                    except TypeError:
                        pass
                if Wz is None:
                    raise TypeError("notears library does not accept a mask parameter.")
        except Exception:
            Wz = None
    else:
        Wz = None

    if Wz is None:
        if loss_type != "l2":
            raise RuntimeError("Fallback NOTEARS supports loss_type='l2' only.")
        if W_mask is None:
            W_mask = np.ones((d, d), dtype=np.float64)
            np.fill_diagonal(W_mask, 0.0)

        Wz = _notears_linear_masked_l2(
            Z_all,
            W_mask=W_mask,
            lambda1=notears_lambda,
            max_iter=max_iter,
            h_tol=h_tol if h_tol is not None else 1e-8,
            rho_max=rho_max if rho_max is not None else 1e16,
            w_threshold=w_threshold,
            random_state=random_state,
        )

    np.fill_diagonal(Wz, 0.0)
    if transpose_W:
        Wz = Wz.T

    Wz = _threshold_weights(Wz, w_threshold)

    if W_mask is not None:
        Wz = Wz * W_mask

    return Wz

def dag_feature_projection_blocks_fit(
    X_blocks,
    pca_dims=None,
    notears_lambda=0.1,
    max_iter=100,
    loss_type="l2",
    blocks_order=None,
    random_state=0,
    h_tol=None,
    rho_max=None,
    w_threshold=None,
    transpose_W=False,
    A_block=None,
):
    X_blocks = _ensure_block_order(X_blocks, blocks_order=blocks_order)

    bo = list(blocks_order) if blocks_order is not None else list(X_blocks.keys())

    if list(X_blocks.keys()) != bo:
        raise ValueError(f"[blocks_order mismatch] bo={bo}, X_blocks.keys()={list(X_blocks.keys())}")
    
    if A_block is not None:
        A = np.asarray(A_block, dtype=int)
        if A.shape != (len(bo), len(bo)):
            raise ValueError(f"[A_block shape mismatch] A_block.shape={A.shape}, expected={(len(bo),len(bo))}")

    Xs, scalers = _zscore_blocks_fit(X_blocks)

    Zs, pcas, starts, Z_all, q_dims = _pca_blocks_fit(Xs, pca_dims=pca_dims, random_state=random_state)

    W_mask = None
    if A_block is not None:
        W_mask = _build_latent_mask_from_A_block(starts, q_dims, bo, A_block)
    latent_mask = W_mask
    
    Wz = _call_notears_linear(
        Z_all,
        notears_lambda=notears_lambda,
        max_iter=max_iter,
        loss_type=loss_type,
        h_tol=h_tol,
        rho_max=rho_max,
        w_threshold=w_threshold,
        transpose_W=transpose_W,
        W_mask=W_mask,
        random_state=random_state,
    )

    fitted = dict(
        blocks=list(X_blocks.keys()),
        blocks_order=list(bo),
        pca_dims=pca_dims,
        notears_lambda=notears_lambda,
        max_iter=max_iter,
        loss_type=loss_type,
        random_state=random_state,
        scalers=scalers,
        pcas=pcas,
        starts=starts,
        q_dims=q_dims,
        Wz=Wz,
        latent_mask=latent_mask,
        transpose_W=transpose_W,
        w_threshold=w_threshold,
        A_block=(None if A_block is None else np.asarray(A_block, dtype=int)),
    )
    return fitted

def dag_feature_projection_blocks_transform(
    X_blocks,
    fitted,
    A_block=None,
    blocks_order=None
):

    X_blocks = _ensure_block_order(X_blocks, blocks_order=blocks_order if blocks_order is not None else fitted["blocks"])

    expected_bo = list(fitted.get("blocks_order", fitted["blocks"]))
    bo = list(blocks_order) if blocks_order is not None else expected_bo
    
    if bo != expected_bo:
        raise ValueError(f"[blocks_order mismatch] transform bo={bo}, fitted blocks_order={expected_bo}")
    
    A_use = A_block if A_block is not None else fitted.get("A_block", None)
    if A_use is not None:
        A = np.asarray(A_use, dtype=int)
        if A.shape != (len(expected_bo), len(expected_bo)):
            raise ValueError(f"[A_block shape mismatch] A.shape={A.shape}, expected={(len(expected_bo),len(expected_bo))}")

    if list(X_blocks.keys()) != list(fitted["blocks"]):
        raise ValueError(f"[Block mismatch] got={list(X_blocks.keys())}, expected={list(fitted['blocks'])}")

    Xs = _zscore_blocks_transform(X_blocks, fitted["scalers"])

    Zs, Z_all = _pca_blocks_transform(Xs, fitted["pcas"], fitted["starts"], fitted["q_dims"])

    Wz = np.asarray(fitted["Wz"], dtype=np.float64)
    starts = fitted["starts"]
    q_dims = fitted["q_dims"]

    A_use = A_block if A_block is not None else fitted.get("A_block", None)
    if A_use is not None:
        W_mask = _build_latent_mask_from_A_block(starts, q_dims, expected_bo, A_use)
        Wz = Wz * W_mask

    block_W = {}
    for s in X_blocks:
        for t in X_blocks:
            s_slice = slice(starts[s], starts[s] + q_dims[s])
            t_slice = slice(starts[t], starts[t] + q_dims[t])
            block_W[(s, t)] = Wz[s_slice, t_slice]

    projs = {}
    for s in X_blocks:
        for t in X_blocks:
            Zhat_t_from_s = np.asarray(Zs[s]) @ np.asarray(block_W[(s, t)])
            if fitted["pcas"][t] is not None:
                Xhat_t_from_s = fitted["pcas"][t].inverse_transform(Zhat_t_from_s)
            else:
                Xhat_t_from_s = Zhat_t_from_s
            Xhat_t_from_s = fitted["scalers"][t].inverse_transform(Xhat_t_from_s)
            projs[(s, t)] = Xhat_t_from_s

    if A_block is not None:
        A = np.asarray(A_block, dtype=int)
        selected = {}
        for i, s in enumerate(expected_bo):
            for j, t in enumerate(expected_bo):
                if A[i, j] != 0:
                    selected[(s, t)] = projs[(s, t)]
        projs = selected

    return dict(projs=projs, Z_all=Z_all, Zs=Zs, block_W=block_W)

def _validate_A_block_and_proj_pairs(A_block, blocks_order, proj_pairs, strict=True):
    if A_block is None:
        raise ValueError("A_block is None. 방향성 prior를 쓰려면 A_block을 반드시 지정해야 합니다.")
    if blocks_order is None:
        raise ValueError("blocks_order is None. A_block 해석을 고정하려면 blocks_order가 필요합니다.")
    if proj_pairs is None or len(proj_pairs) == 0:
        raise ValueError("proj_pairs is empty. 최소 1개 이상의 projection pair가 필요합니다.")

    A = np.asarray(A_block, dtype=int)
    B = list(blocks_order)

    if A.shape != (len(B), len(B)):
        raise ValueError(f"A_block shape mismatch: got {A.shape}, expected {(len(B), len(B))}")

    Bset = set(B)
    bad = [(s,t) for (s,t) in proj_pairs if (s not in Bset or t not in Bset)]
    if bad:
        raise ValueError(f"proj_pairs contains unknown blocks vs blocks_order: {bad[:10]}")

    idx = {b:i for i,b in enumerate(B)}
    not_allowed = []
    for (s,t) in proj_pairs:
        if int(A[idx[s], idx[t]]) == 0:
            not_allowed.append((s,t))

    if not_allowed:
        msg = f"[A_block/proj_pairs mismatch] These proj_pairs are NOT allowed by A_block: {not_allowed}"
        if strict:
            raise ValueError(msg)
        else:
            print("[WARN]", msg)

    allowed = []
    for i,s in enumerate(B):
        for j,t in enumerate(B):
            if int(A[i,j]) != 0:
                allowed.append((s,t))
    unused_allowed = sorted(list(set(allowed) - set(proj_pairs)))
    if len(unused_allowed) > 0:
        print(f"[INFO] A_block allows {len(unused_allowed)} additional block-directions not used in proj_pairs. "
              f"Example: {unused_allowed[:8]}")

    return True


def _summarize_notears_block_stats(fitted, blocks_order=None, w_threshold=None):
    Wz = np.asarray(fitted["Wz"], dtype=np.float64)
    starts = fitted["starts"]
    q_dims = fitted["q_dims"]
    blocks = list(fitted["blocks"])
    order = list(blocks_order) if blocks_order is not None else blocks

    if set(order) != set(blocks):
        raise ValueError(f"blocks_order mismatch with fitted blocks. order={order}, fitted={blocks}")

    W = Wz.copy()
    if w_threshold is not None:
        W[np.abs(W) < float(w_threshold)] = 0.0

    rows = []
    for s in order:
        s_slice = slice(starts[s], starts[s] + q_dims[s])
        for t in order:
            t_slice = slice(starts[t], starts[t] + q_dims[t])
            Wst = W[s_slice, t_slice]

            absW = np.abs(Wst)
            nnz = int(np.count_nonzero(Wst))
            denom = int(Wst.size) if Wst.size > 0 else 1
            rows.append({
                "src_block": s,
                "tgt_block": t,
                "q_src": int(q_dims[s]),
                "q_tgt": int(q_dims[t]),
                "count_nonzero": nnz,
                "density": float(nnz / denom),
                "l1_sum": float(absW.sum()),
                "mean_abs": float(absW.mean()) if absW.size > 0 else 0.0,
                "max_abs": float(absW.max()) if absW.size > 0 else 0.0,
                "fro_norm": float(np.sqrt((Wst * Wst).sum())) if Wst.size > 0 else 0.0,
            })
    return pd.DataFrame(rows)


def _save_notears_fold_artifacts(fitted, outp, seed, fold_id, tag_suffix,
                                 blocks_order=None, A_block=None, w_threshold=None):
    outp = str(outp)
    os.makedirs(outp, exist_ok=True)

    df_stats = _summarize_notears_block_stats(
        fitted=fitted,
        blocks_order=blocks_order,
        w_threshold=w_threshold
    )
    df_stats.insert(0, "seed", int(seed))
    df_stats.insert(1, "fold", int(fold_id))
    df_stats.to_csv(f"{outp}/notears_block_stats_seed{seed}_fold{fold_id}{tag_suffix}.csv", index=False)

    pivot = df_stats.pivot(index="src_block", columns="tgt_block", values="fro_norm")
    pivot.to_csv(f"{outp}/notears_block_fro_seed{seed}_fold{fold_id}{tag_suffix}.csv")

    Wz = np.asarray(fitted["Wz"], dtype=np.float64)
    np.save(f"{outp}/notears_Wz_seed{seed}_fold{fold_id}{tag_suffix}.npy", Wz)

    if A_block is not None:
        A = np.asarray(A_block, dtype=int)
        np.save(f"{outp}/A_block_seed{seed}_fold{fold_id}{tag_suffix}.npy", A)

    if "latent_mask" in fitted and fitted["latent_mask"] is not None:
        np.save(f"{outp}/latent_mask_seed{seed}_fold{fold_id}{tag_suffix}.npy",
                np.asarray(fitted["latent_mask"], dtype=np.float64))

    return df_stats

def compute_latent_importance_from_Wz(Wz, mode="out_in", eps=1e-12, normalize="mean1"):
    W = np.asarray(Wz, dtype=np.float64)
    A = np.abs(W)

    out_strength = A.sum(axis=1)
    in_strength  = A.sum(axis=0)

    if mode == "out":
        a = out_strength
    elif mode == "in":
        a = in_strength
    elif mode == "out_in":
        a = out_strength + in_strength
    else:
        raise ValueError("mode must be one of {'out','in','out_in'}")

    a = np.maximum(a, 0.0)
    if not np.isfinite(a).all():
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

    if float(a.sum()) <= eps:
        a = np.ones_like(a, dtype=np.float64)

    if normalize == "mean1":
        a = a / max(float(a.mean()), eps)
    elif normalize == "sum1":
        a = a / max(float(a.sum()), eps)
    elif normalize is None:
        pass
    else:
        raise ValueError("normalize must be one of {None,'mean1','sum1'}")

    return a.astype(np.float64, copy=False)

def _apply_weight_controls(a, tau=1.0, clip_q=None, smooth_to_uniform=0.0, eps=1e-12):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    a = np.maximum(a, 0.0)

    if clip_q is not None:
        q = float(clip_q)
        q = min(max(q, 0.0), 1.0)
        cap = float(np.quantile(a, q)) if a.size > 0 else 0.0
        if np.isfinite(cap) and cap > 0:
            a = np.minimum(a, cap)

    tau = float(tau) if tau is not None else 1.0
    tau = max(tau, 1e-6)
    if abs(tau - 1.0) > 1e-12:
        a = np.exp(np.log(a + eps) / tau)

    lam = float(smooth_to_uniform) if smooth_to_uniform is not None else 0.0
    lam = min(max(lam, 0.0), 1.0)
    if lam > 0:
        a_mean = float(a.mean()) if a.size > 0 else 1.0
        if not np.isfinite(a_mean) or a_mean <= 0:
            a_mean = 1.0
        a = (1.0 - lam) * a + lam * a_mean

    m = float(a.mean()) if a.size > 0 else 1.0
    if not np.isfinite(m) or m <= eps:
        return np.ones_like(a, dtype=np.float64)
    return (a / m).astype(np.float64)


def _compute_a_z_block_from_Wz(fitted, block_name, use_post_threshold=True, eps=1e-12):
    Wz = np.asarray(fitted["Wz"], dtype=np.float64)
    starts = fitted["starts"]
    q_dims = fitted["q_dims"]

    if block_name not in starts or block_name not in q_dims:
        raise KeyError(f"Unknown block_name={block_name}")

    W = Wz.copy()
    if use_post_threshold and (fitted.get("w_threshold", None) is not None):
        thr = float(fitted["w_threshold"])
        W[np.abs(W) < thr] = 0.0

    np.fill_diagonal(W, 0.0)

    A = np.abs(W)
    strength = A.sum(axis=0) + A.sum(axis=1)

    sl = slice(starts[block_name], starts[block_name] + q_dims[block_name])
    a_z = strength[sl].reshape(-1)

    if (not np.isfinite(a_z).all()) or (float(a_z.sum()) <= eps):
        a_z = np.ones_like(a_z, dtype=np.float64)

    a_z = a_z / max(float(a_z.mean()), eps)
    return a_z.astype(np.float64)


def _compute_a_x_for_target_block(fitted, tgt_block, a_z_t=None,
                                 tau=1.0, clip_q=None, smooth_to_uniform=0.0, eps=1e-12):
    sc = fitted["scalers"][tgt_block]
    scale = getattr(sc, "scale_", None)
    if scale is None:
        scale = np.ones(int(sc.mean_.shape[0]), dtype=np.float64)

    pc = fitted["pcas"].get(tgt_block, None)
    p_t = int(scale.shape[0])

    if pc is None:
        a_orig = np.ones(p_t, dtype=np.float64)
        return _apply_weight_controls(a_orig, tau=tau, clip_q=clip_q, smooth_to_uniform=smooth_to_uniform, eps=eps)

    comps = np.asarray(pc.components_, dtype=np.float64)
    if comps.shape[1] != p_t:
        raise ValueError(f"[PCA/component mismatch] block={tgt_block}: comps.shape={comps.shape} vs p_t={p_t}")

    q_t = comps.shape[0]
    if a_z_t is None:
        a_z_t = np.ones(q_t, dtype=np.float64)
    a_z_t = np.asarray(a_z_t, dtype=np.float64).reshape(-1)
    if a_z_t.shape[0] != q_t:
        raise ValueError(f"[a_z dim mismatch] block={tgt_block}: len(a_z_t)={len(a_z_t)} vs q_t={q_t}")

    a_std = (comps * comps).T @ a_z_t
    a_orig = a_std / (np.maximum(scale, 0.0) ** 2 + eps)

    a_orig = _apply_weight_controls(a_orig, tau=tau, clip_q=clip_q, smooth_to_uniform=smooth_to_uniform, eps=eps)
    return a_orig.astype(np.float64)

def get_k_best(X_train, y_train, X_test, k_num, 
               pre_var_thresh=0.0,
               method="f_classif",
               random_state=0
              ):
    X_train = np.asarray(X_train)
    X_test  = np.asarray(X_test)
    y_train = np.asarray(y_train).astype(int)

    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("X_train and X_test must be 2D arrays.")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Feature dimension mismatch between train and test.")

    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise ValueError("X contains NaN or Inf. Please impute/clean before feature selection.")

    if pre_var_thresh is not None and pre_var_thresh >= 0:
        vt = VarianceThreshold(threshold=pre_var_thresh)
        X_train_v = vt.fit_transform(X_train)
        X_test_v  = vt.transform(X_test)
        vt_mask = vt.get_support()
    else:
        vt = None
        X_train_v, X_test_v = X_train, X_test
        vt_mask = np.ones(X_train.shape[1], dtype=bool)

    p = X_train_v.shape[1]
    if p == 0:
        raise ValueError("No features left after VarianceThreshold.")
    k = int(k_num)
    k = max(1, min(k, p))

    if method == "f_classif":
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X_train_v, y_train)
        Xtr_sel = selector.transform(X_train_v)
        Xte_sel = selector.transform(X_test_v)
        sel_mask_v = selector.get_support()

    elif method == "mutual_info":
        selector = SelectKBest(
            score_func=lambda X, y: mutual_info_classif(X, y, random_state=int(random_state)),
            k=k
        )
        selector.fit(X_train_v, y_train)
        Xtr_sel = selector.transform(X_train_v)
        Xte_sel = selector.transform(X_test_v)
        sel_mask_v = selector.get_support()

    elif method == "l1_logistic":
        clf = SkLogisticRegression(
            penalty="l1",
            solver="saga",
            max_iter=5000,
            class_weight="balanced",
            random_state=int(random_state)
        )
        sfm = SelectFromModel(clf, max_features=k)
        sfm.fit(X_train_v, y_train)
        Xtr_sel = sfm.transform(X_train_v)
        Xte_sel = sfm.transform(X_test_v)
        sel_mask_v = sfm.get_support()

        selector = {"variance_threshold": vt, "l1_logistic": sfm}
    else:
        raise ValueError("method must be one of {'f_classif','mutual_info','l1_logistic'}")
    support_mask = np.zeros_like(vt_mask, dtype=bool)
    support_mask[np.where(vt_mask)[0][sel_mask_v]] = True

    meta = {"variance_threshold": vt, "method": method, "selector": selector}
    return Xtr_sel, Xte_sel, meta, support_mask

def apply_feature_selector(X, meta):
    X = np.asarray(X)

    vt = meta.get("variance_threshold", None)
    method = meta.get("method", None)
    selector = meta.get("selector", None)

    Xv = vt.transform(X) if vt is not None else X

    if method in ("f_classif", "mutual_info"):
        return selector.transform(Xv)

    if method == "l1_logistic":
        if isinstance(selector, dict) and "l1_logistic" in selector:
            return selector["l1_logistic"].transform(Xv)
        return selector.transform(Xv)

    raise ValueError(f"Unknown method in meta: {method}")

def normalize_dataset(data, norm="l2", axis=1, copy=True):
    if "X" not in data:
        raise KeyError("data must contain key 'X'.")

    X = np.asarray(data["X"])
    if X.ndim != 2:
        raise ValueError("data['X'] must be a 2D array.")

    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or Inf. Please impute/clean before normalization.")

    data_new = dict(data) if copy else data

    Xn = preprocessing.normalize(X, norm=norm, axis=axis, copy=True)
    data_new["X"] = Xn
    return data_new

def tumor_types(cancer_type):
    Map = {
        "GBMLGG": ["GBM", "LGG"],
        "COADREAD": ["COAD", "READ"],
        "KIPAN": ["KIRC", "KICH", "KIRP"],
        "STES": ["ESCA", "STAD"],
        "PanGI": ["COAD", "STAD", "READ", "ESCA"],
        "PanGyn": ["OV", "CESC", "USC", "UCEC"],
        "PanSCCs": ["LUSC", "HNSC", "ESCA", "CESC", "BLCA"],
        "PanPan": [
            "ACC",
            "BLCA",
            "BRCA",
            "CESC",
            "CHOL",
            "COAD",
            "DLBC",
            "ESCA",
            "GBM",
            "HNSC",
            "KICH",
            "KIRC",
            "KIRP",
            "LAML",
            "LGG",
            "LIHC",
            "LUAD",
            "LUSC",
            "MESO",
            "OV",
            "PAAD",
            "PCPG",
            "PRAD",
            "READ",
            "SARC",
            "SKCM",
            "STAD",
            "TGCT",
            "THCA",
            "THYM",
            "UCEC",
            "UCS",
            "UVM",
        ],
    }
    if cancer_type not in Map:
        Map[cancer_type] = [cancer_type]

    return Map[cancer_type]

def get_race(cancer_type, input_path):
    input_path = Path(input_path)
    path = input_path / "Genetic_Ancestry.xlsx"
    if not path.exists():
        raise FileNotFoundError(f"Genetic_Ancestry.xlsx not found: {path}")

    code_to_race = {
        "EA": "WHITE",
        "AA": "BLACK",
        "EAA": "ASIAN",
        "NA": "NAT_A",
        "OA": "OTHER",
    }
    allowed = set(code_to_race.keys())

    dfs = []
    for disease in tumor_types(cancer_type):
        try:
            df = pd.read_excel(path, sheet_name=disease, keep_default_na=False)
        except Exception as e:
            raise RuntimeError(f"Failed to read sheet '{disease}' from {path}: {e}")

        df.columns = [str(c).strip() for c in df.columns]

        if "Patient_ID" not in df.columns:
            raise KeyError(f"'Patient_ID' column not found in sheet '{disease}'. Got columns: {list(df.columns)}")
        if "EIGENSTRAT" not in df.columns:
            raise KeyError(f"'EIGENSTRAT' column not found in sheet '{disease}'. Got columns: {list(df.columns)}")

        tmp = df[["Patient_ID", "EIGENSTRAT"]].copy()
        tmp["Patient_ID"] = tmp["Patient_ID"].astype(str).str.strip()
        tmp["EIGENSTRAT"] = tmp["EIGENSTRAT"].astype(str).str.strip().str.upper()

        tmp = tmp.set_index("Patient_ID")
        dfs.append(tmp)

    df_race = pd.concat(dfs, axis=0)

    df_race = df_race[df_race["EIGENSTRAT"].isin(allowed)].copy()
    df_race["race"] = df_race["EIGENSTRAT"].map(code_to_race)

    df_race = df_race[~df_race.index.duplicated(keep="first")]

    if df_race.shape[0] == 0:
        raise ValueError(
            "No valid race labels after filtering EIGENSTRAT codes. "
            f"Allowed={sorted(list(allowed))}. Check Genetic_Ancestry.xlsx contents."
        )

    return df_race[["race"]]

def add_race_CT(
    cancer_type, df, target, groups, input_path,
    feature_columns=None,
    dropna_cols=True,
    c_indicator="censor",
):
    df_race = get_race(cancer_type, input_path)
    df_race = df_race[df_race["race"].isin(groups)]

    df_C_T = get_CT(target, input_path)

    df = df.join(df_race, how="inner")

    if feature_columns is not None:
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature_columns after race join: {missing[:10]}")
        df = df[feature_columns + ["race"]]

    if dropna_cols:
        df = df.dropna(axis="columns")

    df = df.join(df_C_T, how="inner")

    for col in ["C", "T", "race"]:
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' not found after joins. Got: {list(df.columns)[:20]} ...")

    C = df["C"].astype(int).to_numpy()
    uniq = set(np.unique(C).tolist())
    if not uniq.issubset({0, 1}):
        raise ValueError(f"C must be binary {{0,1}}. Got unique values={sorted(list(uniq))}")

    if c_indicator == "censor":
        E = (1 - C).astype(int)
        C_censor = C.astype(int)
    elif c_indicator == "event":
        E = C.astype(int)
        C_censor = (1 - C).astype(int)
    else:
        raise ValueError("c_indicator must be one of {'censor','event'}")

    T_days = df["T"].astype(float).to_numpy()
    R = df["race"].astype(str).to_numpy()

    Xdf = df.drop(columns=["C", "race", "T"])
    X = Xdf.to_numpy(dtype="float32")

    if not np.isfinite(X).all():
        raise ValueError("X contains NaN/Inf after joins. Fix/impute before downstream steps.")

    data = {
        "X": X,
        "T": T_days.astype(np.float32),
        "C": C_censor.astype(np.int32),     
        "E": E.astype(np.int32),            
        "R": R,
        "Samples": Xdf.index.values,
        "FeatureName": list(Xdf.columns),
        "_C_indicator": c_indicator
    }
    return data

def get_CT(target, input_path):
    input_path = Path(input_path)
    path = input_path / "TCGA-CDR-SupplementalTableS1.xlsx"
    sheet = "TCGA-CDR"

    if not path.exists():
        raise FileNotFoundError(f"TCGA-CDR-SupplementalTableS1.xlsx not found: {path}")

    if target == "OS":
        cols = "B,Z,AA"
    elif target == "DSS":
        cols = "B,AB,AC"
    elif target == "DFI":
        cols = "B,AD,AE"
    elif target == "PFI":
        cols = "B,AF,AG"
    else:
        raise ValueError("target must be one of {'OS','DSS','DFI','PFI'}")

    df = pd.read_excel(path, sheet_name=sheet, usecols=cols, keep_default_na=True)

    if df.shape[1] != 3:
        raise ValueError(
            f"Expected 3 columns (barcode + 2 fields) but got {df.shape[1]} using usecols={cols}"
        )

    barcode_col = df.columns[0]
    event_col   = df.columns[1]
    time_col    = df.columns[2]

    df = df.set_index(barcode_col)
    df = df[[event_col, time_col]].copy()
    df.columns = ["E", "T"]

    df = df.dropna(subset=["E", "T"])
    df = df[df["E"].isin([0, 1])]
    df["E"] = df["E"].astype(int)
    df["T"] = df["T"].astype(float)

    df["C"] = 1 - df["E"]
    df = df.drop(columns=["E"])

    return df[["C", "T"]]

def get_n_years(dataset, years, days_per_year=365, keep_meta=True, allow_empty=False):
    for k in ["X", "T", "C", "R"]:
        if k not in dataset:
            raise KeyError(f"dataset must contain key '{k}'")

    X = np.asarray(dataset["X"])
    T = np.asarray(dataset["T"], dtype=np.float64)
    C = np.asarray(dataset["C"], dtype=np.int32)
    R = np.asarray(dataset["R"]).astype(str)

    n = X.shape[0]
    if not (T.shape[0] == n and C.shape[0] == n and R.shape[0] == n):
        raise ValueError("Length mismatch among X, T, C, R")

    if not np.isfinite(T).all():
        raise ValueError("T contains NaN/Inf.")
    if np.any(T < 0):
        raise ValueError("T contains negative values.")
    if not np.isin(C, [0, 1]).all():
        raise ValueError("C must be binary {0,1} with 1=censored, 0=event.")

    time_p = float(days_per_year) * float(years)

    keep_mask = ~((T <= time_p) & (C == 1))

    if (not allow_empty) and (keep_mask.sum() == 0):
        raise ValueError("No samples left after removing censored at/before time_p cases.")

    Xk = X[keep_mask]
    Tk = T[keep_mask]
    Ck = C[keep_mask]
    Rk = R[keep_mask]

    Y = np.ones(Tk.shape[0], dtype=np.int32)
    Y[(Tk <= time_p) & (Ck == 0)] = 0

    y_strat = np.char.add(np.char.add(Y.astype(str), "_"), Rk.astype(str))
    y_sub = Rk

    return (Xk.astype(np.float32), Y, Rk, y_sub, y_strat)

def find_best_threshold_by_f1(y_true, score, average="binary"):
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    score  = np.asarray(score).astype(float).reshape(-1)

    if np.unique(y_true).size < 2:
        thr = 0.5
        pred = (score >= thr).astype(int)
        return thr, f1_score(y_true, pred, average=average), accuracy_score(y_true, pred)

    thr_list = np.unique(score)
    if thr_list.size > 200:
        thr_list = np.linspace(score.min(), score.max(), 200)

    best_thr, best_f1, best_acc = 0.5, -1.0, -1.0
    for thr in thr_list:
        pred = (score >= thr).astype(int)
        f1  = f1_score(y_true, pred, average=average)
        acc = accuracy_score(y_true, pred)
        if f1 > best_f1:
            best_thr, best_f1, best_acc = float(thr), float(f1), float(acc)

    return best_thr, best_f1, best_acc

def _safe_binary_metrics(y_true, y_score, y_pred=None):
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_score = np.asarray(y_score).astype(float).reshape(-1)

    if y_pred is None:
        y_pred = (y_score >= 0.5).astype(int)
    else:
        y_pred = np.asarray(y_pred).astype(int).reshape(-1)

    out = {
        "n": int(len(y_true)),
        "n_event": int(np.sum(y_true == 1)),
        "prevalence": float(np.mean(y_true)) if len(y_true) > 0 else np.nan,
        "auc": np.nan,
        "prauc": np.nan,
        "acc": np.nan,
        "f1": np.nan,
        "brier": np.nan,
    }

    if len(y_true) == 0:
        return out

    if len(np.unique(y_true)) >= 2:
        out["auc"] = roc_auc_score(y_true, y_score)
        out["prauc"] = average_precision_score(y_true, y_score)
    out["acc"] = accuracy_score(y_true, y_pred)
    out["f1"] = f1_score(y_true, y_pred, average="weighted")
    out["brier"] = brier_score_loss(y_true, np.clip(y_score, 1e-8, 1 - 1e-8))
    return out

def save_subgroup_fold_metrics(df_fold, save_path, seed, fold_id, method_tag):
    rows = []

    for subgroup, subdf in [
        ("ALL", df_fold),
        ("WHITE", df_fold[df_fold["R"] == "WHITE"]),
        ("BLACK", df_fold[df_fold["R"] == "BLACK"]),
    ]:
        if len(subdf) == 0:
            continue

        m = _safe_binary_metrics(
            y_true=subdf["Y"].values,
            y_score=subdf["scr"].values,
            y_pred=subdf["pred_thr"].values
        )

        cidx = np.nan
        sub_c = subdf.dropna(subset=["T", "E", "risk"]).copy()
        if (len(sub_c) >= 2) and (sub_c["E"].sum() > 0) and (sub_c["T"].nunique() >= 2):
            cidx = concordance_index_simple(
                T=sub_c["T"].values,
                risk=sub_c["risk"].values,
                E=sub_c["E"].astype(int).values
            )

        rows.append({
            "seed": int(seed),
            "fold": int(fold_id),
            "method": str(method_tag),
            "subgroup": subgroup,
            "n": m["n"],
            "n_positive_event_free": m["n_event"],
            "positive_rate_event_free": m["prevalence"],
            "auc": m["auc"],
            "prauc": m["prauc"],
            "acc": m["acc"],
            "f1": m["f1"],
            "brier": m["brier"],
            "cindex": cidx,
            "best_thr": float(subdf["best_thr"].iloc[0]) if len(subdf) > 0 else np.nan,
        })

    pd.DataFrame(rows).to_csv(save_path, index=False)

def _calibration_intercept_slope(y_true, y_score, eps=1e-8):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_score = np.clip(y_score, eps, 1 - eps)

    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan

    z = logit(y_score)
    X = np.column_stack([np.ones_like(z), z])

    beta = np.zeros(2, dtype=float)
    for _ in range(50):
        eta = X @ beta
        p = expit(eta)
        W = np.clip(p * (1 - p), eps, None)
        z_work = eta + (y_true - p) / W
        XtW = X.T * W
        beta_new = np.linalg.pinv(XtW @ X) @ (XtW @ z_work)
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new

    intercept = float(beta[0])
    slope = float(beta[1])
    return intercept, slope


def save_calibration_outputs(df_fold, save_points_path, save_summary_path, n_bins=10):
    rows_points, rows_summary = [], []

    for subgroup, subdf in [
        ("ALL", df_fold),
        ("WHITE", df_fold[df_fold["R"] == "WHITE"]),
        ("BLACK", df_fold[df_fold["R"] == "BLACK"]),
    ]:
        if len(subdf) < 5:
            continue

        y_true = subdf["Y"].values.astype(int)
        y_score = np.clip(subdf["scr"].values.astype(float), 1e-8, 1 - 1e-8)

        intercept, slope = _calibration_intercept_slope(y_true, y_score)
        rows_summary.append({
            "subgroup": subgroup,
            "n": int(len(subdf)),
            "brier": brier_score_loss(y_true, y_score) if len(np.unique(y_true)) >= 2 else np.nan,
            "calibration_intercept": intercept,
            "calibration_slope": slope,
        })

        try:
            prob_true, prob_pred = calibration_curve(
                y_true, y_score, n_bins=n_bins, strategy="quantile"
            )
            for i, (pp, pt) in enumerate(zip(prob_pred, prob_true)):
                rows_points.append({
                    "subgroup": subgroup,
                    "bin_id": int(i),
                    "mean_pred": float(pp),
                    "observed_rate": float(pt),
                })
        except Exception:
            pass

    pd.DataFrame(rows_points).to_csv(save_points_path, index=False)
    pd.DataFrame(rows_summary).to_csv(save_summary_path, index=False)

def _net_benefit(y_true, y_score, pt):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n = len(y_true)
    if n == 0 or pt <= 0 or pt >= 1:
        return np.nan

    pred_pos = (y_score >= pt).astype(int)
    tp = np.sum((pred_pos == 1) & (y_true == 1))
    fp = np.sum((pred_pos == 1) & (y_true == 0))
    return (tp / n) - (fp / n) * (pt / (1 - pt))


def save_dca_outputs(df_fold, save_path, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.05, 0.81, 0.01)

    rows = []

    for subgroup, subdf in [
        ("ALL", df_fold),
        ("WHITE", df_fold[df_fold["R"] == "WHITE"]),
        ("BLACK", df_fold[df_fold["R"] == "BLACK"]),
    ]:
        if len(subdf) == 0:
            continue

        y_true = subdf["Y"].values.astype(int)
        y_score = subdf["scr"].values.astype(float)
        prevalence = np.mean(y_true) if len(y_true) > 0 else np.nan

        for pt in thresholds:
            nb_model = _net_benefit(y_true, y_score, pt)
            nb_all = prevalence - (1 - prevalence) * (pt / (1 - pt))
            nb_none = 0.0

            rows.append({
                "subgroup": subgroup,
                "threshold": float(pt),
                "net_benefit_model": float(nb_model),
                "net_benefit_all": float(nb_all),
                "net_benefit_none": float(nb_none),
            })

    pd.DataFrame(rows).to_csv(save_path, index=False)

def make_method_summary_row(df_score, seed, method_tag, lr=None, dr=None, alpha=None,
                            a_block_type=None, integration_type=None):

    m = _safe_binary_metrics(
        y_true=df_score["Y"].values,
        y_score=df_score["scr"].values,
        y_pred=df_score["pred_thr"].values
    )

    cidx = np.nan
    df_c = df_score.dropna(subset=["T", "E", "risk"]).copy()
    if (len(df_c) >= 2) and (df_c["E"].sum() > 0) and (df_c["T"].nunique() >= 2):
        cidx = concordance_index_simple(
            df_c["T"].values,
            df_c["risk"].values,
            df_c["E"].astype(int).values
        )

    return {
        "seed": int(seed),
        "method": str(method_tag),
        "auc": m["auc"],
        "prauc": m["prauc"],
        "acc": m["acc"],
        "f1": m["f1"],
        "brier": m["brier"],
        "cindex": cidx,
        "lr": lr,
        "dr": dr,
        "alpha": alpha,
        "a_block_type": a_block_type,
        "integration_type": integration_type,
    }

def save_kpca_separation_metrics(Z, Y, R, save_path):
    rows = []
    Z = np.asarray(Z, dtype=float)
    Y = np.asarray(Y)
    R = np.asarray(R).astype(str)

    if Z.ndim != 2 or Z.shape[0] < 3:
        pd.DataFrame().to_csv(save_path, index=False)
        return

    if len(np.unique(Y)) >= 2 and len(Y) > len(np.unique(Y)):
        try:
            sY = silhouette_score(Z, Y)
        except Exception:
            sY = np.nan
    else:
        sY = np.nan

    if len(np.unique(R)) >= 2 and len(R) > len(np.unique(R)):
        try:
            sR = silhouette_score(Z, R)
        except Exception:
            sR = np.nan
    else:
        sR = np.nan

    rows.append({
        "silhouette_by_Y": sY,
        "silhouette_by_R": sR,
    })
    pd.DataFrame(rows).to_csv(save_path, index=False)

def summarize_mkl_weights_by_category(df_mkl):
    df = df_mkl.copy()
    def _cat(tag):
        tag = str(tag)
        if "__g" in tag:
            pair, gamma = tag.split("__g", 1)
        else:
            pair, gamma = tag, "NA"

        if "->" in pair:
            s, t = pair.split("->", 1)
            if s == t:
                pair_cat = "within_block"
            elif pair in ["miRNA->mRNA", "methyl->mRNA"]:
                pair_cat = pair.replace("->", "_to_")
            else:
                pair_cat = "other_cross_block"
        else:
            pair_cat = "other"

        return pd.Series({
            "pair_category": pair_cat,
            "gamma_category": f"g{gamma}"
        })

    extra = df["kernel_tag"].apply(_cat)
    df = pd.concat([df, extra], axis=1)

    out = df.groupby(["pair_category", "gamma_category"], as_index=False)["weight"].agg(["mean", "std", "sum"])
    out = out.reset_index()
    return out

def summarize_notears_stability(notears_dir, pattern="notears_block_stats_seed*_fold*.csv"):
    files = sorted(glob.glob(os.path.join(str(notears_dir), pattern)))
    if len(files) == 0:
        return pd.DataFrame()

    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, axis=0, ignore_index=True)

    eps = 1e-12
    df["selected"] = (df["fro_norm"].abs() > eps).astype(int)

    out = df.groupby(["src_block", "tgt_block"], as_index=False).agg(
        mean_fro=("fro_norm", "mean"),
        sd_fro=("fro_norm", "std"),
        mean_density=("density", "mean"),
        sd_density=("density", "std"),
        selection_frequency=("selected", "mean"),
        mean_l1=("l1_sum", "mean"),
        sd_l1=("l1_sum", "std"),
        n_runs=("selected", "size"),
    )
    return out

def summarize_threshold_stability(thr_df, seed, method_tag):
    if thr_df is None or len(thr_df) == 0:
        return pd.DataFrame()

    return pd.DataFrame([{
        "seed": int(seed),
        "method": str(method_tag),
        "mean_threshold": float(thr_df["best_thr"].mean()),
        "sd_threshold": float(thr_df["best_thr"].std(ddof=0)),
        "min_threshold": float(thr_df["best_thr"].min()),
        "max_threshold": float(thr_df["best_thr"].max()),
        "n_folds": int(len(thr_df)),
    }])

def save_cohort_summary(inter_data, years, save_path):
    T = np.asarray(inter_data["T"], dtype=float)
    C = np.asarray(inter_data["C"], dtype=int)
    R = np.asarray(inter_data["R"]).astype(str)

    time_p = 365 * int(years)
    removed_early = ((T <= time_p) & (C == 1)).astype(int)

    rows = []
    for subgroup, mask in [("ALL", np.ones(len(R), dtype=bool)),
                           ("WHITE", R == "WHITE"),
                           ("BLACK", R == "BLACK")]:
        if mask.sum() == 0:
            continue

        rows.append({
            "subgroup": subgroup,
            "N": int(mask.sum()),
            "event_count": int(np.sum((1 - C[mask]) == 1)),
            "censor_count": int(np.sum(C[mask] == 1)),
            "event_rate": float(np.mean((1 - C[mask]) == 1)),
            "median_followup_days": float(np.median(T[mask])),
            "mean_followup_days": float(np.mean(T[mask])),
            "removed_early_censor_count": int(np.sum(removed_early[mask])),
        })

    pd.DataFrame(rows).to_csv(save_path, index=False)

def summarize_cox_risk_effect(km_df, save_path=None):
    if (not HAVE_LIFELINES) or (km_df is None) or (len(km_df) == 0):
        out = pd.DataFrame()
        if save_path is not None:
            out.to_csv(save_path, index=False)
        return out

    rows = []
    for subgroup, subdf in [("ALL", km_df),
                            ("WHITE", km_df[km_df["R"] == "WHITE"]),
                            ("BLACK", km_df[km_df["R"] == "BLACK"])]:
        subdf = subdf.dropna(subset=["risk", "T_days", "Event"]).copy()
        if len(subdf) < 10 or subdf["Event"].sum() == 0:
            continue

        subdf["risk_z"] = (subdf["risk"] - subdf["risk"].mean()) / (subdf["risk"].std(ddof=0) + 1e-12)

        try:
            cph = CoxPHFitter()
            cph.fit(subdf[["T_days", "Event", "risk_z"]], duration_col="T_days", event_col="Event")
            s = cph.summary.loc["risk_z"]

            rows.append({
                "subgroup": subgroup,
                "n": int(len(subdf)),
                "n_event": int(subdf["Event"].sum()),
                "hr_per_1sd": float(np.exp(s["coef"])),
                "coef": float(s["coef"]),
                "se": float(s["se(coef)"]),
                "p_value": float(s["p"]),
                "ci_lower": float(np.exp(s["coef lower 95%"])),
                "ci_upper": float(np.exp(s["coef upper 95%"])),
            })
        except Exception:
            continue

    out = pd.DataFrame(rows)
    if save_path is not None:
        out.to_csv(save_path, index=False)
    return out

def paired_bootstrap_metric_diff(y_true, s1, s2, metric="auc", n_boot=2000, seed=0):
    y_true = np.asarray(y_true).astype(int)
    s1 = np.asarray(s1).astype(float)
    s2 = np.asarray(s2).astype(float)

    rng = np.random.default_rng(seed)
    n = len(y_true)

    def _metric(y, s, kind):
        if len(np.unique(y)) < 2:
            return np.nan
        if kind == "auc":
            return roc_auc_score(y, s)
        elif kind == "prauc":
            return average_precision_score(y, s)
        elif kind == "brier":
            return brier_score_loss(y, np.clip(s, 1e-8, 1-1e-8))
        else:
            raise ValueError("metric must be one of {'auc','prauc','brier'}")

    diffs = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        s1b = s1[idx]
        s2b = s2[idx]
        m1 = _metric(yb, s1b, metric)
        m2 = _metric(yb, s2b, metric)
        if np.isfinite(m1) and np.isfinite(m2):
            diffs.append(m2 - m1)

    if len(diffs) == 0:
        return {
            "metric": metric,
            "diff_mean": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_boot_valid": 0
        }

    diffs = np.asarray(diffs, dtype=float)
    return {
        "metric": metric,
        "diff_mean": float(np.mean(diffs)),
        "ci_lower": float(np.quantile(diffs, 0.025)),
        "ci_upper": float(np.quantile(diffs, 0.975)),
        "n_boot_valid": int(len(diffs))
    }

def paired_permutation_test(y_true, s1, s2, metric="auc", n_perm=5000, seed=0):
    y_true = np.asarray(y_true).astype(int)
    s1 = np.asarray(s1).astype(float)
    s2 = np.asarray(s2).astype(float)

    rng = np.random.default_rng(seed)

    def _metric(y, s, kind):
        if len(np.unique(y)) < 2:
            return np.nan
        if kind == "auc":
            return roc_auc_score(y, s)
        elif kind == "prauc":
            return average_precision_score(y, s)
        elif kind == "brier":
            return brier_score_loss(y, np.clip(s, 1e-8, 1-1e-8))
        else:
            raise ValueError("metric must be one of {'auc','prauc','brier'}")

    obs1 = _metric(y_true, s1, metric)
    obs2 = _metric(y_true, s2, metric)
    if (not np.isfinite(obs1)) or (not np.isfinite(obs2)):
        return {"metric": metric, "observed_diff": np.nan, "p_value": np.nan}

    obs = obs2 - obs1
    cnt = 0
    valid = 0

    for _ in range(int(n_perm)):
        swap = rng.integers(0, 2, size=len(y_true)).astype(bool)
        p1 = s1.copy()
        p2 = s2.copy()
        tmp = p1[swap].copy()
        p1[swap] = p2[swap]
        p2[swap] = tmp

        m1 = _metric(y_true, p1, metric)
        m2 = _metric(y_true, p2, metric)
        if np.isfinite(m1) and np.isfinite(m2):
            valid += 1
            if abs(m2 - m1) >= abs(obs):
                cnt += 1

    p = (cnt + 1) / (valid + 1) if valid > 0 else np.nan
    return {"metric": metric, "observed_diff": float(obs), "p_value": float(p)}

def summarize_paired_method_comparison(paired_all_df, method_col="method"):
    df = paired_all_df.copy()

    keep_cols = ["seed", "method", "auc", "prauc", "acc", "f1", "brier", "cindex", "lr", "dr"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    wide = df.pivot_table(index="seed", columns=method_col, aggfunc="mean")
    wide.columns = [f"{m}_{k}" for k, m in wide.columns]
    wide = wide.reset_index()

    rows = []
    for metric in ["auc", "prauc", "acc", "f1", "brier", "cindex"]:
        c1 = f"{metric}_TL_only"
        c2 = f"{metric}_TLDA"
        if c1 in wide.columns and c2 in wide.columns:
            diff = wide[c2] - wide[c1]
            rows.append({
                "metric": metric,
                "n_seeds": int(diff.notna().sum()),
                "mean_TL": float(wide[c1].mean()),
                "mean_TLDA": float(wide[c2].mean()),
                "mean_diff_TLDA_minus_TL": float(diff.mean()),
                "sd_diff": float(diff.std(ddof=1)) if diff.notna().sum() >= 2 else np.nan,
            })
    return pd.DataFrame(rows), wide

def aggregate_subgroup_metrics(score_dir, method_tag):
    files = sorted(glob.glob(os.path.join(score_dir, f"subgroup_fold_metrics_seed*_fold*_{method_tag}.csv")))
    if len(files) == 0:
        return pd.DataFrame()

    df = pd.concat([pd.read_csv(f) for f in files], axis=0, ignore_index=True)
    out = df.groupby(["subgroup"], as_index=False).agg(
        mean_auc=("auc", "mean"),
        sd_auc=("auc", "std"),
        mean_prauc=("prauc", "mean"),
        sd_prauc=("prauc", "std"),
        mean_acc=("acc", "mean"),
        sd_acc=("acc", "std"),
        mean_f1=("f1", "mean"),
        sd_f1=("f1", "std"),
        mean_brier=("brier", "mean"),
        sd_brier=("brier", "std"),
        mean_cindex=("cindex", "mean"),
        sd_cindex=("cindex", "std"),
        n_runs=("auc", "size"),
    )
    out["method"] = method_tag
    return out

def aggregate_calibration_summaries(score_dir, method_tag):
    files = sorted(glob.glob(os.path.join(score_dir, f"calibration_summary_seed*_fold*_{method_tag}.csv")))
    if len(files) == 0:
        return pd.DataFrame()

    df = pd.concat([pd.read_csv(f) for f in files], axis=0, ignore_index=True)
    out = df.groupby(["subgroup"], as_index=False).agg(
        mean_brier=("brier", "mean"),
        sd_brier=("brier", "std"),
        mean_intercept=("calibration_intercept", "mean"),
        sd_intercept=("calibration_intercept", "std"),
        mean_slope=("calibration_slope", "mean"),
        sd_slope=("calibration_slope", "std"),
        n_runs=("brier", "size"),
    )
    out["method"] = method_tag
    return out

def aggregate_dca(score_dir, method_tag):
    files = sorted(glob.glob(os.path.join(score_dir, f"dca_seed*_fold*_{method_tag}.csv")))
    if len(files) == 0:
        return pd.DataFrame()

    df = pd.concat([pd.read_csv(f) for f in files], axis=0, ignore_index=True)
    out = df.groupby(["subgroup", "threshold"], as_index=False).agg(
        mean_nb_model=("net_benefit_model", "mean"),
        sd_nb_model=("net_benefit_model", "std"),
        mean_nb_all=("net_benefit_all", "mean"),
        sd_nb_all=("net_benefit_all", "std"),
        mean_nb_none=("net_benefit_none", "mean"),
        n_runs=("net_benefit_model", "size"),
    )
    out["method"] = method_tag
    return out

def train_and_predict(
    X_train_target,
    y_train_target,
    X_train_source,
    y_train_source,
    X_val_target,
    y_val_target,
    X_test,
    y_test,
    repetition,
    sample_per_class,
    alpha=ALPHA,
    learning_rate=0.01,
    hiddenLayers=None,
    dr=0.5,
    momentum=0.0,
    batch_size=32,
    n_features=None,
    domain_adaptation_task="WHITE_to_BLACK",
    seed=35,
    pairs_dir=None,
    decay=DECAY,
    apply_domain_smote=False,
    pos_neg_ratio=POS_NEG,
    balance_mode="overall",
    smote_k_max=5
):
    if hiddenLayers is None:
        hiddenLayers = [100]
    else:
        if hasattr(hiddenLayers, "numpy"):
            hiddenLayers = hiddenLayers.numpy()
        if isinstance(hiddenLayers, np.ndarray):
            hiddenLayers = hiddenLayers.reshape(-1).tolist()
        hiddenLayers = [int(x) for x in list(hiddenLayers)]

    random.seed(int(seed))
    np.random.seed(int(seed))
    tf.random.set_seed(int(seed))

    X_train_target = np.asarray(X_train_target, dtype=np.float32)
    X_train_source = np.asarray(X_train_source, dtype=np.float32)
    X_val_target   = np.asarray(X_val_target,   dtype=np.float32)
    X_test         = np.asarray(X_test,         dtype=np.float32)

    y_train_target = np.asarray(y_train_target, dtype=np.int32).reshape(-1)
    y_train_source = np.asarray(y_train_source, dtype=np.int32).reshape(-1)
    y_val_target   = np.asarray(y_val_target,   dtype=np.int32).reshape(-1)
    y_test         = np.asarray(y_test,         dtype=np.int32).reshape(-1)

    if n_features is None:
        n_features = int(X_train_target.shape[1])
    n_features = int(n_features)

    if X_train_target.ndim != 2 or X_train_source.ndim != 2:
        raise ValueError("X_train_target/X_train_source must be 2D arrays.")
    if X_train_target.shape[1] != n_features or X_train_source.shape[1] != n_features:
        raise ValueError("Feature dimension mismatch with n_features.")
    if X_val_target.shape[1] != n_features or X_test.shape[1] != n_features:
        raise ValueError("Validation/Test feature dimension mismatch with n_features.")

    input_shape = (n_features,)
    input_a = Input(shape=input_shape, name="input_target")
    input_b = Input(shape=input_shape, name="input_source")

    encoder = Initialization_kernel.Create_Model(hiddenLayers=hiddenLayers, dr=float(dr))

    processed_a = encoder(input_a)
    processed_b = encoder(input_b)

    processed_a = Dropout(float(dr), name="cls_dropout")(processed_a)
    logits = Dense(2, name="logits")(processed_a)
    out_cls = Activation("softmax", name="classification")(logits)

    out_csa = Lambda(
        Initialization_kernel.euclidean_distance,
        output_shape=Initialization_kernel.eucl_dist_output_shape,
        name="CSA",
    )([processed_a, processed_b])

    model = Model(inputs=[input_a, input_b], outputs=[out_cls, out_csa])

    optimizer = tf.keras.optimizers.legacy.SGD(
        learning_rate=float(learning_rate),
        momentum=float(momentum),
        decay=float(decay)
    )

    cls_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    model.compile(
        optimizer=optimizer,
        loss={
            "classification": cls_loss,
            "CSA": Initialization_kernel.contrastive_loss,
        },
        loss_weights={"classification": 1.0 - float(alpha), "CSA": float(alpha)},
    )

    if pairs_dir is None:
        pairs_dir = "./CCSA_pairs"
    pairs_dir = str(Path(pairs_dir).resolve())
    Path(pairs_dir).mkdir(parents=True, exist_ok=True)

    print("Domain Adaptation Task:", domain_adaptation_task)

    Initialization_kernel.Create_Pairs(
        domain_adaptation_task,
        repetition,
        sample_per_class,
        X_train_target,
        y_train_target,
        X_train_source,
        y_train_source,
        n_features=n_features,
        pairs_dir=pairs_dir,
        seed=int(seed),
        pos_neg_ratio=float(pos_neg_ratio),
        balance_target_to_source=bool(apply_domain_smote),
        balance_mode=str(balance_mode),
        smote_k_max=int(smote_k_max),
    )

    Initialization_kernel.training_the_model(
        model=model,
        domain_adaptation_task=domain_adaptation_task,
        repetition=repetition,
        sample_per_class=sample_per_class,
        batch_size=batch_size,
        X_val_target=X_val_target,
        y_val_target=y_val_target,
        seed=seed,
        max_epochs=100,
        patience=PATIENCE,
        pairs_dir=pairs_dir
    )

    out_val = model.predict([X_val_target, X_val_target], batch_size=int(batch_size), verbose=0)
    p_val = out_val[0]
    val_score = np.asarray(p_val, dtype=np.float64)[:, 1]

    flip = False
    if len(np.unique(y_val_target)) == 2:
        vauc = roc_auc_score(y_val_target, val_score)
        if vauc < 0.5:
            flip = True
    if flip:
        val_score = 1.0 - val_score

    best_thr, best_val_f1, best_val_acc = find_best_threshold_by_f1(
        y_val_target, val_score, average="binary"
    )

    out_test = model.predict([X_test, X_test], batch_size=int(batch_size), verbose=0)
    p_test = out_test[0]
    test_score = np.asarray(p_test, dtype=np.float64)[:, 1]
    if flip:
        test_score = 1.0 - test_score

    pred_test = (test_score >= best_thr).astype(int)

    test_auc = roc_auc_score(y_test, test_score) if len(np.unique(y_test)) == 2 else np.nan

    print("flip:", flip, "best_thr:", best_thr,
          "val_f1:", best_val_f1, "val_acc:", best_val_acc,
          "test_pred_ratio:", float(np.mean(pred_test)))

    return test_score.astype(float, copy=False), test_auc, float(best_thr), pred_test

def report_q_dims(fitted, print_table=True):
    q_dims = fitted["q_dims"]
    blocks_order = fitted.get("blocks_order", fitted["blocks"])
    rows = []
    q_total = 0
    for b in blocks_order:
        q = int(q_dims[b])
        rows.append({"block": b, "q_dim": q})
        q_total += q

    df = pd.DataFrame(rows)
    if print_table:
        print(df.to_string(index=False))
        print(f"[q_total] = {q_total}")
    return df, q_total

def export_pca_top_loadings(
    fitted,
    block_feature_names=None,
    top_k=20,
    out_dir=None,
    prefix="pca_loadings",
):
    pcas = fitted["pcas"]
    blocks_order = fitted.get("blocks_order", fitted["blocks"])

    all_rows = []

    for blk in blocks_order:
        pc = pcas.get(blk, None)
        if pc is None:
            continue

        comps = np.asarray(pc.components_)
        q, p = comps.shape

        if block_feature_names is not None and blk in block_feature_names:
            names = list(block_feature_names[blk])
            if len(names) != p:
                raise ValueError(f"[feature name mismatch] block={blk}: len(names)={len(names)} but p={p}")
        else:
            names = [f"{blk}_feat_{i}" for i in range(p)]

        for comp_id in range(q):
            w = comps[comp_id, :]
            idx = np.argsort(np.abs(w))[::-1][:min(top_k, p)]
            for rank, j in enumerate(idx, start=1):
                all_rows.append({
                    "block": blk,
                    "pc": comp_id + 1,
                    "rank": rank,
                    "feature": names[j],
                    "loading": float(w[j]),
                    "abs_loading": float(abs(w[j])),
                    "explained_var_ratio": float(pc.explained_variance_ratio_[comp_id]) if hasattr(pc, "explained_variance_ratio_") else np.nan
                })

    df = pd.DataFrame(all_rows).sort_values(["block", "pc", "rank"]).reset_index(drop=True)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{prefix}_top{top_k}.csv")
        df.to_csv(out_path, index=False)
        print(f"[SAVED] {out_path}")

    return df

def summarize_top_loading_stability(fig7_dir, method_tag, top_k=20):
    files = sorted(glob.glob(os.path.join(fig7_dir, f"{method_tag}_seed*_fold*_pca_loadings_top{top_k}.csv")))
    if len(files) == 0:
        return pd.DataFrame()

    dfs = []
    for f in files:
        tmp = pd.read_csv(f)
        dfs.append(tmp)

    df = pd.concat(dfs, axis=0, ignore_index=True)

    out = df.groupby(["block", "feature"], as_index=False).agg(
        n_selected=("feature", "size"),
        mean_rank=("rank", "mean"),
        sd_rank=("rank", "std"),
        mean_abs_loading=("abs_loading", "mean"),
        sd_abs_loading=("abs_loading", "std"),
        mean_signed_loading=("loading", "mean"),
        sd_signed_loading=("loading", "std"),
        mean_explained_var_ratio=("explained_var_ratio", "mean"),
    )

    out = out.sort_values(
        ["block", "n_selected", "mean_abs_loading"],
        ascending=[True, False, False]
    ).reset_index(drop=True)

    return out

def make_consensus_top_features(stability_df, top_n=20):
    if stability_df is None or len(stability_df) == 0:
        return pd.DataFrame()

    rows = []
    for blk, sub in stability_df.groupby("block"):
        sub2 = sub.sort_values(
            ["n_selected", "mean_abs_loading", "mean_rank"],
            ascending=[False, False, True]
        ).head(top_n).copy()
        sub2["consensus_rank"] = np.arange(1, len(sub2) + 1)
        rows.append(sub2)

    return pd.concat(rows, axis=0, ignore_index=True) if len(rows) > 0 else pd.DataFrame()

def _save_kpca_fold_artifacts(
    outp, run_tag_safe, seed, fold_id,
    mode_name,
    sample_ids,
    Y, R, split,
    Z,
    eigvals,
    extra_meta=None
):
    save_dir = os.path.join(outp, "figures", "Figure_KPCA", f"{run_tag_safe}_seed{seed}_fold{fold_id}")
    os.makedirs(save_dir, exist_ok=True)

    Z = np.asarray(Z, dtype=np.float32)
    eigvals = np.asarray(eigvals, dtype=np.float64).reshape(-1)

    denom = float(np.sum(eigvals)) if float(np.sum(eigvals)) > 0 else 1.0
    eig_ratio = eigvals / denom
    eig_cum = np.cumsum(eig_ratio)

    df = pd.DataFrame({
        "Sample": np.asarray(sample_ids),
        "Y": np.asarray(Y).astype(int),
        "R": np.asarray(R).astype(str),
        "split": np.asarray(split).astype(str),
        "KPC1": Z[:, 0] if Z.shape[1] > 0 else np.nan,
        "KPC2": Z[:, 1] if Z.shape[1] > 1 else np.nan,
        "KPC3": Z[:, 2] if Z.shape[1] > 2 else np.nan,
    })
    if extra_meta is not None:
        for k, v in dict(extra_meta).items():
            df[k] = v

    df.to_csv(os.path.join(save_dir, f"kpca_scores_{mode_name}.csv"), index=False)

    pd.DataFrame({
        "component": np.arange(1, len(eigvals) + 1),
        "eigenvalue": eigvals,
        "ratio": eig_ratio,
        "cum_ratio": eig_cum
    }).to_csv(os.path.join(save_dir, f"kpca_scree_{mode_name}.csv"), index=False)

    np.savez_compressed(
        os.path.join(save_dir, f"kpca_raw_{mode_name}.npz"),
        Z=Z, eigvals=eigvals, eig_ratio=eig_ratio, eig_cum=eig_cum
    )
    
    save_kpca_separation_metrics(
        Z=Z,
        Y=np.asarray(Y),
        R=np.asarray(R),
        save_path=os.path.join(save_dir, f"kpca_separation_{mode_name}.csv")
    )

def _ensure_knew_train(K, n_new, n_train, name="K"):
    K = np.asarray(K)
    if K.shape == (n_new, n_train):
        return K
    if K.shape == (n_train, n_new):
        return K.T
    raise ValueError(f"{name} shape must be ({n_new},{n_train}) or ({n_train},{n_new}); got {K.shape}")

def run_CCSA_transfer(
    seed,
    dataset,
    n_features,
    fold=3,
    alpha=ALPHA,
    learning_rate=0.01,
    hiddenLayers=None,
    dr=0.5,
    groups=("WHITE", "BLACK"),
    momentum=0.0,
    decay=0,
    batch_size=32,
    sample_per_class=2,
    repetition=5,
    K_precomputed=None,
    kpca_q=None,
    K_folds=None,
    K_folds_tt_psd=None,
    K_base_list=None,
    kernel_tags=None,
    mkl_smoothing=0.15,
    X_blocks_all=None,
    dag_pca_dims=None,
    dag_notears_lambda=0.1,
    dag_max_iter=30,
    dag_loss_type="l2",
    dag_A_block=None,
    dag_blocks_order=None,
    dag_random_state=0,
    dag_h_tol=None,
    dag_rho_max=None,
    dag_w_threshold=None,
    dag_transpose_W=False,
    proj_pairs=None,
    gamma_mults=(0.5, 1.0, 2.0),
    subsample_m=300,
    use_nystrom=False,
    nystrom_s=128,
    sample_ids=None,
    feat_names=None,
    T_days=None,
    C_flag=None,
    out_dir=None,
    run_tag=None,
    block_feature_names=None
):
    POS_NEG_RATIO = POS_NEG
    BALANCE_MODE = "overall"

    if hiddenLayers is None:
        hiddenLayers = [100]
    if proj_pairs is None:
        proj_pairs = [("miRNA", "mRNA"), ("methyl", "mRNA"), ("miRNA", "miRNA"), ("methyl", "methyl")]

    outp = out_dir or "outputs"
    os.makedirs(outp, exist_ok=True)

    run_tag_safe = (str(run_tag).strip() if run_tag is not None else "TL")
    tag_suffix = f"_{run_tag_safe}" if run_tag_safe else ""
    da_task = f"{groups[0]}_to_{groups[1]}"

    fig_root    = os.path.join(outp, "figures")
    fig7_dir    = os.path.join(fig_root, "Figure7")
    notears_dir = os.path.join(fig_root, "Figure_NOTEARS")
    mkl_dir     = os.path.join(fig_root, "Figure_MKL")
    score_dir   = os.path.join(fig_root, "Figure_Scores_KM")
    pairs_root  = os.path.join(fig_root, "Figure_CCSA_pairs")
    kpca_dir    = os.path.join(fig_root, "Figure_KPCA")
    for _d in [fig_root, fig7_dir, notears_dir, mkl_dir, score_dir, pairs_root, kpca_dir]:
        os.makedirs(_d, exist_ok=True)

    def psd_fix(K, mode="jitter", eps=1e-6):
        K = np.asarray(K, dtype=np.float64)
        K = 0.5 * (K + K.T)
        if mode == "jitter":
            return (K + float(eps) * np.eye(K.shape[0], dtype=np.float64))
        elif mode == "clip":
            w, V = np.linalg.eigh(K)
            w = np.maximum(w, float(eps))
            return (V * w) @ V.T
        else:
            raise ValueError("mode must be 'jitter' or 'clip'.")

    def _finalize_train_gram(Ktr):
        Ktr = np.asarray(Ktr, dtype=np.float64)
        Ktr = 0.5 * (Ktr + Ktr.T)
        return psd_fix(Ktr, mode="jitter", eps=1e-6)

    def _center_o_n2(K):
        K = np.asarray(K, dtype=np.float64)
        K = 0.5 * (K + K.T)
        row_m = K.mean(axis=1, keepdims=True)
        col_m = K.mean(axis=0, keepdims=True)
        grand = float(K.mean())
        return K - row_m - col_m + grand

    def _fro_norm(K, eps=1e-12):
        K = np.asarray(K, dtype=np.float64)
        return float(np.sqrt(np.sum(K * K) + eps))

    def _estimate_mkl_weights_centered_alignment(K_list_tt, y01, smoothing=0.15):
        y01 = np.asarray(y01).astype(np.float32).reshape(-1)
        m = len(K_list_tt)
        if m == 0:
            raise ValueError("K_list_tt is empty.")
        if len(np.unique(y01)) < 2:
            return np.ones(m, dtype=np.float64) / m

        y = 2.0 * y01 - 1.0
        Ky = np.outer(y, y).astype(np.float64)
        Ky_c = _center_o_n2(Ky)
        Ky_c = Ky_c / max(_fro_norm(Ky_c), 1e-12)

        scores = []
        for K in K_list_tt:
            Kc = _center_o_n2(K)
            Kc = Kc / max(_fro_norm(Kc), 1e-12)
            scores.append(float(np.sum(Kc * Ky_c)))

        scores = np.maximum(np.asarray(scores, dtype=np.float64), 0.0)
        w = (np.ones_like(scores) / len(scores)) if float(scores.sum()) <= 0.0 else (scores / float(scores.sum()))

        lam = float(0.15 if smoothing is None else smoothing)
        lam = min(max(lam, 0.0), 1.0)
        w = (1.0 - lam) * w + lam * (np.ones(m, dtype=np.float64) / m)
        w = w / float(w.sum())
        return w

    def _weighted_sum(mats, w):
        out = None
        for j, M in enumerate(mats):
            out = float(w[j]) * np.asarray(M) if out is None else (out + float(w[j]) * np.asarray(M))
        return out

    def _median_length_scale_subsample(X, max_m=300, rng=0):
        X = np.asarray(X, dtype=np.float32)
        n = X.shape[0]
        if n > max_m:
            idx = np.random.default_rng(rng).choice(n, size=max_m, replace=False)
            Xs = X[idx]
        else:
            Xs = X
        D2 = pairwise_distances(Xs, Xs, metric="sqeuclidean")
        return float(np.median(D2[D2 > 0])) if np.any(D2 > 0) else 1.0

    def _rbf_cross(A, B, gamma):
        return rbf_kernel(A, B, gamma=gamma).astype(np.float32, copy=False)

    X, Y, R, y_sub, y_strat = dataset
    X = np.asarray(X)
    Y = np.asarray(Y).astype(np.int32).reshape(-1)
    R = np.asarray(R).astype(str)
    n_total = len(Y)

    for name_, arr_ in [("sample_ids", sample_ids), ("T_days", T_days), ("C_flag", C_flag)]:
        if arr_ is not None and len(arr_) != n_total:
            raise ValueError(f"{name_} must have length N={n_total}, got {len(arr_)}")

    src_pos = np.where(R == str(groups[0]))[0].astype(int)
    tgt_pos = np.where(R == str(groups[1]))[0].astype(int)
    if len(src_pos) == 0 or len(tgt_pos) == 0:
        raise ValueError(f"Empty source/target after split. src={len(src_pos)}, tgt={len(tgt_pos)}")

    Y_train_source = Y[src_pos].astype(np.int32)
    X_train_source_raw = X[src_pos].astype(np.float32)

    X_src_raw0 = np.asarray(X_train_source_raw, dtype=np.float32).copy()
    Y_src0     = np.asarray(Y_train_source, dtype=np.int32).copy()

    Y_tgt_all = Y[tgt_pos].astype(np.int32)
    X_tgt_all = X[tgt_pos].astype(np.float32)

    if K_precomputed is not None:
        if not (isinstance(K_precomputed, np.ndarray) and K_precomputed.shape == (n_total, n_total)):
            raise ValueError(f"K_precomputed must be (N,N) with N={n_total}, got {getattr(K_precomputed, 'shape', None)}")

    if K_folds is not None:
        if not (isinstance(K_folds, list) and len(K_folds) >= fold):
            raise ValueError(f"K_folds must be list length >= fold({fold}).")
        for i, K in enumerate(K_folds[:fold]):
            if not (isinstance(K, np.ndarray) and K.shape == (n_total, n_total)):
                raise ValueError(f"K_folds[{i}] must be (N,N) with N={n_total}, got {getattr(K, 'shape', None)}")

    if K_base_list is not None:
        if not (isinstance(K_base_list, list) and len(K_base_list) > 0):
            raise ValueError("K_base_list must be a non-empty list of NxN kernels.")
        for j, K in enumerate(K_base_list):
            if not (isinstance(K, np.ndarray) and K.shape == (n_total, n_total)):
                raise ValueError(f"K_base_list[{j}] must be (N,N) with N={n_total}, got {getattr(K, 'shape', None)}")
        if kernel_tags is None:
            kernel_tags = [f"kernel_{j}" for j in range(len(K_base_list))]
        if len(kernel_tags) != len(K_base_list):
            raise ValueError("len(kernel_tags) must match len(K_base_list).")

    build_base_in_fold = (X_blocks_all is not None) and (dag_blocks_order is not None) and (dag_A_block is not None) and (proj_pairs is not None)
    if build_base_in_fold:
        if ("dag_feature_projection_blocks_fit" not in globals()) or ("dag_feature_projection_blocks_transform" not in globals()):
            raise RuntimeError("fold-internal DAG path requires dag_feature_projection_blocks_fit/transform in globals().")
        if not isinstance(X_blocks_all, dict):
            raise TypeError("X_blocks_all must be a dict/OrderedDict aligned with dataset row-order.")
        X_blocks_all = OrderedDict(X_blocks_all)
        for k_blk, v_blk in X_blocks_all.items():
            if np.asarray(v_blk).shape[0] != n_total:
                raise ValueError(f"X_blocks_all['{k_blk}'] first dim must be N={n_total}.")

    kf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=int(seed))
    df_score_parts = []
    paired_metric_rows = []
    weights_folds = []
    tags_template = None

    use_legacy_kernel = (K_base_list is not None) or (K_folds is not None) or (K_precomputed is not None)

    for fold_id, (train_index, test_index) in enumerate(kf.split(X_tgt_all, Y_tgt_all), 1):
        seed_fold = int(seed) * 1000 + int(fold_id)

        tgt_train_full_pos = tgt_pos[train_index]
        tgt_test_pos       = tgt_pos[test_index]

        X_train_target_full = X_tgt_all[train_index]
        X_test_target_raw   = X_tgt_all[test_index]
        Y_train_target_full = Y_tgt_all[train_index]
        Y_test_target       = Y_tgt_all[test_index]
        Y_train_source = Y_src0

        if (not build_base_in_fold) and (not use_legacy_kernel) and (n_features > 0 and n_features < X_train_target_full.shape[1]):
            X_fs_train = np.vstack([X_src_raw0, X_train_target_full]).astype(np.float32)
            y_fs_train = np.concatenate([Y_src0, Y_train_target_full]).astype(int)

            _, _, meta_fs, _ = get_k_best(
                X_fs_train, y_fs_train,
                X_fs_train,
                k_num=n_features,
                pre_var_thresh=0.0,
                method="mutual_info",
                random_state=seed_fold
            )

            X_train_source_raw  = apply_feature_selector(X_src_raw0, meta_fs).astype(np.float32, copy=False)
            X_train_target_full = apply_feature_selector(X_train_target_full, meta_fs).astype(np.float32, copy=False)
            X_test_target_raw   = apply_feature_selector(X_test_target_raw, meta_fs).astype(np.float32, copy=False)
        else:
            X_train_source_raw = X_src_raw0.copy()

        if not build_base_in_fold:
            assert X_train_source_raw.shape[1] == X_train_target_full.shape[1] == X_test_target_raw.shape[1]

        idx0 = np.where(Y_train_target_full == 0)[0]
        idx1 = np.where(Y_train_target_full == 1)[0]
        take = min(sample_per_class, len(idx0), len(idx1))
        if take == 0:
            print(f"[WARN] fold {fold_id}: one class missing. Skipping fold.")
            continue

        rng = np.random.RandomState(seed_fold)
        sel0 = rng.choice(idx0, size=take, replace=False)
        sel1 = rng.choice(idx1, size=take, replace=False)
        target_samples_rel = np.concatenate([sel0, sel1])

        mask_val = np.ones(len(Y_train_target_full), dtype=bool)
        mask_val[target_samples_rel] = False
        if mask_val.sum() == 0:
            print(f"[WARN] fold {fold_id}: validation empty. Skipping fold.")
            continue

        tgt_sel_pos = tgt_train_full_pos[target_samples_rel]
        tgt_val_pos = tgt_train_full_pos[mask_val]

        X_train_target = X_train_target_full[target_samples_rel]
        Y_train_target = Y_train_target_full[target_samples_rel].astype(np.int32)

        X_val_target = X_train_target_full[mask_val]
        Y_val_target = Y_train_target_full[mask_val].astype(np.int32)

        use_kernel = build_base_in_fold or use_legacy_kernel

        if use_kernel:
            train_union_pos = np.concatenate([src_pos, tgt_sel_pos]).astype(int)
            y_union = np.concatenate([Y_train_source, Y_train_target]).astype(np.int32)

            if build_base_in_fold:
                _validate_A_block_and_proj_pairs(
                    A_block=dag_A_block,
                    blocks_order=dag_blocks_order,
                    proj_pairs=proj_pairs,
                    strict=True
                )

                X_blocks_tr = OrderedDict({k: np.asarray(v)[train_union_pos, :] for k, v in X_blocks_all.items()})

                fitted = dag_feature_projection_blocks_fit(
                    X_blocks_tr,
                    pca_dims=dag_pca_dims,
                    notears_lambda=dag_notears_lambda,
                    max_iter=dag_max_iter,
                    loss_type=dag_loss_type,
                    A_block=dag_A_block,
                    blocks_order=dag_blocks_order,
                    random_state=dag_random_state,
                    h_tol=dag_h_tol,
                    rho_max=dag_rho_max,
                    w_threshold=dag_w_threshold,
                    transpose_W=dag_transpose_W,
                )

                USE_WEIGHTED_RBF = True
                AZ_POST_THRESHOLD = True
                AX_CLIP_Q = 0.99
                AX_TAU = 2.0
                AX_SMOOTH_UNIFORM = 0.05
                EPS_W = 1e-12

                a_x_cache = {}
                a_z_cache = {}

                a_z = compute_latent_importance_from_Wz(
                    fitted["Wz"],
                    mode="out_in",
                    normalize="mean1"
                )

                report_q_dims(fitted, print_table=True)

                export_pca_top_loadings(
                    fitted,
                    block_feature_names=block_feature_names,
                    top_k=20,
                    out_dir=fig7_dir,
                    prefix=f"{run_tag_safe}_seed{seed}_fold{fold_id}_pca_loadings"
                )

                if out_dir is not None:
                    _save_notears_fold_artifacts(
                        fitted=fitted,
                        outp=notears_dir,
                        seed=seed,
                        fold_id=fold_id,
                        tag_suffix=tag_suffix,
                        blocks_order=dag_blocks_order,
                        A_block=dag_A_block,
                        w_threshold=dag_w_threshold
                    )

                tr_out = dag_feature_projection_blocks_transform(
                    X_blocks_tr, fitted, A_block=dag_A_block, blocks_order=dag_blocks_order
                )
                projs_tr_all = tr_out["projs"]

                X_blocks_val = OrderedDict({k: np.asarray(v)[tgt_val_pos, :] for k, v in X_blocks_all.items()})
                val_out = dag_feature_projection_blocks_transform(
                    X_blocks_val, fitted, A_block=dag_A_block, blocks_order=dag_blocks_order
                )
                projs_val_all = val_out["projs"]

                X_blocks_te = OrderedDict({k: np.asarray(v)[tgt_test_pos, :] for k, v in X_blocks_all.items()})
                te_out = dag_feature_projection_blocks_transform(
                    X_blocks_te, fitted, A_block=dag_A_block, blocks_order=dag_blocks_order
                )
                projs_te_all = te_out["projs"]

                Ktt_list, Kval_list, Kte_list = [], [], []
                tags_local = []

                for p_i, (s_blk, t_blk) in enumerate(proj_pairs):
                    key = (s_blk, t_blk)
                    if key not in projs_tr_all:
                        raise KeyError(f"Projection key {key} not found in projs_tr_all.")

                    Xtr = np.asarray(projs_tr_all[key], dtype=np.float32)
                    Xva = np.asarray(projs_val_all[key], dtype=np.float32)
                    Xte = np.asarray(projs_te_all[key], dtype=np.float32)

                    if USE_WEIGHTED_RBF:
                        if t_blk not in a_x_cache:
                            a_z_t = _compute_a_z_block_from_Wz(
                                fitted, block_name=t_blk, use_post_threshold=AZ_POST_THRESHOLD, eps=EPS_W
                            )
                            a_z_cache[t_blk] = a_z_t

                            a_x_t = _compute_a_x_for_target_block(
                                fitted, tgt_block=t_blk, a_z_t=a_z_t,
                                tau=AX_TAU, clip_q=AX_CLIP_Q, smooth_to_uniform=AX_SMOOTH_UNIFORM, eps=EPS_W
                            )
                            a_x_cache[t_blk] = a_x_t

                        a_x_t = a_x_cache[t_blk]
                        sqrt_ax = np.sqrt(np.maximum(a_x_t, 0.0)).astype(np.float32)

                        Xtr_w = Xtr * sqrt_ax[None, :]
                        Xva_w = Xva * sqrt_ax[None, :]
                        Xte_w = Xte * sqrt_ax[None, :]
                    else:
                        Xtr_w, Xva_w, Xte_w = Xtr, Xva, Xte

                    ell2 = _median_length_scale_subsample(Xtr_w, max_m=subsample_m, rng=seed_fold + p_i)
                    ell2 = max(ell2, 1e-12)
                    base_gamma = 1.0 / (2.0 * ell2)

                    for mult in gamma_mults:
                        gamma = float(base_gamma * float(mult))
                        Ktt = _rbf_cross(Xtr_w, Xtr_w, gamma=gamma)
                        Kva = _rbf_cross(Xva_w, Xtr_w, gamma=gamma)
                        Kte = _rbf_cross(Xte_w, Xtr_w, gamma=gamma)
                        Ktt = (Ktt + Ktt.T) * 0.5

                        Ktt_list.append(Ktt)
                        Kval_list.append(Kva)
                        Kte_list.append(Kte)
                        tags_local.append(f"{s_blk}->{t_blk}__g{float(mult):g}")

                tags_template = tags_template or list(tags_local)

                n_src = len(src_pos)
                Ktt_list_src = [K[:n_src, :n_src] for K in Ktt_list]
                y_mkl = Y_train_source
                w = _estimate_mkl_weights_centered_alignment(Ktt_list_src, y_mkl, smoothing=mkl_smoothing)
                weights_folds.append(w.copy())

                Ktr  = _weighted_sum(Ktt_list, w)
                Kval = _weighted_sum(Kval_list, w)
                Kte  = _weighted_sum(Kte_list,  w)

                Ktr = _finalize_train_gram(Ktr)

                if out_dir is not None:
                    pd.DataFrame({"kernel_tag": tags_local, "weight": w}).to_csv(
                        os.path.join(mkl_dir, f"mkl_weights_seed{seed}_fold{fold_id}{tag_suffix}.csv"),
                        index=False
                    )

            else:
                train_union_pos = np.concatenate([src_pos, tgt_sel_pos]).astype(int)
                y_union = np.concatenate([Y_train_source, Y_train_target]).astype(np.int32)

                if K_base_list is not None:
                    Ktr_list = [K[np.ix_(train_union_pos, train_union_pos)] for K in K_base_list]
                    n_src = len(src_pos)
                    Ktr_list_src = [Kuu[:n_src, :n_src] for Kuu in Ktr_list]
                    y_mkl = Y_train_source
                    w = _estimate_mkl_weights_centered_alignment(Ktr_list_src, y_mkl, smoothing=mkl_smoothing)
                    weights_folds.append(w.copy())

                    Ktr  = _weighted_sum(Ktr_list, w)
                    Kval = _weighted_sum([K[np.ix_(tgt_val_pos,  train_union_pos)] for K in K_base_list], w)
                    Kte  = _weighted_sum([K[np.ix_(tgt_test_pos, train_union_pos)] for K in K_base_list], w)

                    Ktr = _finalize_train_gram(Ktr)

                    if out_dir is not None:
                        pd.DataFrame({"kernel_tag": kernel_tags, "weight": w}).to_csv(
                            os.path.join(mkl_dir, f"mkl_weights_seed{seed}_fold{fold_id}{tag_suffix}.csv"),
                            index=False
                        )
                else:
                    K_use = K_folds[fold_id - 1] if K_folds is not None else K_precomputed
                    Ktr = K_use[np.ix_(train_union_pos, train_union_pos)]
                    if K_folds_tt_psd is not None and (fold_id - 1) < len(K_folds_tt_psd):
                        Ktr_cand = K_folds_tt_psd[fold_id - 1]
                        if Ktr_cand.shape == Ktr.shape:
                            Ktr = Ktr_cand
                    Ktr = _finalize_train_gram(Ktr)
                    Kval = K_use[np.ix_(tgt_val_pos,  train_union_pos)]
                    Kte  = K_use[np.ix_(tgt_test_pos, train_union_pos)]

            lam_all = np.linalg.eigvalsh((Ktr + Ktr.T) * 0.5)
            lam_all_sorted = lam_all[::-1].clip(min=0)
            cum = np.cumsum(lam_all_sorted)
            ratio = cum / max(cum[-1], 1e-12)
            q_auto = int(np.searchsorted(ratio, 0.90) + 1)
            q_cap = (kpca_q if kpca_q is not None else 200)
            q = max(1, min(q_auto, q_cap, Ktr.shape[0] - 1))

            kpca = KernelPCA(n_components=q, kernel="precomputed")
            Ztr = kpca.fit_transform(Ktr)

            n_train_union = Ktr.shape[0]
            n_val = Kval.shape[0] if Kval.shape[0] != n_train_union else Kval.shape[1]
            n_te  = Kte.shape[0]  if Kte.shape[0]  != n_train_union else Kte.shape[1]

            Kval = _ensure_knew_train(Kval, n_val, n_train_union, name="Kval")
            Kte  = _ensure_knew_train(Kte,  n_te,  n_train_union, name="Kte")

            Kval = np.asarray(Kval, dtype=np.float64)
            Kte  = np.asarray(Kte,  dtype=np.float64)

            n_src = len(src_pos)
            Z_src = Ztr[:n_src, :]
            Z_tgt_tr = Ztr[n_src:, :]
            Z_val = kpca.transform(Kval)
            Z_te  = kpca.transform(Kte)

            if out_dir is not None:
                union_ids = (np.asarray(sample_ids)[train_union_pos] if sample_ids is not None else train_union_pos)
                union_R = np.asarray(R)[train_union_pos]
                union_Y = np.asarray(y_union)

                split_union = np.array(["train"] * len(train_union_pos))

                val_ids = (np.asarray(sample_ids)[tgt_val_pos] if sample_ids is not None else tgt_val_pos)
                te_ids  = (np.asarray(sample_ids)[tgt_test_pos] if sample_ids is not None else tgt_test_pos)

                val_R = np.asarray(R)[tgt_val_pos]
                te_R  = np.asarray(R)[tgt_test_pos]

                split_val = np.array(["val"] * len(tgt_val_pos))
                split_te  = np.array(["test"] * len(tgt_test_pos))

                Z_all = np.vstack([Ztr, Z_val, Z_te])
                ids_all = np.concatenate([union_ids, val_ids, te_ids])
                Y_all = np.concatenate([union_Y, Y_val_target, Y_test_target])
                R_all = np.concatenate([union_R, val_R, te_R])
                split_all = np.concatenate([split_union, split_val, split_te])

                eigvals = getattr(kpca, "eigenvalues_", None)
                if eigvals is None:
                    eigvals = getattr(kpca, "lambdas_", None)
                if eigvals is None:
                    eigvals = lam_all_sorted[:q]

                _save_kpca_fold_artifacts(
                    outp=outp,
                    run_tag_safe=run_tag_safe, seed=seed, fold_id=fold_id,
                    mode_name="integrated",
                    sample_ids=ids_all, Y=Y_all, R=R_all, split=split_all,
                    Z=Z_all, eigvals=eigvals,
                    extra_meta={"seed_fold": seed_fold, "q": q}
                )

            X_train_source_fold = Z_src.astype(np.float32, copy=False)
            X_train_target_fold = Z_tgt_tr.astype(np.float32, copy=False)
            X_val_target_fold   = Z_val.astype(np.float32, copy=False)
            X_test_target_fold  = Z_te.astype(np.float32, copy=False)

        else:
            X_union = np.vstack([X_train_source_raw, X_train_target]).astype(np.float32)
            n_union, p = X_union.shape
            q_cap = (kpca_q if kpca_q is not None else 200)
            q = max(1, min(q_cap, n_union - 1, p))

            if q >= 2:
                pca = PCA(n_components=q, svd_solver="auto", random_state=seed_fold)
                Z_union = pca.fit_transform(X_union)
                Z_src = Z_union[:len(Y_train_source), :]
                Z_tgt = Z_union[len(Y_train_source):, :]

                X_train_source_fold = Z_src.astype(np.float32, copy=False)
                X_train_target_fold = Z_tgt.astype(np.float32, copy=False)
                X_val_target_fold   = pca.transform(X_val_target).astype(np.float32, copy=False)
                X_test_target_fold  = pca.transform(X_test_target_raw).astype(np.float32, copy=False)
            else:
                X_train_source_fold = X_train_source_raw
                X_train_target_fold = X_train_target
                X_val_target_fold   = X_val_target
                X_test_target_fold  = X_test_target_raw

        feat_dim = int(np.asarray(X_train_source_fold).shape[1])

        pairs_dir = os.path.join(pairs_root, f"{run_tag_safe}_seed{seed}_fold{fold_id}")
        os.makedirs(pairs_dir, exist_ok=True)

        best_score, fold_auc, best_thr, pred_thr = train_and_predict(
            np.asarray(X_train_target_fold, dtype=np.float32), np.asarray(Y_train_target, dtype=np.int32),
            np.asarray(X_train_source_fold, dtype=np.float32), np.asarray(Y_train_source, dtype=np.int32),
            np.asarray(X_val_target_fold, dtype=np.float32),   np.asarray(Y_val_target, dtype=np.int32),
            np.asarray(X_test_target_fold, dtype=np.float32),  np.asarray(Y_test_target, dtype=np.int32),
            repetition=repetition,
            sample_per_class=sample_per_class,
            alpha=alpha,
            learning_rate=learning_rate,
            hiddenLayers=hiddenLayers,
            dr=dr,
            decay=decay,
            momentum=momentum,
            batch_size=batch_size,
            n_features=feat_dim,
            domain_adaptation_task=da_task,
            seed=int(seed_fold),
            pairs_dir=pairs_dir,
            apply_domain_smote=False,
            balance_mode=str(BALANCE_MODE),
            pos_neg_ratio=float(POS_NEG_RATIO),
            smote_k_max=5,
        )

        best_score = np.asarray(best_score, dtype=float).reshape(-1)
        pred_thr = np.asarray(pred_thr).astype(int).reshape(-1)
        pred_05 = (best_score > 0.5).astype(int)
        risk = 1.0 - best_score

        tgt_test_pos = np.asarray(tgt_test_pos, dtype=int)

        T_test = (np.asarray(T_days)[tgt_test_pos] if T_days is not None
                  else np.full(len(tgt_test_pos), np.nan))

        if C_flag is not None:
            E_all = (1 - np.asarray(C_flag).astype(int)).reshape(-1)
            E_test = E_all[tgt_test_pos]
        else:
            E_test = np.full(len(tgt_test_pos), np.nan)

        sid_test = (np.asarray(sample_ids)[tgt_test_pos] if sample_ids is not None else tgt_test_pos)

        df_fold = pd.DataFrame({
            "scr": best_score,
            "risk": risk,
            "pred_thr": pred_thr,
            "best_thr": float(best_thr),
            "pred_05": pred_05,
            "Y": np.asarray(Y_test_target).reshape(-1).astype(int),
            "T": T_test,
            "E": E_test,
            "fold": int(fold_id),
            "R": np.asarray(R)[tgt_test_pos],
            "sample_id": sid_test,
            "seed": int(seed),
            "method": str(run_tag_safe)
        }, index=tgt_test_pos)

        global_idx = tgt_test_pos
        sid = (np.asarray(sample_ids)[global_idx] if sample_ids is not None else global_idx)
        T_fold = (np.asarray(T_days)[global_idx] if T_days is not None else np.full(len(global_idx), np.nan))
        C_fold = (np.asarray(C_flag)[global_idx] if C_flag is not None else np.full(len(global_idx), np.nan))
        Event = (1 - C_fold) if C_flag is not None else np.full(len(global_idx), np.nan)

        X_test_raw = X_test_target_raw
        cols = (
            list(feat_names)
            if (feat_names is not None and len(feat_names) == X_test_raw.shape[1])
            else [f"feat_{i}" for i in range(X_test_raw.shape[1])]
        )
        df_feat = pd.DataFrame(X_test_raw, columns=cols)
        df_feat.insert(0, "Sample", sid)
        df_feat.insert(1, "score", best_score)

        save_subgroup_fold_metrics(
            df_fold=df_fold,
            save_path=os.path.join(
                score_dir,
                f"subgroup_fold_metrics_seed{seed}_fold{fold_id}{tag_suffix}.csv"
            ),
            seed=seed,
            fold_id=fold_id,
            method_tag=run_tag_safe
        )

        save_calibration_outputs(
            df_fold=df_fold,
            save_points_path=os.path.join(
                score_dir,
                f"calibration_points_seed{seed}_fold{fold_id}{tag_suffix}.csv"
            ),
            save_summary_path=os.path.join(
                score_dir,
                f"calibration_summary_seed{seed}_fold{fold_id}{tag_suffix}.csv"
            ),
            n_bins=10
        )

        save_dca_outputs(
            df_fold=df_fold,
            save_path=os.path.join(
                score_dir,
                f"dca_seed{seed}_fold{fold_id}{tag_suffix}.csv"
            ),
            thresholds=np.arange(0.05, 0.81, 0.01)
        )

        df_fold.to_csv(
            os.path.join(score_dir, f"df_score_seed{seed}_fold{fold_id}{tag_suffix}.csv"),
            index=True
        )
        pd.DataFrame({
            "Sample": sid, "risk": 1.0 - best_score, "Y": np.asarray(Y_test_target).reshape(-1),
            "T_days": T_fold, "Event": Event, "R": np.asarray(R)[global_idx],
        }).to_csv(
            os.path.join(score_dir, f"km_input_seed{seed}_fold{fold_id}{tag_suffix}.csv"),
            index=False
        )
        df_feat.to_csv(
            os.path.join(score_dir, f"feature_score_test_seed{seed}_fold{fold_id}{tag_suffix}.csv"),
            index=False
        )

        paired_metric_rows.append(make_method_summary_row(
            df_score=df_fold,
            seed=seed,
            method_tag=run_tag_safe,
            lr=learning_rate,
            dr=dr,
            alpha=alpha,
            a_block_type="proposed" if dag_A_block is not None else "none",
            integration_type="NOTEARS_MKL" if use_kernel else "PCA_only",
        ))

        df_score_parts.append(df_fold)
        
    if len(df_score_parts) == 0:
        df_res = pd.DataFrame({
            "folds": fold,
            "TL_auc": np.nan,
            "TL_acc": np.nan,
            "TL_cindex": np.nan,
            "TL_f1": np.nan,
            "TL_prauc": np.nan
        }, index=[seed])
        return df_res, np.array([])

    df_score = pd.concat(df_score_parts, axis=0)
    df_score = df_score[~df_score.index.duplicated(keep="last")].sort_index()

    thr_summary = df_score.groupby("fold")["best_thr"].first().reset_index()

    thr_stab = summarize_threshold_stability(thr_summary, seed=seed, method_tag=run_tag_safe)
    thr_stab.to_csv(
        os.path.join(score_dir, f"threshold_stability_seed{seed}{tag_suffix}.csv"),
        index=False
    )

    df_score.to_csv(
        os.path.join(score_dir, f"df_score_seed{seed}{tag_suffix}.csv"),
        index=True
    )
    thr_summary.to_csv(
        os.path.join(score_dir, f"thresholds_seed{seed}{tag_suffix}.csv"),
        index=False
    )

    if len(paired_metric_rows) > 0:
        pd.DataFrame(paired_metric_rows).to_csv(
            os.path.join(score_dir, f"paired_fold_metrics_seed{seed}{tag_suffix}.csv"),
            index=False
    )

    y_true = df_score["Y"].astype(int).values
    y_scr  = df_score["scr"].astype(float).values
    y_pred = df_score["pred_thr"].astype(int).values
    
    roc_auc = roc_auc_score(y_true, y_scr) if len(np.unique(y_true)) == 2 else np.nan
    pr_auc  = average_precision_score(y_true, y_scr) if len(np.unique(y_true)) == 2 else np.nan
    acc = accuracy_score(y_true, y_pred)
    f1v = f1_score(y_true, y_pred, average="weighted")

    df_cidx = df_score.dropna(subset=["T", "E", "risk"]).copy()
    if len(df_cidx) > 0:
        df_cidx["E"] = df_cidx["E"].astype(int)

    if (len(df_cidx) < 2) or (df_cidx["E"].sum() == 0) or (df_cidx["T"].nunique() < 2):
        cindex = np.nan
    else:
        cindex = concordance_index_simple(
            T=df_cidx["T"].values,
            risk=df_cidx["risk"].values,
            E=df_cidx["E"].values
        )

    df_res = pd.DataFrame({
        "folds": fold,
        "TL_auc": roc_auc,
        "TL_acc": acc,
        "TL_cindex": cindex,
        "TL_f1": f1v,
        "TL_prauc": pr_auc
    }, index=[seed])

    if (len(weights_folds) > 0) and (out_dir is not None):
        W = np.vstack(weights_folds)
        tags_for_summary = (
            tags_template if (build_base_in_fold and tags_template is not None)
            else (kernel_tags if kernel_tags is not None else [f"k{j}" for j in range(W.shape[1])])
        )
        df_mkl_sum = pd.DataFrame({
            "kernel_tag": tags_for_summary,
            "weight": W.mean(axis=0),
            "w_std":  W.std(axis=0)
        })
        df_mkl_sum.to_csv(
            os.path.join(mkl_dir, f"mkl_weights_seed{seed}_summary{tag_suffix}.csv"),
            index=False
        )

        df_mkl_cat = summarize_mkl_weights_by_category(df_mkl_sum.rename(columns={"weight": "weight"}))
        df_mkl_cat.to_csv(
            os.path.join(mkl_dir, f"mkl_weight_by_category_seed{seed}_summary{tag_suffix}.csv"),
            index=False
        )

    score_vec_target_order = df_score["scr"].reindex(tgt_pos).to_numpy(dtype=float)
    return df_res, score_vec_target_order

def run_CCSA_transfer_SMOTE(
    seed,
    dataset,
    n_features,
    fold=3,
    alpha=ALPHA,
    learning_rate=0.01,
    hiddenLayers=None,
    dr=0.5,
    groups=("WHITE", "BLACK"),
    momentum=0.0,
    decay=0,
    batch_size=32,
    sample_per_class=2,
    repetition=5,
    K_precomputed=None,
    kpca_q=None,
    K_folds_tt_psd=None,
    K_folds=None,
    K_base_list=None,
    kernel_tags=None,
    mkl_smoothing=0.15,
    X_blocks_all=None,
    dag_pca_dims=None,
    dag_notears_lambda=0.1,
    dag_max_iter=30,
    dag_loss_type="l2",
    dag_A_block=None,
    dag_blocks_order=None,
    dag_random_state=0,
    dag_h_tol=None,
    dag_rho_max=None,
    dag_w_threshold=None,
    dag_transpose_W=False,
    proj_pairs=None,
    gamma_mults=(0.5, 1.0, 2.0),
    subsample_m=300,
    use_nystrom=False,
    nystrom_s=128,
    sample_ids=None,
    feat_names=None,
    T_days=None,
    C_flag=None,
    out_dir=None,
    run_tag=None,
    block_feature_names=None
):
    POS_NEG_RATIO = POS_NEG
    BALANCE_MODE  = "overall"

    if hiddenLayers is None:
        hiddenLayers = [100]
    if proj_pairs is None:
        proj_pairs = [("miRNA", "mRNA"), ("methyl", "mRNA"), ("miRNA", "miRNA"), ("methyl", "methyl")]

    outp = out_dir or "outputs"
    os.makedirs(outp, exist_ok=True)

    run_tag_safe = (str(run_tag).strip() if run_tag is not None else "TLDA")
    tag_suffix = f"_{run_tag_safe}" if run_tag_safe else ""
    da_task = f"{groups[0]}_to_{groups[1]}"

    fig_root = os.path.join(outp, "figures")  
    fig7_dir = os.path.join(fig_root, "Figure7")  
    notears_dir = os.path.join(fig_root, "Figure_NOTEARS")
    mkl_dir     = os.path.join(fig_root, "Figure_MKL")
    score_dir   = os.path.join(fig_root, "Figure_Scores_KM")
    pairs_root  = os.path.join(fig_root, "Figure_CCSA_pairs")
    kpca_dir    = os.path.join(fig_root, "Figure_KPCA")
    for _d in [fig_root, fig7_dir, notears_dir, mkl_dir, score_dir, pairs_root, kpca_dir]:
        os.makedirs(_d, exist_ok=True)

    def psd_fix(K, mode="jitter", eps=1e-6):
        K = np.asarray(K, dtype=np.float64)
        K = 0.5 * (K + K.T)
        if mode == "jitter":
            return (K + float(eps) * np.eye(K.shape[0], dtype=np.float64))
        elif mode == "clip":
            w, V = np.linalg.eigh(K)
            w = np.maximum(w, float(eps))
            return (V * w) @ V.T
        else:
            raise ValueError("mode must be 'jitter' or 'clip'.")

    def _finalize_train_gram(Ktr):
        Ktr = np.asarray(Ktr, dtype=np.float64)
        Ktr = 0.5 * (Ktr + Ktr.T)
        return psd_fix(Ktr, mode="jitter", eps=1e-6)

    def _center_o_n2(K):
        K = np.asarray(K, dtype=np.float64)
        K = 0.5 * (K + K.T)
        row_m = K.mean(axis=1, keepdims=True)
        col_m = K.mean(axis=0, keepdims=True)
        grand = float(K.mean())
        return K - row_m - col_m + grand

    def _fro_norm(K, eps=1e-12):
        K = np.asarray(K, dtype=np.float64)
        return float(np.sqrt(np.sum(K * K) + eps))

    def _estimate_mkl_weights_centered_alignment(K_list_tt, y01, smoothing=0.15):
        y01 = np.asarray(y01).astype(np.float32).reshape(-1)
        m = len(K_list_tt)
        if m == 0:
            raise ValueError("K_list_tt is empty.")
        if len(np.unique(y01)) < 2:
            return np.ones(m, dtype=np.float64) / m

        y = 2.0 * y01 - 1.0
        Ky = np.outer(y, y).astype(np.float64)
        Ky_c = _center_o_n2(Ky)
        Ky_c = Ky_c / max(_fro_norm(Ky_c), 1e-12)

        scores = []
        for K in K_list_tt:
            Kc = _center_o_n2(K)
            Kc = Kc / max(_fro_norm(Kc), 1e-12)
            scores.append(float(np.sum(Kc * Ky_c)))

        scores = np.maximum(np.asarray(scores, dtype=np.float64), 0.0)
        w = (np.ones_like(scores) / len(scores)) if float(scores.sum()) <= 0.0 else (scores / float(scores.sum()))

        lam = float(0.15 if smoothing is None else smoothing)
        lam = min(max(lam, 0.0), 1.0)
        w = (1.0 - lam) * w + lam * (np.ones(m, dtype=np.float64) / m)
        w = w / float(w.sum())
        return w

    def _weighted_sum(mats, w):
        out = None
        for j, M in enumerate(mats):
            out = float(w[j]) * np.asarray(M) if out is None else (out + float(w[j]) * np.asarray(M))
        return out

    def _median_length_scale_subsample(X, max_m=300, rng=0):
        X = np.asarray(X, dtype=np.float32)
        n = X.shape[0]
        if n > max_m:
            idx = np.random.default_rng(rng).choice(n, size=max_m, replace=False)
            Xs = X[idx]
        else:
            Xs = X
        D2 = pairwise_distances(Xs, Xs, metric="sqeuclidean")
        return float(np.median(D2[D2 > 0])) if np.any(D2 > 0) else 1.0

    def _rbf_cross(A, B, gamma):
        return rbf_kernel(A, B, gamma=gamma).astype(np.float32, copy=False)

    X, Y, R, y_sub, y_strat = dataset
    X = np.asarray(X)
    Y = np.asarray(Y).astype(np.int32).reshape(-1)
    R = np.asarray(R).astype(str)
    n_total = len(Y)

    for name_, arr_ in [("sample_ids", sample_ids), ("T_days", T_days), ("C_flag", C_flag)]:
        if arr_ is not None and len(arr_) != n_total:
            raise ValueError(f"{name_} must have length N={n_total}, got {len(arr_)}")

    src_pos = np.where(R == str(groups[0]))[0].astype(int)
    tgt_pos = np.where(R == str(groups[1]))[0].astype(int)
    if len(src_pos) == 0 or len(tgt_pos) == 0:
        raise ValueError(f"Empty source/target after split. src={len(src_pos)}, tgt={len(tgt_pos)}")

    Y_train_source = Y[src_pos].astype(np.int32)
    X_train_source_raw = X[src_pos].astype(np.float32)

    X_src_raw0 = np.asarray(X_train_source_raw, dtype=np.float32).copy()
    Y_src0     = np.asarray(Y_train_source, dtype=np.int32).copy()

    Y_tgt_all = Y[tgt_pos].astype(np.int32)
    X_tgt_all = X[tgt_pos].astype(np.float32)

    if K_precomputed is not None:
        if not (isinstance(K_precomputed, np.ndarray) and K_precomputed.shape == (n_total, n_total)):
            raise ValueError(f"K_precomputed must be (N,N) with N={n_total}, got {getattr(K_precomputed,'shape',None)}")

    if K_folds is not None:
        if not (isinstance(K_folds, list) and len(K_folds) >= fold):
            raise ValueError(f"K_folds must be list length >= fold({fold}).")
        for i, K in enumerate(K_folds[:fold]):
            if not (isinstance(K, np.ndarray) and K.shape == (n_total, n_total)):
                raise ValueError(f"K_folds[{i}] must be (N,N) with N={n_total}, got {getattr(K,'shape',None)}")

    if K_base_list is not None:
        if not (isinstance(K_base_list, list) and len(K_base_list) > 0):
            raise ValueError("K_base_list must be a non-empty list of NxN kernels.")
        for j, K in enumerate(K_base_list):
            if not (isinstance(K, np.ndarray) and K.shape == (n_total, n_total)):
                raise ValueError(f"K_base_list[{j}] must be (N,N) with N={n_total}, got {getattr(K,'shape',None)}")
        if kernel_tags is None:
            kernel_tags = [f"kernel_{j}" for j in range(len(K_base_list))]
        if len(kernel_tags) != len(K_base_list):
            raise ValueError("len(kernel_tags) must match len(K_base_list).")

    build_base_in_fold = (X_blocks_all is not None) and (dag_blocks_order is not None) and (dag_A_block is not None) and (proj_pairs is not None)
    if build_base_in_fold:
        if ("dag_feature_projection_blocks_fit" not in globals()) or ("dag_feature_projection_blocks_transform" not in globals()):
            raise RuntimeError("fold-internal DAG path requires dag_feature_projection_blocks_fit/transform in globals().")
        if not isinstance(X_blocks_all, dict):
            raise TypeError("X_blocks_all must be dict aligned with dataset row-order.")
        X_blocks_all = OrderedDict(X_blocks_all)
        for k_blk, v_blk in X_blocks_all.items():
            if np.asarray(v_blk).shape[0] != n_total:
                raise ValueError(f"X_blocks_all['{k_blk}'] first dim must be N={n_total}.")

    kf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=int(seed))
    df_score_parts = []
    paired_metric_rows = []
    weights_folds = []
    tags_template = None

    use_legacy_kernel = (K_base_list is not None) or (K_folds is not None) or (K_precomputed is not None)

    for fold_id, (train_index, test_index) in enumerate(kf.split(X_tgt_all, Y_tgt_all), 1):
        seed_fold = int(seed) * 1000 + int(fold_id)

        tgt_train_full_pos = tgt_pos[train_index]
        tgt_test_pos       = tgt_pos[test_index]

        X_train_target_full = X_tgt_all[train_index]
        X_test_target_raw   = X_tgt_all[test_index]
        Y_train_target_full = Y_tgt_all[train_index]
        Y_test_target       = Y_tgt_all[test_index]
        Y_train_source = Y_src0

        if (not build_base_in_fold) and (not use_legacy_kernel) and (n_features > 0 and n_features < X_train_target_full.shape[1]):
            X_fs_train = np.vstack([X_src_raw0, X_train_target_full]).astype(np.float32)
            y_fs_train = np.concatenate([Y_src0, Y_train_target_full]).astype(int)

            _, _, meta_fs, _ = get_k_best(
                X_fs_train, y_fs_train,
                X_fs_train,
                k_num=n_features,
                pre_var_thresh=0.0,
                method="mutual_info",
                random_state=seed_fold
            )

            X_train_source_raw  = apply_feature_selector(X_src_raw0, meta_fs).astype(np.float32, copy=False)
            X_train_target_full = apply_feature_selector(X_train_target_full, meta_fs).astype(np.float32, copy=False)
            X_test_target_raw   = apply_feature_selector(X_test_target_raw, meta_fs).astype(np.float32, copy=False)
        else:
            X_train_source_raw = X_src_raw0.copy()

        if not build_base_in_fold:
            assert X_train_source_raw.shape[1] == X_train_target_full.shape[1] == X_test_target_raw.shape[1]

        idx0 = np.where(Y_train_target_full == 0)[0]
        idx1 = np.where(Y_train_target_full == 1)[0]
        take = min(sample_per_class, len(idx0), len(idx1))
        if take == 0:
            print(f"[WARN] fold {fold_id}: one class missing. Skipping fold.")
            continue

        rng = np.random.RandomState(seed_fold)
        sel0 = rng.choice(idx0, size=take, replace=False)
        sel1 = rng.choice(idx1, size=take, replace=False)
        target_samples_rel = np.concatenate([sel0, sel1])

        mask_val = np.ones(len(Y_train_target_full), dtype=bool)
        mask_val[target_samples_rel] = False
        if mask_val.sum() == 0:
            print(f"[WARN] fold {fold_id}: validation empty. Skipping fold.")
            continue

        tgt_sel_pos = tgt_train_full_pos[target_samples_rel]
        tgt_val_pos = tgt_train_full_pos[mask_val]

        X_train_target = X_train_target_full[target_samples_rel]
        Y_train_target = Y_train_target_full[target_samples_rel].astype(np.int32)

        X_val_target = X_train_target_full[mask_val]
        Y_val_target = Y_train_target_full[mask_val].astype(np.int32)

        use_kernel = build_base_in_fold or use_legacy_kernel

        if use_kernel:
            train_union_pos = np.concatenate([src_pos, tgt_sel_pos]).astype(int)
            y_union = np.concatenate([Y_train_source, Y_train_target]).astype(np.int32)

            if build_base_in_fold:
                _validate_A_block_and_proj_pairs(
                    A_block=dag_A_block,
                    blocks_order=dag_blocks_order,
                    proj_pairs=proj_pairs,
                    strict=True
                )

                X_blocks_tr = OrderedDict({k: np.asarray(v)[train_union_pos, :] for k, v in X_blocks_all.items()})

                fitted = dag_feature_projection_blocks_fit(
                    X_blocks_tr,
                    pca_dims=dag_pca_dims,
                    notears_lambda=dag_notears_lambda,
                    max_iter=dag_max_iter,
                    loss_type=dag_loss_type,
                    A_block=dag_A_block,
                    blocks_order=dag_blocks_order,
                    random_state=dag_random_state,
                    h_tol=dag_h_tol,
                    rho_max=dag_rho_max,
                    w_threshold=dag_w_threshold,
                    transpose_W=dag_transpose_W,
                )

                USE_WEIGHTED_RBF = True
                AZ_POST_THRESHOLD = True
                AX_CLIP_Q = 0.99
                AX_TAU = 2.0
                AX_SMOOTH_UNIFORM = 0.05
                EPS_W = 1e-12

                a_x_cache = {}
                a_z_cache = {}

                a_z = compute_latent_importance_from_Wz(
                    fitted["Wz"],
                    mode="out_in",
                    normalize="mean1"
                )

                report_q_dims(fitted, print_table=True)

                export_pca_top_loadings(
                    fitted,
                    block_feature_names=block_feature_names,
                    top_k=20,
                    out_dir=fig7_dir,
                    prefix=f"{run_tag_safe}_seed{seed}_fold{fold_id}_pca_loadings"
                )

                if out_dir is not None:
                    _save_notears_fold_artifacts(
                        fitted=fitted,
                        outp=notears_dir,
                        seed=seed,
                        fold_id=fold_id,
                        tag_suffix=tag_suffix,
                        blocks_order=dag_blocks_order,
                        A_block=dag_A_block,
                        w_threshold=dag_w_threshold
                    )

                tr_out = dag_feature_projection_blocks_transform(
                    X_blocks_tr, fitted, A_block=dag_A_block, blocks_order=dag_blocks_order
                )
                projs_tr_all = tr_out["projs"]

                X_blocks_val = OrderedDict({k: np.asarray(v)[tgt_val_pos, :] for k, v in X_blocks_all.items()})
                val_out = dag_feature_projection_blocks_transform(
                    X_blocks_val, fitted, A_block=dag_A_block, blocks_order=dag_blocks_order
                )
                projs_val_all = val_out["projs"]

                X_blocks_te = OrderedDict({k: np.asarray(v)[tgt_test_pos, :] for k, v in X_blocks_all.items()})
                te_out = dag_feature_projection_blocks_transform(
                    X_blocks_te, fitted, A_block=dag_A_block, blocks_order=dag_blocks_order
                )
                projs_te_all = te_out["projs"]

                Ktt_list, Kval_list, Kte_list = [], [], []
                tags_local = []

                for p_i, (s_blk, t_blk) in enumerate(proj_pairs):
                    key = (s_blk, t_blk)
                    if key not in projs_tr_all:
                        raise KeyError(f"Projection key {key} not found in projs_tr_all.")

                    Xtr = np.asarray(projs_tr_all[key], dtype=np.float32)
                    Xva = np.asarray(projs_val_all[key], dtype=np.float32)
                    Xte = np.asarray(projs_te_all[key], dtype=np.float32)

                    if USE_WEIGHTED_RBF:
                        if t_blk not in a_x_cache:
                            a_z_t = _compute_a_z_block_from_Wz(
                                fitted, block_name=t_blk, use_post_threshold=AZ_POST_THRESHOLD, eps=EPS_W
                            )
                            a_z_cache[t_blk] = a_z_t

                            a_x_t = _compute_a_x_for_target_block(
                                fitted, tgt_block=t_blk, a_z_t=a_z_t,
                                tau=AX_TAU, clip_q=AX_CLIP_Q, smooth_to_uniform=AX_SMOOTH_UNIFORM, eps=EPS_W
                            )
                            a_x_cache[t_blk] = a_x_t

                        a_x_t = a_x_cache[t_blk]
                        sqrt_ax = np.sqrt(np.maximum(a_x_t, 0.0)).astype(np.float32)

                        Xtr_w = Xtr * sqrt_ax[None, :]
                        Xva_w = Xva * sqrt_ax[None, :]
                        Xte_w = Xte * sqrt_ax[None, :]
                    else:
                        Xtr_w, Xva_w, Xte_w = Xtr, Xva, Xte

                    ell2 = _median_length_scale_subsample(Xtr_w, max_m=subsample_m, rng=seed_fold + p_i)
                    ell2 = max(ell2, 1e-12)
                    base_gamma = 1.0 / (2.0 * ell2)

                    for mult in gamma_mults:
                        gamma = float(base_gamma * float(mult))
                        Ktt = _rbf_cross(Xtr_w, Xtr_w, gamma=gamma)
                        Kva = _rbf_cross(Xva_w, Xtr_w, gamma=gamma)
                        Kte = _rbf_cross(Xte_w, Xtr_w, gamma=gamma)
                        Ktt = (Ktt + Ktt.T) * 0.5

                        Ktt_list.append(Ktt)
                        Kval_list.append(Kva)
                        Kte_list.append(Kte)
                        tags_local.append(f"{s_blk}->{t_blk}__g{float(mult):g}")

                tags_template = tags_template or list(tags_local)

                n_src = len(src_pos)
                Ktt_list_src = [K[:n_src, :n_src] for K in Ktt_list]
                y_mkl = Y_train_source
                w = _estimate_mkl_weights_centered_alignment(Ktt_list_src, y_mkl, smoothing=mkl_smoothing)
                weights_folds.append(w.copy())

                Ktr  = _weighted_sum(Ktt_list, w)
                Kval = _weighted_sum(Kval_list, w)
                Kte  = _weighted_sum(Kte_list,  w)

                Ktr = _finalize_train_gram(Ktr)

                if out_dir is not None:
                    pd.DataFrame({"kernel_tag": tags_local, "weight": w}).to_csv(
                        os.path.join(mkl_dir, f"mkl_weights_seed{seed}_fold{fold_id}{tag_suffix}.csv"),
                        index=False
                    )

            else:
                train_union_pos = np.concatenate([src_pos, tgt_sel_pos]).astype(int)
                y_union = np.concatenate([Y_train_source, Y_train_target]).astype(np.int32)

                if K_base_list is not None:
                    Ktr_list = [K[np.ix_(train_union_pos, train_union_pos)] for K in K_base_list]
                    n_src = len(src_pos)
                    Ktr_list_src = [Kuu[:n_src, :n_src] for Kuu in Ktr_list]
                    y_mkl = Y_train_source
                    w = _estimate_mkl_weights_centered_alignment(Ktr_list_src, y_mkl, smoothing=mkl_smoothing)
                    weights_folds.append(w.copy())

                    Ktr  = _weighted_sum(Ktr_list, w)
                    Kval = _weighted_sum([K[np.ix_(tgt_val_pos,  train_union_pos)] for K in K_base_list], w)
                    Kte  = _weighted_sum([K[np.ix_(tgt_test_pos, train_union_pos)] for K in K_base_list], w)

                    Ktr = _finalize_train_gram(Ktr)

                    if out_dir is not None:
                        pd.DataFrame({"kernel_tag": kernel_tags, "weight": w}).to_csv(
                            os.path.join(mkl_dir, f"mkl_weights_seed{seed}_fold{fold_id}{tag_suffix}.csv"),
                            index=False
                        )
                else:
                    K_use = K_folds[fold_id - 1] if K_folds is not None else K_precomputed
                    Ktr = K_use[np.ix_(train_union_pos, train_union_pos)]
                    if K_folds_tt_psd is not None and (fold_id - 1) < len(K_folds_tt_psd):
                        Ktr_cand = K_folds_tt_psd[fold_id - 1]
                        if Ktr_cand.shape == Ktr.shape:
                            Ktr = Ktr_cand
                    Ktr = _finalize_train_gram(Ktr)
                    Kval = K_use[np.ix_(tgt_val_pos,  train_union_pos)]
                    Kte  = K_use[np.ix_(tgt_test_pos, train_union_pos)]

            lam_all = np.linalg.eigvalsh((Ktr + Ktr.T) * 0.5)
            lam_all_sorted = lam_all[::-1].clip(min=0)
            cum = np.cumsum(lam_all_sorted)
            ratio = cum / max(cum[-1], 1e-12)
            q_auto = int(np.searchsorted(ratio, 0.90) + 1)
            q_cap = (kpca_q if kpca_q is not None else 200)
            q = max(1, min(q_auto, q_cap, Ktr.shape[0] - 1))

            kpca = KernelPCA(n_components=q, kernel="precomputed")
            Ztr = kpca.fit_transform(Ktr)

            n_train_union = Ktr.shape[0]
            n_val = Kval.shape[0] if Kval.shape[0] != n_train_union else Kval.shape[1]
            n_te  = Kte.shape[0]  if Kte.shape[0]  != n_train_union else Kte.shape[1]

            Kval = _ensure_knew_train(Kval, n_val, n_train_union, name="Kval")
            Kte  = _ensure_knew_train(Kte,  n_te,  n_train_union, name="Kte")

            Kval = np.asarray(Kval, dtype=np.float64)
            Kte  = np.asarray(Kte,  dtype=np.float64)

            n_src = len(src_pos)
            Z_src = Ztr[:n_src, :]
            Z_tgt_tr = Ztr[n_src:, :]
            Z_val = kpca.transform(Kval)
            Z_te  = kpca.transform(Kte)

            if out_dir is not None:
                union_ids = (np.asarray(sample_ids)[train_union_pos] if sample_ids is not None else train_union_pos)
                union_R = np.asarray(R)[train_union_pos]
                union_Y = np.asarray(y_union)

                split_union = np.array(["train"] * len(train_union_pos))

                val_ids = (np.asarray(sample_ids)[tgt_val_pos] if sample_ids is not None else tgt_val_pos)
                te_ids  = (np.asarray(sample_ids)[tgt_test_pos] if sample_ids is not None else tgt_test_pos)

                val_R = np.asarray(R)[tgt_val_pos]
                te_R  = np.asarray(R)[tgt_test_pos]

                split_val = np.array(["val"] * len(tgt_val_pos))
                split_te  = np.array(["test"] * len(tgt_test_pos))

                Z_all = np.vstack([Ztr, Z_val, Z_te])
                ids_all = np.concatenate([union_ids, val_ids, te_ids])
                Y_all = np.concatenate([union_Y, Y_val_target, Y_test_target])
                R_all = np.concatenate([union_R, val_R, te_R])
                split_all = np.concatenate([split_union, split_val, split_te])

                eigvals = getattr(kpca, "eigenvalues_", None)
                if eigvals is None:
                    eigvals = getattr(kpca, "lambdas_", None)
                if eigvals is None:
                    eigvals = lam_all_sorted[:q]
                
                _save_kpca_fold_artifacts(
                    outp=outp,
                    run_tag_safe=run_tag_safe, seed=seed, fold_id=fold_id,
                    mode_name="integrated",
                    sample_ids=ids_all, Y=Y_all, R=R_all, split=split_all,
                    Z=Z_all, eigvals=eigvals,
                    extra_meta={"seed_fold": seed_fold, "q": q}
                )

            X_train_source_fold = Z_src.astype(np.float32, copy=False)
            X_train_target_fold = Z_tgt_tr.astype(np.float32, copy=False)
            X_val_target_fold   = Z_val.astype(np.float32, copy=False)
            X_test_target_fold  = Z_te.astype(np.float32, copy=False)

        else:
            X_union = np.vstack([X_train_source_raw, X_train_target]).astype(np.float32)
            n_union, p = X_union.shape
            q_cap = (kpca_q if kpca_q is not None else 200)
            q = max(1, min(q_cap, n_union - 1, p))

            if q >= 2:
                pca = PCA(n_components=q, svd_solver="auto", random_state=seed_fold)
                Z_union = pca.fit_transform(X_union)
                Z_src = Z_union[:len(Y_train_source), :]
                Z_tgt = Z_union[len(Y_train_source):, :]

                X_train_source_fold = Z_src.astype(np.float32, copy=False)
                X_train_target_fold = Z_tgt.astype(np.float32, copy=False)
                X_val_target_fold   = pca.transform(X_val_target).astype(np.float32, copy=False)
                X_test_target_fold  = pca.transform(X_test_target_raw).astype(np.float32, copy=False)
            else:
                X_train_source_fold = X_train_source_raw
                X_train_target_fold = X_train_target
                X_val_target_fold   = X_val_target
                X_test_target_fold  = X_test_target_raw

        feat_dim = int(np.asarray(X_train_source_fold).shape[1])

        pairs_dir = os.path.join(pairs_root, f"{run_tag_safe}_seed{seed}_fold{fold_id}")
        os.makedirs(pairs_dir, exist_ok=True)

        best_score, fold_auc, best_thr, pred_thr = train_and_predict(
            np.asarray(X_train_target_fold, dtype=np.float32), np.asarray(Y_train_target, dtype=np.int32),
            np.asarray(X_train_source_fold, dtype=np.float32), np.asarray(Y_train_source, dtype=np.int32),
            np.asarray(X_val_target_fold, dtype=np.float32),   np.asarray(Y_val_target, dtype=np.int32),
            np.asarray(X_test_target_fold, dtype=np.float32),  np.asarray(Y_test_target, dtype=np.int32),
            repetition=repetition,
            sample_per_class=sample_per_class,
            alpha=alpha,
            learning_rate=learning_rate,
            hiddenLayers=hiddenLayers,
            dr=dr,
            decay=decay,
            momentum=momentum,
            batch_size=batch_size,
            n_features=feat_dim,
            domain_adaptation_task=da_task,
            seed=int(seed_fold),
            pairs_dir=pairs_dir,
            apply_domain_smote=True,
            balance_mode=str(BALANCE_MODE),
            pos_neg_ratio=float(POS_NEG_RATIO),
            smote_k_max=5,
        )

        best_score = np.asarray(best_score, dtype=float).reshape(-1)
        pred_thr = np.asarray(pred_thr).astype(int).reshape(-1)
        pred_05 = (best_score > 0.5).astype(int)
        risk = 1.0 - best_score

        tgt_test_pos = np.asarray(tgt_test_pos, dtype=int)

        T_test = (np.asarray(T_days)[tgt_test_pos] if T_days is not None
                  else np.full(len(tgt_test_pos), np.nan))

        if C_flag is not None:
            E_all = (1 - np.asarray(C_flag).astype(int)).reshape(-1)
            E_test = E_all[tgt_test_pos]
        else:
            E_test = np.full(len(tgt_test_pos), np.nan)

        sid_test = (np.asarray(sample_ids)[tgt_test_pos] if sample_ids is not None else tgt_test_pos)

        df_fold = pd.DataFrame({
            "scr": best_score,
            "risk": risk,
            "pred_thr": pred_thr,
            "best_thr": float(best_thr),
            "pred_05": pred_05,
            "Y": np.asarray(Y_test_target).reshape(-1).astype(int),
            "T": T_test,
            "E": E_test,
            "fold": int(fold_id),
            "R": np.asarray(R)[tgt_test_pos],
            "sample_id": sid_test,
            "seed": int(seed),
            "method": str(run_tag_safe)
        }, index=tgt_test_pos)

        global_idx = tgt_test_pos
        sid = (np.asarray(sample_ids)[global_idx] if sample_ids is not None else global_idx)
        T_fold = (np.asarray(T_days)[global_idx] if T_days is not None else np.full(len(global_idx), np.nan))
        C_fold = (np.asarray(C_flag)[global_idx] if C_flag is not None else np.full(len(global_idx), np.nan))
        Event = (1 - C_fold) if C_flag is not None else np.full(len(global_idx), np.nan)

        X_test_raw = X_test_target_raw
        cols = (
            list(feat_names)
            if (feat_names is not None and len(feat_names) == X_test_raw.shape[1])
            else [f"feat_{i}" for i in range(X_test_raw.shape[1])]
        )
        df_feat = pd.DataFrame(X_test_raw, columns=cols)
        df_feat.insert(0, "Sample", sid)
        df_feat.insert(1, "score", best_score)

        save_subgroup_fold_metrics(
            df_fold=df_fold,
            save_path=os.path.join(
                score_dir,
                f"subgroup_fold_metrics_seed{seed}_fold{fold_id}{tag_suffix}.csv"
            ),
            seed=seed,
            fold_id=fold_id,
            method_tag=run_tag_safe
        )

        save_calibration_outputs(
            df_fold=df_fold,
            save_points_path=os.path.join(
                score_dir,
                f"calibration_points_seed{seed}_fold{fold_id}{tag_suffix}.csv"
            ),
            save_summary_path=os.path.join(
                score_dir,
                f"calibration_summary_seed{seed}_fold{fold_id}{tag_suffix}.csv"
            ),
            n_bins=10
        )

        save_dca_outputs(
            df_fold=df_fold,
            save_path=os.path.join(
                score_dir,
                f"dca_seed{seed}_fold{fold_id}{tag_suffix}.csv"
            ),
            thresholds=np.arange(0.05, 0.81, 0.01)
        )

        df_fold.to_csv(
            os.path.join(score_dir, f"df_score_seed{seed}_fold{fold_id}{tag_suffix}.csv"),
            index=True
        )
        pd.DataFrame({
            "Sample": sid, "risk": 1.0 - best_score, "Y": np.asarray(Y_test_target).reshape(-1),
            "T_days": T_fold, "Event": Event, "R": np.asarray(R)[global_idx],
        }).to_csv(
            os.path.join(score_dir, f"km_input_seed{seed}_fold{fold_id}{tag_suffix}.csv"),
            index=False
        )
        df_feat.to_csv(
            os.path.join(score_dir, f"feature_score_test_seed{seed}_fold{fold_id}{tag_suffix}.csv"),
            index=False
        )
        paired_metric_rows.append(make_method_summary_row(
            df_score=df_fold,
            seed=seed,
            method_tag=run_tag_safe,
            lr=learning_rate,
            dr=dr,
            alpha=alpha,
            a_block_type="proposed" if dag_A_block is not None else "none",
            integration_type="NOTEARS_MKL" if use_kernel else "PCA_only",
        ))

        df_score_parts.append(df_fold)

    if len(df_score_parts) == 0:
        df_out = pd.DataFrame({
            "folds": fold,
            "TLDA_auc": np.nan,
            "TLDA_acc": np.nan,
            "TLDA_cindex": np.nan,
            "TLDA_f1": np.nan,
            "TLDA_prauc": np.nan
        }, index=[seed])
        return df_out, np.array([])

    df_score = pd.concat(df_score_parts, axis=0)
    df_score = df_score[~df_score.index.duplicated(keep="last")].sort_index()

    thr_summary = df_score.groupby("fold")["best_thr"].first().reset_index()

    thr_stab = summarize_threshold_stability(thr_summary, seed=seed, method_tag=run_tag_safe)
    thr_stab.to_csv(
        os.path.join(score_dir, f"threshold_stability_seed{seed}{tag_suffix}.csv"),
        index=False
    )

    df_score.to_csv(
        os.path.join(score_dir, f"df_score_seed{seed}{tag_suffix}.csv"),
        index=True
    )
    thr_summary.to_csv(
        os.path.join(score_dir, f"thresholds_seed{seed}{tag_suffix}.csv"),
        index=False
    )
    if len(paired_metric_rows) > 0:
        pd.DataFrame(paired_metric_rows).to_csv(
            os.path.join(score_dir, f"paired_fold_metrics_seed{seed}{tag_suffix}.csv"),
            index=False
    )

    y_true = df_score["Y"].astype(int).values
    y_scr  = df_score["scr"].astype(float).values
    y_pred = df_score["pred_thr"].astype(int).values
    
    roc_auc = roc_auc_score(y_true, y_scr) if len(np.unique(y_true)) == 2 else np.nan
    pr_auc  = average_precision_score(y_true, y_scr) if len(np.unique(y_true)) == 2 else np.nan
    acc = accuracy_score(y_true, y_pred)
    f1v = f1_score(y_true, y_pred, average="weighted")

    if (len(weights_folds) > 0) and (out_dir is not None):
        W = np.vstack(weights_folds)
        tags_for_summary = (
            tags_template if (build_base_in_fold and tags_template is not None)
            else (kernel_tags if kernel_tags is not None else [f"k{j}" for j in range(W.shape[1])])
        )
        df_mkl_sum = pd.DataFrame({
            "kernel_tag": tags_for_summary,
            "weight": W.mean(axis=0),
            "w_std":  W.std(axis=0)
        })
        df_mkl_sum.to_csv(
            os.path.join(mkl_dir, f"mkl_weights_seed{seed}_summary{tag_suffix}.csv"),
            index=False
        )

        df_mkl_cat = summarize_mkl_weights_by_category(df_mkl_sum.rename(columns={"weight": "weight"}))
        df_mkl_cat.to_csv(
            os.path.join(mkl_dir, f"mkl_weight_by_category_seed{seed}_summary{tag_suffix}.csv"),
            index=False
        )

    df_cidx = df_score.dropna(subset=["T", "E", "risk"]).copy()
    if len(df_cidx) > 0:
        df_cidx["E"] = df_cidx["E"].astype(int)

    if (len(df_cidx) < 2) or (df_cidx["E"].sum() == 0) or (df_cidx["T"].nunique() < 2):
        cindex = np.nan
    else:
        cindex = concordance_index_simple(
            T=df_cidx["T"].values,
            risk=df_cidx["risk"].values,
            E=df_cidx["E"].values
        )

    df_out = pd.DataFrame({
        "folds": fold,
        "TLDA_auc": roc_auc,
        "TLDA_acc": acc,
        "TLDA_cindex": cindex,
        "TLDA_f1": f1v,
        "TLDA_prauc": pr_auc
    }, index=[seed])

    score_vec_target_order = df_score["scr"].reindex(tgt_pos).to_numpy(dtype=float)
    return df_out, score_vec_target_order

def run_BRCA_inter_cv(data, years, target, fn, tn, lr, dr):
    if not isinstance(data, dict) or ("X" not in data):
        raise ValueError("run_BRCA_inter_cv expects `data` dict containing at least key 'X'.")
    if np.asarray(data["X"]).shape[0] < 10:
        return None

    def _ensure_good_cwd(fallback_dir=None):
        try:
            _ = os.getcwd()
            return
        except FileNotFoundError:
            pass

        fb = fallback_dir or data.get("work_dir", None) or "/tmp"
        try:
            os.makedirs(fb, exist_ok=True)
            os.chdir(fb)
        except Exception:
            os.chdir("/tmp")

    _ensure_good_cwd()

    dataset_base = dict(data)
    dataset_tl   = normalize_dataset(dataset_base)
    dataset_tl   = get_n_years(dataset_tl, years)
    dataset_dbg  = get_n_years(dataset_base, years)

    _X_all = np.asarray(data["X"])
    N_all  = int(_X_all.shape[0])

    _S_all = np.asarray(data.get("Samples", np.arange(N_all))).astype(str)
    _T_all = np.asarray(data["T"])
    _C_all = np.asarray(data["C"])
    _R_all = np.asarray(data["R"]).astype(str)

    _mask_years_all = ~((_T_all <= 365 * years) & (_C_all == 1))
    keep_pos = np.nonzero(_mask_years_all)[0]

    X_tl, Y_tl, R_tl, y_sub_tl, y_strat_tl = dataset_tl
    Y_tl = np.asarray(Y_tl).astype(np.int32)
    n_keep = int(np.asarray(X_tl).shape[0])

    if len(keep_pos) != n_keep:
        raise ValueError(
            f"[Mismatch] years-filtered count differs: len(keep_pos)={len(keep_pos)} vs dataset_tl N={n_keep}. "
            f"Check get_n_years() rule or input meta alignment."
        )

    sample_ids_mix = _S_all[_mask_years_all].astype(str)
    T_days_mix     = _T_all[_mask_years_all]
    C_flag_mix     = _C_all[_mask_years_all]
    R_mix          = _R_all[_mask_years_all]

    if not np.array_equal(np.asarray(R_tl).astype(str), np.asarray(R_mix).astype(str)):
        raise ValueError(
            "[Order mismatch] dataset_tl R order != years-filtered R from raw data. "
            "This means get_n_years() reorders rows OR data['X'] and meta(Samples/T/C/R) are misaligned."
        )

    sample_ids_black = sample_ids_mix[np.asarray(R_mix) == "BLACK"].astype(str)

    X_dbg, Y_dbg, R_dbg, y_sub_dbg, y_strat_dbg = dataset_dbg
    df_dbg = pd.DataFrame({"RY": y_strat_dbg})
    df_dbg["R"] = np.asarray(R_dbg).astype(str)
    df_dbg["Y"] = np.asarray(Y_dbg).astype(int)

    print(np.asarray(X_dbg).shape)
    vc = df_dbg["RY"].value_counts().to_dict()
    print(vc)
    for key in vc:
        print(key, vc[key])

    _exclude = {"Samples", "T", "C", "Y", "R"}
    feat_names = None
    if isinstance(data, dict):
        if data.get("FeatureName", None) is not None:
            feat_names = list(data["FeatureName"])
        elif data.get("feature_names", None) is not None:
            feat_names = list(data["feature_names"])
        elif isinstance(data.get("df", None), pd.DataFrame):
            feat_names = [c for c in data["df"].columns if c not in _exclude]

    def _slice_blocks_all_maybe(X_blocks_all, keep_pos_, N_all_, n_keep_):
        if X_blocks_all is None:
            return None
        if not isinstance(X_blocks_all, dict):
            raise TypeError("X_blocks_all must be a dict/OrderedDict of {block: (N x p)}.")

        out = OrderedDict()
        for blk, Xb in X_blocks_all.items():
            Xb = np.asarray(Xb)
            if Xb.ndim != 2:
                raise ValueError(f"X_blocks_all['{blk}'] must be 2D, got shape={Xb.shape}")
            if Xb.shape[0] == N_all_:
                out[blk] = Xb[keep_pos_, :]
            elif Xb.shape[0] == n_keep_:
                out[blk] = Xb
            else:
                raise ValueError(
                    f"X_blocks_all['{blk}'] first dim mismatch: got {Xb.shape[0]}, "
                    f"expected {N_all_} (raw) or {n_keep_} (years-filtered)."
                )
        return out

    kwargs_kernel = {}

    X_blocks_all = data.get("X_blocks_all", None)
    if X_blocks_all is None:
        raise ValueError(
            "Fold-internal kernel build is required: data['X_blocks_all'] must be provided. "
            "If you want to use precomputed kernels, pass K_folds/K_precomputed and modify run_CCSA_transfer accordingly."
        )

    X_blocks_all_f = _slice_blocks_all_maybe(X_blocks_all, keep_pos, N_all, n_keep)
    kwargs_kernel.update({
        "X_blocks_all": X_blocks_all_f,
        "dag_pca_dims": data.get("dag_pca_dims", None),
        "dag_notears_lambda": data.get("dag_notears_lambda", 0.1),
        "dag_max_iter": data.get("dag_max_iter", 30),
        "dag_loss_type": data.get("dag_loss_type", "l2"),
        "dag_A_block": data.get("dag_A_block", None),
        "dag_blocks_order": data.get("dag_blocks_order", None),
        "dag_random_state": data.get("dag_random_state", 0),
        "dag_h_tol": data.get("dag_h_tol", None),
        "dag_rho_max": data.get("dag_rho_max", None),
        "dag_w_threshold": data.get("dag_w_threshold", None),
        "dag_transpose_W": data.get("dag_transpose_W", False),

        "proj_pairs": data.get("proj_pairs", None),
        "gamma_mults": data.get("gamma_mults", (0.5, 1.0, 2.0)),
        "subsample_m": data.get("subsample_m", 300),
        "use_nystrom": data.get("use_nystrom", False),
        "nystrom_s": data.get("nystrom_s", 128),
        "mkl_smoothing": float(data.get("mkl_smoothing", 0.15)),
    })

    base_root = Path(data.get("base_out_dir", os.getcwd()))
    base_out = base_root / "outputs" / f"{years}yr_{target}"

    cohort_dir = base_out / "figures" / "Figure_CohortSummary"
    cohort_dir.mkdir(parents=True, exist_ok=True)

    save_cohort_summary(
        inter_data=data.get("cohort_summary_input", data),
        years=years,
        save_path=str(cohort_dir / f"cohort_summary_{years}yr_{target}.csv")
    )
    
    results_dir = base_out / "results_tables"
    results_dir.mkdir(parents=True, exist_ok=True)

    hp_tag = f"lr{lr:g}_dr{dr:g}"
    out_dir_tl_p   = base_out / hp_tag / "TL"
    out_dir_tlda_p = base_out / hp_tag / "TLDA"
    out_dir_tl_p.mkdir(parents=True, exist_ok=True)
    out_dir_tlda_p.mkdir(parents=True, exist_ok=True)
    
    out_dir_tl   = str(out_dir_tl_p)
    out_dir_tlda = str(out_dir_tlda_p)

    parameters_common = dict(
        fold=int(fn),
        n_features=50,
        alpha=ALPHA,
        batch_size=20,
        learning_rate=float(lr),
        hiddenLayers=HIDDENLAYERS,
        dr=float(dr),
        momentum=0.9,
        decay=DECAY,
        sample_per_class=5,
        kpca_q=None,
    )

    def _call_with_supported_kwargs(func, *args, **kwargs):
        sig = inspect.signature(func)
        supported = {kk: vv for kk, vv in kwargs.items() if kk in sig.parameters}
        return func(*args, **supported)

    def _collect_scores_from_km_files(seed_local, folds, outp_local, run_tag=None):
        tag = (str(run_tag) if run_tag is not None else "").strip()
        suf = f"_{tag}" if tag else ""
    
        score_root = os.path.join(outp_local, "figures", "Figure_Scores_KM")
    
        dfs = []
        for fid in range(1, int(folds) + 1):
            f1 = os.path.join(score_root, f"km_input_seed{seed_local}_fold{fid}{suf}.csv")
            f0 = os.path.join(score_root, f"km_input_seed{seed_local}_fold{fid}.csv")
            fpath = f1 if os.path.exists(f1) else f0
            if os.path.exists(fpath):
                dfs.append(pd.read_csv(fpath))
    
        if len(dfs) == 0:
            return pd.Series(dtype=float)
    
        km = pd.concat(dfs, axis=0, ignore_index=True)
        if ("Sample" not in km.columns) or ("risk" not in km.columns):
            return pd.Series(dtype=float)
    
        km = km.dropna(subset=["Sample"])
        km["Sample"] = km["Sample"].astype(str)
        s_eventfree = 1.0 - km.drop_duplicates(subset=["Sample"], keep="last").set_index("Sample")["risk"].astype(float)
        return s_eventfree

    res = pd.DataFrame()
    paired_all_rows = []
    score_dict = {}

    run_tag_tl   = "TL_only"
    run_tag_tlda = "TLDA"

    seed_base = int(data.get("seed_base", SEED))

    for i in range(int(tn)):
        seed_rep = seed_base + int(i)
        print("***** Seed + i:", seed_rep, "*****")

        s_time = time.time()
        df_tl, _ = _call_with_supported_kwargs(
            run_CCSA_transfer,
            seed_rep, dataset_tl,
            block_feature_names=data.get("block_feature_names", None),
            sample_ids=sample_ids_mix,
            feat_names=feat_names,
            T_days=T_days_mix,
            C_flag=C_flag_mix,
            out_dir=out_dir_tl,
            run_tag=run_tag_tl,
            **parameters_common,
            **kwargs_kernel,
        )
        e_time = time.time()
        print(f"N{i} DomainAdaptation model (TL_only): {(e_time - s_time)/60:.2f} min")
        print(df_tl.to_string(index=False))

        s_time = time.time()
        df_tl2, _ = _call_with_supported_kwargs(
            run_CCSA_transfer_SMOTE,
            seed_rep, dataset_tl,
            block_feature_names=data.get("block_feature_names", None),
            sample_ids=sample_ids_mix,
            feat_names=feat_names,
            T_days=T_days_mix,
            C_flag=C_flag_mix,
            out_dir=out_dir_tlda,
            run_tag=run_tag_tlda,
            **parameters_common,
            **kwargs_kernel,
        )
        e_time = time.time()
        print(f"N{i} DomainAdaptation model (TLDA): {(e_time - s_time)/60:.2f} min")
        print(df_tl2.to_string(index=False))

        df_tl_long = df_tl.copy()
        df_tl_long["seed"] = seed_rep
        df_tl_long["method"] = "TL_only"
        df_tl_long["lr"] = lr
        df_tl_long["dr"] = dr

        df_tl2_long = df_tl2.copy()
        df_tl2_long["seed"] = seed_rep
        df_tl2_long["method"] = "TLDA"
        df_tl2_long["lr"] = lr
        df_tl2_long["dr"] = dr

        paired_all_rows.append(df_tl_long)
        paired_all_rows.append(df_tl2_long)

        df_join = pd.concat([df_tl, df_tl2], sort=False, axis=1)
        res = pd.concat([res, df_join], axis=0, ignore_index=True)

        s_tl   = _collect_scores_from_km_files(seed_rep, fn, out_dir_tl,   run_tag=run_tag_tl)
        s_tlda = _collect_scores_from_km_files(seed_rep, fn, out_dir_tlda, run_tag=run_tag_tlda)

        ccsa1 = s_tl.reindex(sample_ids_black)
        ccsa2 = s_tlda.reindex(sample_ids_black)

        score_dict[seed_rep] = pd.DataFrame(
            {"ccsa_score": ccsa1.values, "ccsa2_score": ccsa2.values},
            index=sample_ids_black
        )

    results_dir = base_out / "figures" / "Figure_ResultSummary"
    results_dir.mkdir(parents=True, exist_ok=True)
    fkey = f"{lr}_{dr}"

    if len(paired_all_rows) > 0:
        paired_all_df = pd.concat(paired_all_rows, axis=0, ignore_index=True)
        paired_all_df.to_csv(
            str(results_dir / f"paired_fold_metrics_all_methods_{fkey}.csv"),
            index=False
        )

    paired_summary_df, paired_wide_df = summarize_paired_method_comparison(paired_all_df)
    paired_summary_df.to_csv(
        str(results_dir / f"paired_method_summary_{fkey}.csv"),
        index=False
    )
    paired_wide_df.to_csv(
        str(results_dir / f"paired_method_seedwide_{fkey}.csv"),
        index=False
    )

    rows_cmp = []
    for seed_rep, sdf in score_dict.items():
        sdf = sdf.dropna().copy()
        if len(sdf) == 0:
            continue

        y_black = pd.Series(
            data=Y_tl[np.asarray(R_mix) == "BLACK"],
            index=sample_ids_black
        ).reindex(sdf.index).values
        
        s_tl = sdf["ccsa_score"].values
        s_tlda = sdf["ccsa2_score"].values
    
        boot_auc = paired_bootstrap_metric_diff(y_black, s_tl, s_tlda, metric="auc", n_boot=2000, seed=seed_rep)
        perm_auc = paired_permutation_test(y_black, s_tl, s_tlda, metric="auc", n_perm=5000, seed=seed_rep)
    
        boot_pr = paired_bootstrap_metric_diff(y_black, s_tl, s_tlda, metric="prauc", n_boot=2000, seed=seed_rep)
        perm_pr = paired_permutation_test(y_black, s_tl, s_tlda, metric="prauc", n_perm=5000, seed=seed_rep)
    
        rows_cmp.append({
            "seed": seed_rep,
            "auc_diff_mean": boot_auc["diff_mean"],
            "auc_ci_lower": boot_auc["ci_lower"],
            "auc_ci_upper": boot_auc["ci_upper"],
            "auc_perm_p": perm_auc["p_value"],
            "prauc_diff_mean": boot_pr["diff_mean"],
            "prauc_ci_lower": boot_pr["ci_lower"],
            "prauc_ci_upper": boot_pr["ci_upper"],
            "prauc_perm_p": perm_pr["p_value"],
        })
    
    pd.DataFrame(rows_cmp).to_csv(
        str(results_dir / f"paired_bootstrap_permutation_{fkey}.csv"),
        index=False
    )
    
    for mode_name, mode_dir in [("TL", out_dir_tl_p), ("TLDA", out_dir_tlda_p)]:
        notears_dir = Path(mode_dir) / "figures" / "Figure_NOTEARS"
        df_notears_stab = summarize_notears_stability(str(notears_dir))
        if len(df_notears_stab) > 0:
            df_notears_stab.to_csv(
                str(results_dir / f"notears_block_stability_{mode_name}_{fkey}.csv"),
                index=False
            )

    for mode_name, mode_dir, tag_name in [
        ("TL", out_dir_tl_p, run_tag_tl),
        ("TLDA", out_dir_tlda_p, run_tag_tlda)
    ]:
        score_dir_local = Path(mode_dir) / "figures" / "Figure_Scores_KM"
    
        df_sub = aggregate_subgroup_metrics(str(score_dir_local), tag_name)
        if len(df_sub) > 0:
            df_sub.to_csv(str(results_dir / f"subgroup_metrics_aggregate_{mode_name}_{fkey}.csv"), index=False)
    
        df_cal = aggregate_calibration_summaries(str(score_dir_local), tag_name)
        if len(df_cal) > 0:
            df_cal.to_csv(str(results_dir / f"calibration_summary_aggregate_{mode_name}_{fkey}.csv"), index=False)
    
        df_dca = aggregate_dca(str(score_dir_local), tag_name)
        if len(df_dca) > 0:
            df_dca.to_csv(str(results_dir / f"dca_aggregate_{mode_name}_{fkey}.csv"), index=False)

    for mode_name, mode_dir in [("TL", out_dir_tl_p), ("TLDA", out_dir_tlda_p)]:
        mkl_dir = Path(mode_dir) / "figures" / "Figure_MKL"
        files = sorted(glob.glob(str(mkl_dir / "mkl_weight_by_category_seed*_summary*.csv")))
        if len(files) > 0:
            dfs = [pd.read_csv(f) for f in files]
            df_cat = pd.concat(dfs, axis=0, ignore_index=True)
            df_cat_sum = df_cat.groupby(["pair_category", "gamma_category"], as_index=False).agg(
                mean_weight=("mean", "mean"),
                sd_weight=("mean", "std"),
                total_sum=("sum", "mean"),
                n_runs=("mean", "size"),
            )
            df_cat_sum.to_csv(
                str(results_dir / f"mkl_weight_by_category_aggregate_{mode_name}_{fkey}.csv"),
                index=False
            )

    for mode_name, mode_dir, tag_name in [
        ("TL", out_dir_tl_p, run_tag_tl),
        ("TLDA", out_dir_tlda_p, run_tag_tlda)
    ]:
        fig7_dir = Path(mode_dir) / "figures" / "Figure7"
    
        df_gene_stab = summarize_top_loading_stability(str(fig7_dir), tag_name, top_k=20)
        if len(df_gene_stab) > 0:
            df_gene_stab.to_csv(
                str(results_dir / f"top_feature_stability_{mode_name}_{fkey}.csv"),
                index=False
            )
    
            df_gene_top = make_consensus_top_features(df_gene_stab, top_n=20)
            df_gene_top.to_csv(
                str(results_dir / f"consensus_top20_features_{mode_name}_{fkey}.csv"),
                index=False
            )

    f_name = (
        "BRCA-AA-EA-mRNA_miRNA_methyl_filtered_kernel_PCC-"
        + str(target) + "-" + str(years) + "YR_MJ_" + fkey + ".xlsx"
    )
    res.to_excel(str(results_dir / f_name), index=False)

    num_cols = res.select_dtypes(include=[np.number]).columns

    numeric_df = res.select_dtypes(include=[np.number])
    
    if numeric_df.shape[1] == 0:
        summary_df = pd.DataFrame({"Column": [], "Mean": [], "Standard Deviation": []})
    else:
        stats = numeric_df.agg(["mean", "std"]).T
        stats = stats.reset_index().rename(
            columns={"index": "Column", "mean": "Mean", "std": "Standard Deviation"}
        )
        summary_df = stats
        
    summary_name = (
        "summary-BRCA-mRNA_miRNA_methyl_filtered_kernel_PCC-"
        + str(target) + "-" + str(years) + "YR_MJ_" + fkey + ".xlsx"
    )
    summary_df.to_excel(str(results_dir / summary_name), index=False)

    pkl_name = (
        "BRCA-AA-EA-mRNA_miRNA_methyl_filtered_kernel_PCC-"
        + str(target) + "-" + str(years) + "YR_MJ_" + fkey + ".pkl"
    )
    df_black_truth = pd.DataFrame({
        "Sample": sample_ids_black,
        "Y_event_free_by_horizon": Y_tl[np.asarray(R_mix) == "BLACK"]
    })
    df_black_truth.to_csv(
        str(results_dir / f"black_target_truth_{fkey}.csv"),
        index=False
    )
    
    with open(str(results_dir / pkl_name), "wb") as f:
        pickle.dump(score_dict, f)

    km_all_rows = []
    for mode_name, mode_dir, tag_name in [("TL", out_dir_tl, run_tag_tl), ("TLDA", out_dir_tlda, run_tag_tlda)]:
        dfs = []
        for seed_rep in range(seed_base, seed_base + int(tn)):
            for fid in range(1, int(fn) + 1):
                f1 = os.path.join(mode_dir, "figures", "Figure_Scores_KM",
                                  f"km_input_seed{seed_rep}_fold{fid}_{tag_name}.csv")
                f0 = os.path.join(mode_dir, "figures", "Figure_Scores_KM",
                                  f"km_input_seed{seed_rep}_fold{fid}.csv")
                fpath = f1 if os.path.exists(f1) else f0
                if os.path.exists(fpath):
                    tmp = pd.read_csv(fpath)
                    tmp["method"] = mode_name
                    dfs.append(tmp)

        if len(dfs) > 0:
            km_all = pd.concat(dfs, axis=0, ignore_index=True)
            summarize_cox_risk_effect(
                km_all,
                save_path=str(results_dir / f"cox_risk_summary_{mode_name}_{fkey}.csv")
            )

    return res, score_dict

def _make_unique_columns(cols):
    cols = [str(c) for c in cols]
    seen = {}
    out = []
    for c in cols:
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}__dup{seen[c]}")
    return out


def _load_tcga_matrix(csv_path, id_len=12):
    DF = pd.read_csv(csv_path)
    DF = DF.T
    DF.columns = _make_unique_columns(DF.iloc[0].astype(str).values)
    DF = DF.iloc[1:].copy()

    DF.index = [str(i)[:id_len] for i in DF.index]
    DF.index = DF.index.astype(str)

    DF = DF.sort_index()
    DF = DF.reset_index().rename(columns={"index": "Sample"})
    DF = DF.drop_duplicates(subset="Sample", keep="first").set_index("Sample")

    DF = DF.apply(pd.to_numeric, errors="coerce")
    return DF


def _filter_nan_and_constant_features(X_df, min_non_nan_ratio=0.95):
    min_non_nan = int(np.ceil(min_non_nan_ratio * X_df.shape[0]))
    X_df = X_df.dropna(axis=1, thresh=min_non_nan)
    v = X_df.var(axis=0)
    X_df = X_df.loc[:, (v > 0) & np.isfinite(v)]
    return X_df


def main(args):
    print(f"------------ Prepare data: ({ynum}yr | endpoint={endpoint}) ------------")

    for _f in ["add_race_CT", "dag_feature_projection_blocks_fit", "dag_feature_projection_blocks_transform"]:
        if _f not in globals():
            raise NameError(
                f"{_f} is not defined in globals(). "
                "Define/import it before running this script."
            )

    _inputs_dir = Path(input_path)

    required_files = [
        _inputs_dir / "mRNA_filtered.csv",
        _inputs_dir / "miRNA_filtered.csv",
        _inputs_dir / "methyl_filtered.csv",
        _inputs_dir / "Genetic_Ancestry.xlsx",
        _inputs_dir / "TCGA-CDR-SupplementalTableS1.xlsx",
    ]
    missing_files = [str(p) for p in required_files if not p.exists()]
    if len(missing_files) > 0:
        raise FileNotFoundError(
            "Required input files are missing:\n" + "\n".join(missing_files)
        )

    df_mrna = _load_tcga_matrix(_inputs_dir / "mRNA_filtered.csv")
    df_mirna = _load_tcga_matrix(_inputs_dir / "miRNA_filtered.csv")
    df_methy = _load_tcga_matrix(_inputs_dir / "methyl_filtered.csv")

    print("mRNA shape (samples x features):", df_mrna.shape)
    print("miRNA shape (samples x features):", df_mirna.shape)
    print("Methy shape (samples x features):", df_methy.shape)

    common = np.intersect1d(
        np.intersect1d(df_mrna.index.values, df_mirna.index.values),
        df_methy.index.values,
    )
    common = np.sort(common.astype(str))

    df_mrna = df_mrna.loc[common].copy()
    df_mirna = df_mirna.loc[common].copy()
    df_methy = df_methy.loc[common].copy()
    print("Common samples:", len(common))

    df_mrna = _filter_nan_and_constant_features(df_mrna, min_non_nan_ratio=0.95)
    df_mirna = _filter_nan_and_constant_features(df_mirna, min_non_nan_ratio=0.95)
    df_methy = _filter_nan_and_constant_features(df_methy, min_non_nan_ratio=0.95)

    df_mrna = df_mrna.fillna(df_mrna.median(axis=0))
    df_mirna = df_mirna.fillna(df_mirna.median(axis=0))
    df_methy = df_methy.fillna(df_methy.median(axis=0))

    df = df_mrna
    df1 = df_mirna
    df2 = df_methy

    df.index = df.index.astype(str)
    df1.index = df1.index.astype(str)
    df2.index = df2.index.astype(str)

    assert list(df.index) == list(df1.index) == list(df2.index), \
        "Sample order mismatch across mRNA/miRNA/methyl."

    print("[OK] Multi-omics loaded/aligned:", df.shape[0], "samples")

    X_merge_full = pd.concat(
        [
            df.add_prefix("mRNA__"),
            df1.add_prefix("miRNA__"),
            df2.add_prefix("methyl__"),
        ],
        axis=1,
    )

    inter_full = add_race_CT(
        cancer_type="BRCA",
        df=X_merge_full,
        target=endpoint,
        groups=("WHITE", "BLACK"),
        input_path=input_path,
    )

    for k in ["Samples", "T", "C", "R", "X"]:
        if k not in inter_full:
            raise KeyError(f"add_race_CT output must contain '{k}'.")

    samples_full = np.asarray(inter_full["Samples"]).astype(str)
    T_days_full = np.asarray(inter_full["T"])
    C_flag_full = np.asarray(inter_full["C"])
    R_full = np.asarray(inter_full["R"]).astype(str)

    missing = [s for s in samples_full if s not in df.index]
    if len(missing) > 0:
        raise KeyError(
            f"Some Samples from add_race_CT are missing in df index. Example: {missing[:5]}"
        )

    df = df.loc[samples_full].copy()
    df1 = df1.loc[samples_full].copy()
    df2 = df2.loc[samples_full].copy()

    print("[OK] Master order fixed by add_race_CT:", len(samples_full), "samples")

    cohort_summary_input = {
        "T": T_days_full.copy(),
        "C": C_flag_full.copy(),
        "R": R_full.copy(),
    }

    time_p = 365 * int(ynum)
    keep_mask_ct = ~((T_days_full <= time_p) & (C_flag_full == 1))
    keep_ids = samples_full[keep_mask_ct]

    y_all = np.ones(int(keep_mask_ct.sum()), dtype=np.int32)
    y_all[(T_days_full[keep_mask_ct] <= time_p) & (C_flag_full[keep_mask_ct] == 0)] = 0

    print("keep_ids:", len(keep_ids), "| y_all distribution:", np.bincount(y_all))

    df = df.loc[keep_ids].copy()
    df1 = df1.loc[keep_ids].copy()
    df2 = df2.loc[keep_ids].copy()

    for _name, _d in [("mRNA", df), ("miRNA", df1), ("methyl", df2)]:
        if not np.issubdtype(_d.values.dtype, np.number):
            raise TypeError(f"{_name} has non-numeric dtype")
        if np.isnan(_d.values).any():
            raise ValueError(f"{_name} contains NaNs. Impute/drop before proceeding.")

    inter_m_mi_BRCA = dict(inter_full)

    inter_m_mi_BRCA["Samples"] = keep_ids.astype(str)
    inter_m_mi_BRCA["T"] = T_days_full[keep_mask_ct]
    inter_m_mi_BRCA["C"] = C_flag_full[keep_mask_ct]
    inter_m_mi_BRCA["R"] = R_full[keep_mask_ct].astype(str)
    inter_m_mi_BRCA["E"] = (1 - inter_m_mi_BRCA["C"]).astype(np.int32)

    X_merge_filtered = pd.concat(
        [
            df.add_prefix("mRNA__"),
            df1.add_prefix("miRNA__"),
            df2.add_prefix("methyl__"),
        ],
        axis=1,
    )

    inter_m_mi_BRCA["df"] = X_merge_filtered
    inter_m_mi_BRCA["feature_names"] = list(X_merge_filtered.columns)
    inter_m_mi_BRCA["FeatureName"] = list(X_merge_filtered.columns)
    inter_m_mi_BRCA["X"] = X_merge_filtered.to_numpy(dtype=np.float32, copy=True)
    inter_m_mi_BRCA["cohort_summary_input"] = cohort_summary_input

    X_blocks_all = OrderedDict({
        "mRNA": df.to_numpy(dtype=np.float32, copy=True),
        "miRNA": df1.to_numpy(dtype=np.float32, copy=True),
        "methyl": df2.to_numpy(dtype=np.float32, copy=True),
    })

    A_block = np.array(
        [
            [0, 0, 0],
            [1, 1, 0],
            [1, 0, 1],
        ],
        dtype=int,
    )

    proj_pairs = [
        ("miRNA", "mRNA"),
        ("methyl", "mRNA"),
        ("miRNA", "miRNA"),
        ("methyl", "methyl"),
    ]

    inter_m_mi_BRCA["X_blocks_all"] = X_blocks_all
    inter_m_mi_BRCA["dag_pca_dims"] = {
        "mRNA": min(16, X_blocks_all["mRNA"].shape[1], max(1, X_blocks_all["mRNA"].shape[0] - 1)),
        "miRNA": min(16, X_blocks_all["miRNA"].shape[1], max(1, X_blocks_all["miRNA"].shape[0] - 1)),
        "methyl": min(16, X_blocks_all["methyl"].shape[1], max(1, X_blocks_all["methyl"].shape[0] - 1)),
    }
    inter_m_mi_BRCA["dag_notears_lambda"] = 0.1
    inter_m_mi_BRCA["dag_max_iter"] = 30
    inter_m_mi_BRCA["dag_loss_type"] = "l2"
    inter_m_mi_BRCA["dag_A_block"] = A_block
    inter_m_mi_BRCA["dag_blocks_order"] = ["mRNA", "miRNA", "methyl"]
    inter_m_mi_BRCA["dag_random_state"] = 0
    inter_m_mi_BRCA["dag_h_tol"] = None
    inter_m_mi_BRCA["dag_rho_max"] = None
    inter_m_mi_BRCA["dag_w_threshold"] = None
    inter_m_mi_BRCA["dag_transpose_W"] = False

    inter_m_mi_BRCA["proj_pairs"] = proj_pairs
    inter_m_mi_BRCA["gamma_mults"] = (0.5, 1.0, 2.0)
    inter_m_mi_BRCA["subsample_m"] = 300
    inter_m_mi_BRCA["use_nystrom"] = False
    inter_m_mi_BRCA["nystrom_s"] = 128
    inter_m_mi_BRCA["mkl_smoothing"] = 0.5
    inter_m_mi_BRCA["block_feature_names"] = {
        "mRNA": list(df.columns),
        "miRNA": list(df1.columns),
        "methyl": list(df2.columns),
    }

    # Command-line seed/output propagation
    inter_m_mi_BRCA["seed_base"] = int(args.seed_base)
    inter_m_mi_BRCA["base_out_dir"] = str(out_dir)

    bo = inter_m_mi_BRCA["dag_blocks_order"]
    A = inter_m_mi_BRCA["dag_A_block"]
    assert A.shape == (len(bo), len(bo))

    print("[READY] inter_m_mi_BRCA prepared for fold-internal DAG+kernel build.")
    print("endpoint:", endpoint, "| ynum:", ynum)
    print("N_keep:", len(inter_m_mi_BRCA["Samples"]))
    print("X shape:", np.asarray(inter_m_mi_BRCA["X"]).shape)
    print("Blocks:", {k: v.shape for k, v in inter_m_mi_BRCA["X_blocks_all"].items()})
    print("Samples(head):", inter_m_mi_BRCA["Samples"][:5])

    print(f"------------ Start: ({ynum}yr | endpoint={endpoint}) ------------")

    assert "X_blocks_all" in inter_m_mi_BRCA and inter_m_mi_BRCA["X_blocks_all"] is not None, \
        "Missing X_blocks_all"
    assert "dag_pca_dims" in inter_m_mi_BRCA and inter_m_mi_BRCA["dag_pca_dims"] is not None, \
        "Missing dag_pca_dims"
    assert "proj_pairs" in inter_m_mi_BRCA and inter_m_mi_BRCA["proj_pairs"] is not None, \
        "Missing proj_pairs"

    assert ("K_base_list" not in inter_m_mi_BRCA) or (inter_m_mi_BRCA["K_base_list"] is None), \
        "K_base_list exists but should be None/absent (avoid leakage)."
    assert ("kernel_tags" not in inter_m_mi_BRCA) or (inter_m_mi_BRCA["kernel_tags"] is None), \
        "kernel_tags exists but should be None/absent (avoid leakage)."

    assert endpoint in {"OS", "DSS", "DFI", "PFI"}, \
        f"endpoint must be one of OS/DSS/DFI/PFI. got={endpoint}"

    for LR in lr_list:
        for DR in dr_list:
            print(f"\n==== RUN: lr={LR}, dr={DR} ====")
            res_df, score_dict = run_BRCA_inter_cv(
                data=inter_m_mi_BRCA,
                years=ynum,
                target=endpoint,
                lr=LR,
                dr=DR,
                fn=folds,
                tn=n_repeats,
            )
            print("Result summary (head):")
            print(res_df.head())
            print(f"\n==== FINISH: lr={LR}, dr={DR} ====")

    print(f"------------ Finish: ({ynum}yr | endpoint={endpoint}) ------------")
    print("================= All trainings are over. =================")


if __name__ == "__main__":
    main(ARGS)
