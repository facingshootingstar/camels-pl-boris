"""Merge the outputs of the Kaggle training kernels (A, B and, if present, C) into results/kaggle_out/.

Inputs : results/full_{A,B,C}/{eval_ref.npz, sim_*.npy, train_logs.json}  (from `kaggle kernels output`)
Outputs: results/kaggle_out/{eval_ref.npz, sim_*.npy, train_logs.json}
Every kernel writes the same eval_ref.npz; the merge fails if any two differ.
"""
import sys, json, glob, os, shutil
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results"); KO = os.path.join(RES, "kaggle_out")
parts = [p for p in (os.path.join(RES, "full_" + k) for k in "ABC") if os.path.isdir(p)]
os.makedirs(KO, exist_ok=True)

ref = np.load(os.path.join(parts[0], "eval_ref.npz"), allow_pickle=True)
for p in parts[1:]:
    other = np.load(os.path.join(p, "eval_ref.npz"), allow_pickle=True)
    assert sorted(ref.files) == sorted(other.files), p
    for k in ref.files:
        a, b = ref[k], other[k]
        same = np.array_equal(a, b, equal_nan=True) if a.dtype.kind == "f" else np.array_equal(a, b)
        assert same, f"eval_ref.npz key {k} differs in {p}"
shutil.copy2(os.path.join(parts[0], "eval_ref.npz"), KO)

logs = {}
for p in parts:
    for f in glob.glob(os.path.join(p, "sim_*.npy")):
        shutil.copy2(f, KO)
    part_logs = json.load(open(os.path.join(p, "train_logs.json")))
    assert not set(part_logs) & set(logs), f"duplicate runs in {p}"
    logs.update(part_logs)
json.dump(logs, open(os.path.join(KO, "train_logs.json"), "w"), indent=1)
print("merged", [os.path.basename(p) for p in parts], "->", len(logs), "runs,",
      len(glob.glob(os.path.join(KO, "sim_*.npy"))), "simulation files")
