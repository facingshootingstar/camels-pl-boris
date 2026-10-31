# Explore CAMELS-PL v1.0.0 (Zenodo 10.5281/zenodo.20133183): structure + Storm Boris peaks.
import os, zipfile, json, glob, urllib.request, hashlib
import pandas as pd

URL = "https://zenodo.org/records/20133183/files/CAMELS-PL.zip?download=1"
ZIP = "/kaggle/working/CAMELS-PL.zip"
ROOT = "/kaggle/working/data"
if not os.path.exists(ZIP):
    urllib.request.urlretrieve(URL, ZIP)
md5 = hashlib.md5(open(ZIP, "rb").read()).hexdigest()
print("md5", md5, "size", os.path.getsize(ZIP))
with zipfile.ZipFile(ZIP) as z:
    names = z.namelist()
    z.extractall(ROOT)
print("n files", len(names))
dirs = sorted({os.path.dirname(n) for n in names})
for d in dirs[:80]:
    print("DIR", d, sum(1 for n in names if os.path.dirname(n) == d))
for n in names[:60]:
    print("F", n)

# print heads of every csv/txt type found (first file per directory)
seen = set()
for n in names:
    d = os.path.dirname(n)
    if d in seen or not n.lower().endswith((".csv", ".txt")):
        continue
    seen.add(d)
    p = os.path.join(ROOT, n)
    try:
        df = pd.read_csv(p, nrows=5, sep=None, engine="python")
        print("=== HEAD", n, df.shape); print(df.to_string()[:1500])
    except Exception as e:
        print("ERR", n, e)
os.remove(ZIP)
json.dump({"md5": md5, "files": names}, open("/kaggle/working/manifest.json", "w"))
