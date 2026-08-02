#Remove any corrupt/partial file
python3 - <<'PY'
import glob, os, numpy as np
root = "./data/trunet_cardiac"        # <-- your --out-root
bad = []
for f in glob.glob(os.path.join(root, "*", "*.npz")):
    try:
        with np.load(f) as d:
            float(d["arr_0"].sum()); int(d["arr_1"].sum())   # force full read
    except Exception as e:
        print("corrupt:", f, "->", e); bad.append(f)
for f in bad:
    os.remove(f)
print(f"checked; removed {len(bad)} corrupt file(s)")
PY
