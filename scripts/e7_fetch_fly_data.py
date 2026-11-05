# -*- coding: utf-8 -*-
"""E7, шаг 1: что есть из пакетов и скачать открытые данные мухи (Zenodo 18555170)."""
import sys, os, time, urllib.request
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

for m in ("numpy", "scipy", "h5py", "pandas", "pyarrow"):
    try:
        mod = __import__(m)
        print("%-8s %s" % (m, getattr(mod, "__version__", "?")))
    except Exception:
        print("%-8s НЕТ" % m)

D = r"D:\Documents\Projects\ConnectomeLab\data\fly"
os.makedirs(D, exist_ok=True)
BASE = "https://zenodo.org/api/records/18555170/files/%s/content"
for key in ("coordinates.mat", "annotations.mat", "connectome.mat"):
    dst = os.path.join(D, key)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print("уже есть %s %.1f МБ" % (key, os.path.getsize(dst) / 1e6))
        continue
    t0 = time.time()
    try:
        urllib.request.urlretrieve(BASE % key, dst)
        print("скачано %-18s %7.1f МБ за %.1f с" % (key, os.path.getsize(dst) / 1e6, time.time() - t0))
    except Exception as exc:
        print("ОШИБКА %s: %s: %s" % (key, type(exc).__name__, exc))
