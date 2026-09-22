import os
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
import sys, pickle, glob, numpy as np
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from hva import KAPPAS, HS
from noise_study import ising_line, kt_line, crossing
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt


def value_at(col, h_star): return np.interp(h_star, HS, col) if np.isfinite(h_star) and HS[0] <= h_star <= HS[-1] else np.nan
def thresholds(z):
    tf = [value_at(z[ki], ising_line(k)) for ki, k in enumerate(KAPPAS) if 0 < k < 0.5]
    ta = [-value_at(z[ki], kt_line(k)) for ki, k in enumerate(KAPPAS) if k > 0.5]
    return float(np.nanmedian(tf)), float(np.nanmedian(ta))
def boundaries(z, T_F, T_A):
    hf = np.array([crossing(HS, z[ki], T_F) if KAPPAS[ki] < 0.5 else np.nan for ki in range(len(KAPPAS))])
    ha = np.array([crossing(HS, -z[ki], T_A) if KAPPAS[ki] > 0.5 else np.nan for ki in range(len(KAPPAS))])
    return hf, ha
def classify(z, T_F, T_A): return np.where(z > T_F, 0, np.where(z < -T_A, 1, 2))
def renorm(z):
    """Depolarizing-aware observable: each column scaled by its deep-ordered value at h = 0.05 (the standard
    'measure the decay on a reference point' mitigation). Coherent drift changes shape, so this cannot undo it."""
    ref = np.abs(z[:, :1]); out = z / np.maximum(ref, 1e-6); out[(ref < 0.2).ravel(), :] = np.nan   # no ordered reference at the multicritical column
    return out


def main():
    res = {}
    for f in glob.glob(os.path.join(DATA, "noise_0.03_*.pkl")): res.update(pickle.load(open(f, "rb"))["res"])
    r6 = pickle.load(open(os.path.join(DATA, "noise_0.06_clean_drift_free_drift_held.pkl"), "rb"))
    res["drift6_free"] = r6["res"]["drift_free"]; res["drift6_held"] = r6["res"]["drift_held"]
    clean = res["clean"]
    names = {"clean": "clean", "depol0.01": "depolarizing p=0.01", "depol0.05": "depolarizing p=0.05",
             "drift_free": "coherent drift 3%, free-running", "drift_held": "coherent drift 3%, TrueLoop",
             "drift6_free": "coherent drift 6%, free-running", "drift6_held": "coherent drift 6%, TrueLoop",
             "both_free_0.05": "drift 3% + depol 0.05, free", "both_held_0.05": "drift 3% + depol 0.05, TrueLoop"}
    order = ["clean", "depol0.01", "depol0.05", "drift_free", "drift_held", "drift6_free", "drift6_held", "both_free_0.05", "both_held_0.05"]

    T_F, T_A = thresholds(clean); ref = classify(clean, T_F, T_A); hf0, ha0 = boundaries(clean, T_F, T_A)
    Tn_F, Tn_A = thresholds(renorm(clean)); refn = classify(renorm(clean), Tn_F, Tn_A); hfn0, han0 = boundaries(renorm(clean), Tn_F, Tn_A)
    print("thresholds on raw correlator: ferro %.3f anti %.3f | on renormalized: ferro %.3f anti %.3f" % (T_F, T_A, Tn_F, Tn_A))
    print("\n%-34s | RAW: misclass  mean|dh_f|  mean|dh_a|  max|dh| | RENORMALIZED: misclass  mean|dh_f|  mean|dh_a|  max|dh|" % "arm")
    rows = {}
    for arm in order:
        z = res[arm]; hf, ha = boundaries(z, T_F, T_A); mis = np.mean(classify(z, T_F, T_A) != ref)
        zn = renorm(z); hfn, han = boundaries(zn, Tn_F, Tn_A); misn = np.mean(classify(zn, Tn_F, Tn_A) != refn)
        d = np.concatenate([hf - hf0, ha - ha0]); dn = np.concatenate([hfn - hfn0, han - han0])
        rows[arm] = dict(hf=hf, ha=ha, hfn=hfn, han=han, mis=mis, misn=misn, d=d, dn=dn)
        f = lambda x: "%.3f" % x if np.isfinite(x) else "  --"
        print("%-34s | %6.1f%%   %6s     %6s     %6s   | %6.1f%%   %6s     %6s     %6s" % (names[arm], 100 * mis, f(np.nanmean(np.abs(hf - hf0))), f(np.nanmean(np.abs(ha - ha0))), f(np.nanmax(np.abs(d))),
              100 * misn, f(np.nanmean(np.abs(hfn - hfn0))), f(np.nanmean(np.abs(han - han0))), f(np.nanmax(np.abs(dn)))))

    ki3 = int(np.argmin(np.abs(KAPPAS - 0.3))); ki7 = int(np.argmin(np.abs(KAPPAS - 0.7)))
    print("\nrenormalized boundaries: ferro->para at kappa=%.2f (Ising line %.3f) | antiphase->para at kappa=%.2f (KT line %.3f)" % (KAPPAS[ki3], ising_line(KAPPAS[ki3]), KAPPAS[ki7], kt_line(KAPPAS[ki7])))
    for arm in order: print("   %-34s h* = %s | h* = %s" % (names[arm], "%.3f" % rows[arm]["hfn"][ki3] if np.isfinite(rows[arm]["hfn"][ki3]) else " --", "%.3f" % rows[arm]["han"][ki7] if np.isfinite(rows[arm]["han"][ki7]) else " --"))
    for a, b, lab in (("drift_free", "drift_held", "3%"), ("drift6_free", "drift6_held", "6%")):
        df, dh = np.nanmean(np.abs(rows[a]["dn"])), np.nanmean(np.abs(rows[b]["dn"]))
        print("drift %s: mean |renormalized boundary shift| free %.3f -> maintained %.3f (%.0f%% removed); misclassified %.1f%% -> %.1f%%" % (lab, df, dh, 100 * (1 - dh / df), 100 * rows[a]["misn"], 100 * rows[b]["misn"]))
    print("signal amplitude (mean |zz2| at h=0.05): clean %.3f, p=0.01 %.3f, p=0.05 %.3f" % tuple(np.mean(np.abs(res[a][:, 0])) for a in ("clean", "depol0.01", "depol0.05")))

    # figures: raw maps, renormalized maps, boundary shifts
    K, Hm = np.meshgrid(KAPPAS, HS, indexing="ij"); kk = np.linspace(0.01, 0.99, 200)
    def panel(ax, z, title, hf, ha):
        ax.pcolormesh(K, Hm, z, cmap="coolwarm", vmin=-1, vmax=1, shading="auto"); ax.plot(kk, ising_line(kk), "k--", lw=1); ax.plot(kk, kt_line(kk), "k--", lw=1)
        ax.plot(KAPPAS, hf, "ko-", ms=3, lw=1.1); ax.plot(KAPPAS, ha, "ks-", ms=3, lw=1.1); ax.set_title(title, fontsize=9); ax.set_xlabel("kappa"); ax.set_ylabel("h"); ax.set_ylim(HS[0], HS[-1])
    sel = ["clean", "depol0.01", "depol0.05", "drift6_free", "drift6_held", "both_free_0.05", "both_held_0.05"]
    fig, axes = plt.subplots(2, 7, figsize=(24, 7.2))
    for j, arm in enumerate(sel):
        panel(axes[0, j], res[arm], names[arm] + " (raw)", rows[arm]["hf"], rows[arm]["ha"])
        panel(axes[1, j], renorm(res[arm]), names[arm] + " (renormalized)", rows[arm]["hfn"], rows[arm]["han"])
    fig.suptitle("ANNNI N=8, VQE-prepared (4-layer HVA, 128 CNOTs): <Z_i Z_{i+2}>. Dashed: analytic lines. Markers: measured boundaries.", fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(DATA, "phase_diagrams.png"), dpi=105)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for arm, c in (("depol0.01", "tab:green"), ("drift_free", "tab:orange"), ("drift_held", "tab:cyan"), ("drift6_free", "tab:red"), ("drift6_held", "tab:blue")):
        ax[0].plot(KAPPAS, rows[arm]["hfn"] - hfn0, "o-", ms=3, label=names[arm], color=c); ax[1].plot(KAPPAS, rows[arm]["han"] - han0, "s-", ms=3, label=names[arm], color=c)
    for a, t in zip(ax, ("ferro -> para boundary shift (renormalized)", "antiphase -> para boundary shift (renormalized)")):
        a.axhline(0, color="k", lw=.8); a.set_xlabel("kappa"); a.set_ylabel("delta h*"); a.set_title(t, fontsize=10); a.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(DATA, "boundary_shifts.png"), dpi=110)
    pickle.dump(dict(rows=rows, res=res, T=(T_F, T_A, Tn_F, Tn_A)), open(os.path.join(DATA, "analysis.pkl"), "wb")); print("figures saved")

if __name__ == "__main__":
    main()
