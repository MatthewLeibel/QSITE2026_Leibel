"""Continuous operation: many passes over the phase diagram while the calibration drifts, with a calibration jump
mid-run, nothing re-optimized, no supervision. Free-running vs maintained on the same drift. N-generic."""
import os
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
import sys, pickle, glob, time, json, numpy as np
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hva_n import ANNNI
try:
    from trueloop import StabilityContract
except ImportError:
    from stability_contract import StabilityContract
from noise_study import ising_line, kt_line, crossing

SHOTS, AMP = 4000, 3; SIGN = float(np.sign(np.sin(AMP * np.pi / 2)))
HS = np.linspace(0.05, 2.0, 15)

def load_points(n):
    if n == 8:
        g = {}
        for f in ("0_4", "4_8", "8_12", "12_15"): g.update(pickle.load(open(os.path.join(DATA, "hva_%s.pkl") % f, "rb")))
        K = np.linspace(0, 1, 15); return [(K[ki], HS[hi], g[(ki, hi)]["params"]) for ki in range(15) for hi in range(15)], K
    pts = []; ks = []
    for f in sorted(glob.glob(os.path.join(DATA, "grid_N%d_k*.pkl") % n)):
        col = pickle.load(open(f, "rb")); k = list(col.values())[0]["kappa"]; ks.append(k)
        pts += [(k, HS[hi], col[hi]["params"]) for hi in range(15)]
    return pts, np.array(ks)

def probe(eps, delta, a, b, rng):
    p1 = np.sin(0.5 * AMP * (np.pi / 2) * (1 + eps) * a) ** 2; px = 0.5 * (1 + np.cos(AMP * (np.pi / 2) * (1 + delta) * b))
    return rng.binomial(SHOTS, np.clip(np.concatenate([p1, px]), 0, 1)) / SHOTS

def run(n, passes, rms, jump_at_pass, seed=5):
    m = ANNNI(n); pts, ks = load_points(n); P = len(pts); cycles = passes * P; rng = np.random.default_rng(seed)
    s = rms / np.sqrt(2); lam = 0.05; step = s / np.sqrt(2) * np.sqrt(2 * lam)
    def new_static(k): return rng.normal(0, s / np.sqrt(2), k)
    st_c, st_e, st_d1, st_d = new_static(1), new_static(n), new_static(1), new_static(2 * n)
    w_c, w_e, w_d1, w_d = np.zeros(1), np.zeros(n), np.zeros(1), np.zeros(2 * n)
    nch = n + 2 * n; contract = StabilityContract(nch, x0=[1.0] * nch, target=[0.5] * nch, bounds=[[0.7, 1.3]] * nch,
                                                  measurement_noise=0.5 / np.sqrt(SHOTS), tolerance=0.01)   # signs learned by preflight, not declared
    a = np.ones(n); b = np.ones(2 * n); host = []; rec = dict(free=[], held=[], res_free=[], res_held=[], cyc=[])
    zz_free = np.zeros((passes, P)); zz_held = np.zeros((passes, P)); zz_clean = np.array([m.zz2_of(m.state(p[2])) for p in pts])
    for t in range(cycles):
        ps, c = divmod(t, P)
        if jump_at_pass is not None and t == jump_at_pass * P + P // 2:      # calibration event: every static offset redrawn
            st_c, st_e, st_d1, st_d = new_static(1), new_static(n), new_static(1), new_static(2 * n)
        w_c = 0.95 * w_c + step * rng.standard_normal(1); w_e = 0.95 * w_e + step * rng.standard_normal(n)
        w_d1 = 0.95 * w_d1 + step * rng.standard_normal(1); w_d = 0.95 * w_d + step * rng.standard_normal(2 * n)
        eps = st_c + w_c + st_e + w_e; delta = st_d1 + w_d1 + st_d + w_d
        eff_e = (1 + eps) * a - 1; eff_d = (1 + delta) * b - 1
        k, h, prm = pts[c]
        zz_free[ps, c] = m.zz2_of(m.state(prm, eps=eps, delta=delta)); zz_held[ps, c] = m.zz2_of(m.state(prm, eps=eff_e, delta=eff_d))
        meas = probe(eps, delta, a, b, rng); t0 = time.perf_counter(); out = contract.step(meas.tolist()); host.append(time.perf_counter() - t0)
        cfg = np.array(out["config"]); a, b = cfg[:n], cfg[n:]
        rec["res_free"].append(float(np.sqrt(np.mean(np.concatenate([eps, delta]) ** 2)))); rec["res_held"].append(float(np.sqrt(np.mean(np.concatenate([eff_e, eff_d]) ** 2))))
    return dict(n=n, pts=pts, ks=ks, zz_clean=zz_clean, zz_free=zz_free, zz_held=zz_held, host_us=1e6 * np.median(host), host_max_us=1e6 * np.max(host),
                res_free=np.array(rec["res_free"]), res_held=np.array(rec["res_held"]), state=contract.state(), cycles=cycles, P=P, nch=nch)

def boundaries(zz_row, pts, ks):
    """renormalized boundary per column: ferro->para for kappa<0.5, antiphase->para for kappa>0.5; thresholds fixed from the clean N=8 study."""
    out = {}
    for k in ks:
        idx = [i for i, p in enumerate(pts) if abs(p[0] - k) < 1e-9]; col = zz_row[idx]; ref = abs(col[0])
        if ref < 0.2: out[k] = np.nan; continue
        z = col / ref
        out[k] = crossing(HS, z, 0.579) if k < 0.5 else crossing(HS, -z, 0.802)
    return out

if __name__ == "__main__":
    n = int(sys.argv[1]); passes = int(sys.argv[2]); rms = float(sys.argv[3]); jump = int(sys.argv[4]) if len(sys.argv) > 4 else None
    t0 = time.perf_counter(); R = run(n, passes, rms, jump); el = time.perf_counter() - t0
    pts, ks = R["pts"], R["ks"]; b0 = boundaries(R["zz_clean"], pts, ks)
    print("N=%d: %d maintained channels, %d passes x %d points = %d cycles, drift %.0f%%, calibration jump at pass %s | %.0f s" % (n, R["nch"], passes, R["P"], R["cycles"], 100 * rms, jump, el))
    print("host time per cycle: median %.0f us, max %.0f us | residual angle error RMS: free %.4f, maintained %.4f" % (R["host_us"], R["host_max_us"], np.sqrt(np.mean(R["res_free"] ** 2)), np.sqrt(np.mean(R["res_held"] ** 2))))
    print("\npass | free: mis%%  mean|dh*|  | maintained: mis%%  mean|dh*|  | residual free / maintained")
    T_F, T_A = 0.559, 0.802
    def cls(z): return np.where(z > T_F, 0, np.where(z < -T_A, 1, 2))
    ref = cls(R["zz_clean"]); bf_all = []; bh_all = []
    for ps in range(passes):
        bf = boundaries(R["zz_free"][ps], pts, ks); bh = boundaries(R["zz_held"][ps], pts, ks)
        df = np.nanmean([abs(bf[k] - b0[k]) for k in ks]); dh = np.nanmean([abs(bh[k] - b0[k]) for k in ks]); bf_all.append([bf[k] for k in ks]); bh_all.append([bh[k] for k in ks])
        sl = slice(ps * R["P"], (ps + 1) * R["P"])
        print("  %2d | %5.1f   %.3f     | %5.1f   %.3f     | %.4f / %.4f%s" % (ps, 100 * np.mean(cls(R["zz_free"][ps]) != ref), df, 100 * np.mean(cls(R["zz_held"][ps]) != ref), dh,
              np.sqrt(np.mean(R["res_free"][sl] ** 2)), np.sqrt(np.mean(R["res_held"][sl] ** 2)), "   <- jump" if ps == jump else ""))
    bf_all = np.array(bf_all); bh_all = np.array(bh_all)
    print("\npass-to-pass std of the boundary (mean over columns): free %.4f, maintained %.4f" % (np.nanmean(np.nanstd(bf_all, axis=0)), np.nanmean(np.nanstd(bh_all, axis=0))))
    if jump is not None:
        t_j = jump * R["P"] + R["P"] // 2; rh = R["res_held"]; rec_t = next((t for t in range(t_j, R["cycles"]) if rh[t] < 1.5 * np.sqrt(np.mean(rh[:t_j] ** 2))), None)
        print("after the calibration jump at cycle %d: maintained residual %.4f -> back within 1.5x of its pre-jump level after %s cycles" % (t_j, rh[t_j], (rec_t - t_j) if rec_t else "n/a"))
    st = R["state"]; keys = [k for k in st.keys() if k not in ("phi", "config", "x")][:12]
    print("certified record (contract.state keys):", ", ".join("%s=%s" % (k, st[k]) for k in keys if not isinstance(st[k], (list, dict))))
    pickle.dump(R, open(os.path.join(DATA, "continuous_N%d_p%d_r%.2f.pkl") % (n, passes, rms), "wb"))
