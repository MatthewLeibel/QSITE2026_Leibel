"""Solve one column (fixed kappa) at N with an L-layer HVA, warm-started from a solved column at a smaller N (HVA angles
transfer across N). Saves per point. Usage: python push_n.py N L kappa warm.pkl"""
import sys, pickle, time, numpy as np; sys.path.insert(0, "/home/claude/annni")
from hva_n import ANNNI
N, L, kappa, warm_path = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
HI = [int(v) for v in sys.argv[5].split(",")] if len(sys.argv) > 5 else list(range(15))
MAXITER = int(sys.argv[6]) if len(sys.argv) > 6 else 300
hs = np.linspace(0.05, 2.0, 15); m = ANNNI(N, layers=L); rng = np.random.default_rng(N)
warm = pickle.load(open(warm_path, "rb"))
out_path = "/home/claude/annni/col_N%d_L%d_k%.3f.pkl" % (N, L, kappa)
try: col = pickle.load(open(out_path, "rb"))
except Exception: col = {}
t0 = time.perf_counter(); prev = None
for hi in sorted(HI, reverse=True):
    if hi in col: prev = col[hi]["params"]; continue
    h = hs[hi]; starts = []
    if prev is not None: starts.append(prev)
    w = warm.get(hi) if isinstance(warm, dict) and hi in warm else None
    if w is not None: starts.append(np.asarray(w["params"] if isinstance(w, dict) else w))
    if not starts: starts.append(rng.normal(0, 0.15, 3 * L))
    best = (np.inf, None, 0)
    for x0 in starts:
        x, E, nf = m.vqe(kappa, h, np.asarray(x0), maxiter=MAXITER)
        if E < best[0]: best = (E, x, nf)
    E, x, nf = best; prev = x
    if N <= 16: Eg, gs = m.ground(kappa, h); zz_ed = m.zz2_of(gs)
    else: Eg, zz_ed = float("nan"), float("nan")
    col[hi] = dict(kappa=kappa, h=h, params=x, E=E, Eg=Eg, zz2=m.zz2_of(m.state(x)), zz2_ed=zz_ed, nfev=nf)
    pickle.dump(col, open(out_path, "wb"))
    print("  N=%d L=%d kappa=%.2f h=%.2f: dE/E %.5f zz2 %.3f (ED %.3f) nfev %d  [%.0f s]" % (N, L, kappa, h, (E - Eg) / abs(Eg) if Eg == Eg else float("nan"), col[hi]["zz2"], zz_ed, nf, time.perf_counter() - t0), flush=True)
