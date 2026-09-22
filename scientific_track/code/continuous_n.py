"""Complexity decoupled from the state: the host maintains 3N channels while the quantum state has 2^N amplitudes.
Continuous scans (passes over a column of the phase diagram) with the calibration drifting at 6%, a calibration event
mid-run, no re-optimization. Arms: free-running, engineer recalibrates every K cycles, stability contract in-band."""
import sys, pickle, glob, time, numpy as np; sys.path.insert(0, "/home/claude/annni"); sys.path.insert(0, "/tmp/off/trueloop_offline_v310")
from hva_n import ANNNI
from noise_study import crossing
try: from trueloop import StabilityContract
except ImportError: from stability_contract import StabilityContract
SHOTS, AMP = 4000, 3; HS = np.linspace(0.05, 2.0, 15)

def probe(eps, delta, a, b, rng):
    p1 = np.sin(0.5 * AMP * (np.pi / 2) * (1 + eps) * a) ** 2; px = 0.5 * (1 + np.cos(AMP * (np.pi / 2) * (1 + delta) * b))
    return rng.binomial(SHOTS, np.clip(np.concatenate([p1, px]), 0, 1)) / SHOTS

def run(n, layers, col, passes, rms=0.06, jump_pass=None, engineer_every=None, seed=5):
    m = ANNNI(n, layers=layers); his = sorted(col); pts = [(col[hi]["h"], col[hi]["params"]) for hi in his]; P = len(pts); cycles = passes * P
    rng = np.random.default_rng(seed); s = rms / np.sqrt(2); lam = 0.05; step = s / np.sqrt(2) * np.sqrt(2 * lam)
    def new_static(k): return rng.normal(0, s / np.sqrt(2), k)
    st = [new_static(1), new_static(n), new_static(1), new_static(2 * n)]; w = [np.zeros(1), np.zeros(n), np.zeros(1), np.zeros(2 * n)]
    nch = 3 * n
    arms = {"free": None, "contract": StabilityContract(nch, x0=[1.0] * nch, target=[0.5] * nch, bounds=[[0.85, 1.15]] * nch, measurement_noise=0.5 / np.sqrt(SHOTS), tolerance=0.01)}   # bounds x drift inside the rung's monotone range
    if engineer_every: arms["engineer"] = None
    corr = {k: (np.ones(n), np.ones(2 * n)) for k in arms}; zz = {k: np.zeros((passes, P)) for k in arms}; res = {k: [] for k in arms}; host = []
    clean = np.array([m.zz2_of(m.state(prm)) for _, prm in pts]); t_state = []
    for t in range(cycles):
        ps, c = divmod(t, P)
        if jump_pass is not None and t == jump_pass * P + P // 2: st = [new_static(1), new_static(n), new_static(1), new_static(2 * n)]
        for i in range(4): w[i] = 0.95 * w[i] + step * rng.standard_normal(w[i].shape)
        eps = st[0] + w[0] + st[1] + w[1]; delta = st[2] + w[2] + st[3] + w[3]
        if "engineer" in arms and t % engineer_every == 0: corr["engineer"] = (1 / (1 + eps), 1 / (1 + delta))     # a full recalibration at the visit
        h, prm = pts[c]
        for k in arms:
            a, b = corr[k]; e_eff = (1 + eps) * a - 1; d_eff = (1 + delta) * b - 1
            t0 = time.perf_counter(); zz[k][ps, c] = m.zz2_of(m.state(prm, eps=e_eff, delta=d_eff)); t_state.append(time.perf_counter() - t0)
            res[k].append(float(np.sqrt(np.mean(np.concatenate([e_eff, d_eff]) ** 2))))
        meas = probe(eps, delta, *corr["contract"], rng); t0 = time.perf_counter(); out = arms["contract"].step(meas.tolist()); host.append(time.perf_counter() - t0)
        cfg = np.array(out["config"]); corr["contract"] = (cfg[:n], cfg[n:])
    hs = np.array([h for h, _ in pts])
    def boundary(row):
        ref = abs(row[0]); return crossing(hs, row / ref, 0.579) if ref > 0.2 else np.nan
    b0 = boundary(clean); bnd = {k: np.array([boundary(zz[k][p]) for p in range(passes)]) for k in arms}
    return dict(n=n, channels=nch, state_dim=2 ** n, cycles=cycles, host_us=1e6 * np.median(host), state_ms=1e3 * np.median(t_state),
                residual={k: float(np.sqrt(np.mean(np.array(v) ** 2))) for k, v in res.items()}, b0=b0, boundary=bnd,
                shift={k: float(np.nanmean(np.abs(bnd[k] - b0))) for k in arms}, std={k: float(np.nanstd(bnd[k])) for k in arms}, zz=zz, clean=clean, hs=hs)

if __name__ == "__main__":
    which = sys.argv[1]
    results = {}
    if which == "n16":
        results[16] = run(16, 8, pickle.load(open("/home/claude/annni/col_N16_L8_k0.286.pkl", "rb")), passes=16, jump_pass=8, engineer_every=50)
    elif which == "small":
        g8 = {}
        for f in ("0_4", "4_8", "8_12", "12_15"): g8.update(pickle.load(open("/home/claude/annni/hva_%s.pkl" % f, "rb")))
        col8 = {hi: g8[(4, hi)] for hi in range(15)}                                    # kappa = 0.286, N=8, four layers (the notebook's circuits)
        results[8] = run(8, 4, col8, passes=10, jump_pass=5, engineer_every=50)
        results[12] = run(12, 8, pickle.load(open("/home/claude/annni/col_N12_L8_k0.286.pkl", "rb")), passes=10, jump_pass=5, engineer_every=50)
        results[16] = run(16, 8, pickle.load(open("/home/claude/annni/col_N16_L8_k0.286.pkl", "rb")), passes=16, jump_pass=8, engineer_every=50)
    else:
        col16 = pickle.load(open("/home/claude/annni/col_N16_L8_k0.286.pkl", "rb"))
        col20 = {hi: dict(h=v["h"], params=v["params"]) for hi, v in col16.items()}       # HVA angles transferred from N=16 without re-optimization
        results[20] = run(20, 8, col20, passes=2, jump_pass=None, engineer_every=None)
    for n, r in results.items():
        print("N=%2d | state dim %9d | channels %2d | %d cycles | host %6.0f us/cycle | state eval %.0f ms | residual free %.4f engineer %.4f contract %.4f | boundary shift free %.3f engineer %.3f contract %.3f | pass std free %.3f contract %.3f" % (
            n, r["state_dim"], r["channels"], r["cycles"], r["host_us"], r["state_ms"], r["residual"]["free"], r["residual"].get("engineer", float("nan")), r["residual"]["contract"],
            r["shift"]["free"], r["shift"].get("engineer", float("nan")), r["shift"]["contract"], r["std"]["free"], r["std"]["contract"]), flush=True)
    old = {}
    try: old = pickle.load(open("/home/claude/annni/decoupling.pkl", "rb"))
    except Exception: pass
    old.update(results); pickle.dump(old, open("/home/claude/annni/decoupling.pkl", "wb"))
