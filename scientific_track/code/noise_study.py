"""ANNNI phase boundaries under two error classes, and their recovery.

Arms (same optimized HVA circuit at every grid point):
  clean                  ideal circuit
  depol p                DepolarizingChannel(p) on the target after every CNOT of every RZZ (challenge convention)
  drift free             coherent calibration error: RX angle x (1+eps_i), RZZ angle x (1+delta_k), drifting over the scan
  drift maintained       same drift, with TrueLoop holding eps and delta in-band from one parallel probe per cycle
  both free / maintained depolarizing p and drift together
The scan visits the 225 grid points as consecutive cycles; the drift is one trajectory shared by every arm."""
import os
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
import sys, json, pickle, time, numpy as np
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim import annni_matrix, exact_ground, observables_psi, observables_rho, apply_1q, apply_cnot, depolarize, rx, _zb
from hva import hva_state, N, LAYERS, KAPPAS, HS, _PAIRS
try:
    from trueloop import StabilityContract
except ImportError:
    from stability_contract import StabilityContract

ZB = _zb(N); DIM = 2 ** N

def hva_rho(params, p, eps, delta):
    """Density-matrix run of the HVA circuit with RZZ = CNOT.RZ.CNOT and depolarizing after each CNOT."""
    prm = np.asarray(params).reshape(LAYERS, 3); plus = np.full(DIM, 1 / np.sqrt(DIM), complex); rho = np.outer(plus, plus.conj())
    for l in range(LAYERS):
        g1, g2, b = prm[l]
        for k, (i, j) in enumerate(_PAIRS):
            phi = 2 * (g1 if k < N else g2) * (1 + delta[k])          # RZZ(phi) = exp(-i phi/2 ZZ) ; exp(-i g ZZ) -> phi = 2g
            rho = apply_cnot(rho, i, j, N); rho = depolarize(rho, p, j, N)
            v = np.exp(-0.5j * phi * ZB[:, j]); rho = rho * np.outer(v, v.conj())
            rho = apply_cnot(rho, i, j, N); rho = depolarize(rho, p, j, N)
        for i in range(N): rho = apply_1q(rho, rx(2 * b * (1 + eps[i])), i, N)
    return rho

# ----------------------------------------------------------------------------- drift trajectory over the scan
def drift_trajectory(cycles, rms, seed, warm=20):
    """eps_i(t) = common(t) + independent_i(t); delta_k(t) likewise. Each a static offset plus a mean-reverting
    wander (lambda = 0.05 per cycle). Total RMS per channel ~ rms, split equally between common and independent."""
    rng = np.random.default_rng(seed); T = cycles + warm; s = rms / np.sqrt(2)
    def channel(n):
        static = rng.normal(0, s / np.sqrt(2), n); w = np.zeros(n); out = np.zeros((T, n))
        step = s / np.sqrt(2) * np.sqrt(2 * 0.05)                       # stationary std of the wander = s/sqrt(2)
        for t in range(T):
            w = 0.95 * w + step * rng.standard_normal(n); out[t] = static + w
        return out
    eps = channel(1) + channel(N); delta = channel(1) + channel(2 * N)
    return eps, delta                                                      # shapes (T, N), (T, 2N); first `warm` rows are warm-up

# ----------------------------------------------------------------------------- in-band probes
SHOTS = 4000; AMP = 3                                                        # three-fold amplification: resolves 0.3% at 4,000 shots, captures errors to +/-33%
SIGN = float(np.sign(np.sin(AMP * np.pi / 2)))                              # slope sign of the amplitude probe at this rung; pair probe has the opposite sign
def probe(eps_now, delta_now, a, b, rng):
    """One parallel acquisition: P(1) after RX(AMP*pi/2) per qubit, P(X=+1) after RZZ(AMP*pi/2) on |++> per pair, with shot noise."""
    p1 = np.sin(0.5 * AMP * (np.pi / 2) * (1 + eps_now) * a) ** 2
    px = 0.5 * (1 + np.cos(AMP * (np.pi / 2) * (1 + delta_now) * b))
    meas = np.concatenate([p1, px]); return rng.binomial(SHOTS, np.clip(meas, 0, 1)) / SHOTS

def run_maintenance(eps_traj, delta_traj, seed):
    """Run the contract across the whole trajectory; return the effective errors the workload sees in each arm."""
    rng = np.random.default_rng(seed + 7); T = eps_traj.shape[0]; n_ch = N + 2 * N
    contract = StabilityContract(n_ch, x0=[1.0] * n_ch, target=[0.5] * n_ch, bounds=[[0.7, 1.3]] * n_ch,
                                 measurement_noise=0.5 / np.sqrt(SHOTS), link_signs=[SIGN] * N + [-SIGN] * (2 * N), tolerance=0.01)
    a = np.ones(N); b = np.ones(2 * N); eff_eps = np.zeros_like(eps_traj); eff_delta = np.zeros_like(delta_traj); n_acq = 0
    for t in range(T):
        eff_eps[t] = (1 + eps_traj[t]) * a - 1; eff_delta[t] = (1 + delta_traj[t]) * b - 1   # what this cycle's workload runs with
        m = probe(eps_traj[t], delta_traj[t], a, b, rng); n_acq += 1
        cfg = np.array(contract.step(m.tolist())["config"]); a, b = cfg[:N], cfg[N:]
    return eff_eps, eff_delta, n_acq, contract.state()

# ----------------------------------------------------------------------------- classification and boundaries
def ising_line(k):
    k = np.asarray(k, float); safe = np.maximum(k, 1e-9); inside = np.clip((1 - 3 * k + 4 * k ** 2) / np.maximum(1 - k, 1e-9), 0, None)
    return np.where(k < 0.5, (1 - k) * (1 - np.sqrt(inside)) / safe, np.nan)
def kt_line(k): k = np.asarray(k, float); return np.where(k > 0.5, 1.05 * np.sqrt(np.clip((k - 0.5) * (k - 0.1), 0, None)), np.nan)

def crossing(hs, vals, thr):
    """h where vals (decreasing in h for ordered->para) crosses thr, linear interpolation; nan if never."""
    for i in range(len(hs) - 1):
        a, b = vals[i], vals[i + 1]
        if (a - thr) * (b - thr) <= 0 and a != b: return hs[i] + (thr - a) * (hs[i + 1] - hs[i]) / (b - a)
    return np.nan

def classify_map(zz2, t_f, t_a):
    return np.where(zz2 > t_f, 0, np.where(zz2 < -t_a, 1, 2))      # 0 ferro, 1 antiphase, 2 para

if __name__ == "__main__":
    RMS = float(sys.argv[1]) if len(sys.argv) > 1 else 0.03; ARMS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["clean"]
    grid = {}
    for f in ("0_4", "4_8", "8_12", "12_15"): grid.update(pickle.load(open(os.path.join(DATA, "hva_%s.pkl") % f, "rb")))
    order = [(ki, hi) for ki in range(len(KAPPAS)) for hi in range(len(HS))]; cycles = len(order); WARM = 20
    eps_traj, delta_traj = drift_trajectory(cycles, RMS, seed=11, warm=WARM)
    eff_eps, eff_delta, n_acq, st = run_maintenance(eps_traj, delta_traj, seed=11)
    free_eps, free_delta = eps_traj[WARM:], delta_traj[WARM:]; held_eps, held_delta = eff_eps[WARM:], eff_delta[WARM:]
    print("drift RMS %.1f%%: free-running eps RMS %.4f delta RMS %.4f | maintained eps RMS %.4f delta RMS %.4f | %d parallel acquisitions for %d cycles"
          % (100 * RMS, np.sqrt(np.mean(free_eps ** 2)), np.sqrt(np.mean(free_delta ** 2)), np.sqrt(np.mean(held_eps ** 2)), np.sqrt(np.mean(held_delta ** 2)), n_acq, cycles + WARM), flush=True)
    res = {}
    for arm in ARMS:
        t0 = time.perf_counter(); zz2 = np.zeros((len(KAPPAS), len(HS)))
        for c, (ki, hi) in enumerate(order):
            g = grid[(ki, hi)]; prm = g["params"]
            if arm == "clean":            o = observables_psi(hva_state(prm), N)
            elif arm.startswith("depol"): o = observables_rho(hva_rho(prm, float(arm[5:]), np.zeros(N), np.zeros(2 * N)), N)
            elif arm == "drift_free":     o = observables_psi(hva_state(prm, eps=free_eps[c], delta=free_delta[c]), N)
            elif arm == "drift_held":     o = observables_psi(hva_state(prm, eps=held_eps[c], delta=held_delta[c]), N)
            elif arm.startswith("both_free"):  o = observables_rho(hva_rho(prm, float(arm.split("_")[2]), free_eps[c], free_delta[c]), N)
            elif arm.startswith("both_held"):  o = observables_rho(hva_rho(prm, float(arm.split("_")[2]), held_eps[c], held_delta[c]), N)
            zz2[ki, hi] = o["zz_next"]
        res[arm] = zz2; print("  %-16s done in %.0f s" % (arm, time.perf_counter() - t0), flush=True)
    pickle.dump(dict(res=res, RMS=RMS, free_eps=free_eps, held_eps=held_eps, free_delta=free_delta, held_delta=held_delta),
                open(os.path.join(DATA, "noise_%s_%s.pkl") % (RMS, "_".join(ARMS)), "wb"))
