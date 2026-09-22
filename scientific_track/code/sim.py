"""ANNNI phase diagram under two error classes.

State preparation: Trotterized adiabatic ramp from the paramagnetic product state into (kappa, h).
Gates per Trotter step: RX on every qubit (transverse field), RZZ on nearest and next-nearest pairs
(each RZZ = CNOT . RZ . CNOT). Noise, exactly the challenge's convention: DepolarizingChannel(p) on the
CNOT target after every CNOT. Coherent error, the class the challenge omits: every RX angle scaled by
(1 + eps_i) for qubit i, every RZZ angle scaled by (1 + delta_pair). A density-matrix simulator in numpy
(fast at N = 8); validated against PennyLane default.mixed in validate_against_pennylane()."""
import numpy as np, itertools

# --------------------------------------------------------------------------- exact diagonalisation
def annni_matrix(n, kappa, h, periodic=True):
    dim = 2 ** n; H = np.zeros((dim, dim)); z = np.array([1.0, -1.0])
    idx = np.arange(dim); bits = ((idx[:, None] >> (n - 1 - np.arange(n))) & 1)      # qubit 0 = most significant
    zb = 1 - 2 * bits                                                                 # +1 / -1
    diag = np.zeros(dim)
    for i in range(n if periodic else n - 1): diag -= zb[:, i] * zb[:, (i + 1) % n]
    for i in range(n if periodic else n - 2): diag += kappa * zb[:, i] * zb[:, (i + 2) % n]
    H[idx, idx] = diag
    for i in range(n):                                                                # -h X_i
        flip = idx ^ (1 << (n - 1 - i)); H[idx, flip] -= h
    return H

def exact_ground(n, kappa, h):
    w, v = np.linalg.eigh(annni_matrix(n, kappa, h)); return w[0], v[:, 0]

# --------------------------------------------------------------------------- observables (starter-kit definitions)
def _zb(n):
    idx = np.arange(2 ** n); return 1 - 2 * ((idx[:, None] >> (n - 1 - np.arange(n))) & 1)

def observables_rho(rho, n):
    zb = _zb(n); d = np.real(np.diag(rho))
    zz1 = [float(np.sum(d * zb[:, i] * zb[:, (i + 1) % n])) for i in range(n)]
    zz2 = [float(np.sum(d * zb[:, i] * zb[:, (i + 2) % n])) for i in range(n)]
    x = []
    for i in range(n):
        flip = np.arange(2 ** n) ^ (1 << (n - 1 - i)); x.append(float(np.real(np.sum(rho[np.arange(2 ** n), flip]))))
    string = [zz1[i] * zz1[(i + 2) % n] for i in range(n)]
    return dict(zz_nearest=float(np.mean(zz1)), zz_next=float(np.mean(zz2)), x_mean=float(np.mean(x)),
                antiphase_string=float(np.mean(string)))

def observables_psi(psi, n): return observables_rho(np.outer(psi, psi.conj()), n)

# --------------------------------------------------------------------------- density-matrix gates
def apply_1q(rho, U, i, n):
    dim = 2 ** n; r = rho.reshape(2 ** i, 2, 2 ** (n - 1 - i), dim)
    r = np.einsum("ab,xbyz->xayz", U, r).reshape(dim, dim)
    r = r.reshape(dim, 2 ** i, 2, 2 ** (n - 1 - i)); r = np.einsum("ab,zxby->zxay", U.conj(), r)
    return r.reshape(dim, dim)

def rx(theta): c, s = np.cos(theta / 2), np.sin(theta / 2); return np.array([[c, -1j * s], [-1j * s, c]])

def apply_rzz(rho, phi, i, j, n):
    zb = _zb(n); ph = np.exp(-0.5j * phi * zb[:, i] * zb[:, j]); return rho * np.outer(ph, ph.conj())

def cnot_perm(i, j, n):
    idx = np.arange(2 ** n); ci = (idx >> (n - 1 - i)) & 1; return idx ^ (ci << (n - 1 - j))

def apply_cnot(rho, i, j, n):
    P = cnot_perm(i, j, n); return rho[np.ix_(P, P)]

X = np.array([[0, 1], [1, 0]], complex); Y = np.array([[0, -1j], [1j, 0]]); Z = np.array([[1, 0], [0, -1]], complex)
def depolarize(rho, p, i, n):
    """(1-p) rho + p/3 (X rho X + Y rho Y + Z rho Z) = (1 - 4p/3) rho + (2p/3) I (x) Tr_i(rho)."""
    if p <= 0: return rho
    dim = 2 ** n; r = rho.reshape(2 ** i, 2, 2 ** (n - 1 - i), 2 ** i, 2, 2 ** (n - 1 - i))
    tr = np.einsum("xayzaw->xyzw", r)                                  # partial trace over qubit i
    full = np.zeros_like(r); full[:, 0, :, :, 0, :] = tr; full[:, 1, :, :, 1, :] = tr
    return (1 - 4 * p / 3) * rho + (2 * p / 3) * full.reshape(dim, dim)

# --------------------------------------------------------------------------- the preparation circuit
def prepare(n, kappa, h, p=0.0, eps=None, delta=None, T=8.0, M=32, h0=4.0, decompose_zz=True):
    """Trotterized adiabatic preparation. eps: per-qubit RX amplitude error (n,). delta: per-pair RZZ error (2n,).
    With decompose_zz=True each RZZ is CNOT.RZ.CNOT and depolarizing follows each CNOT (challenge convention)."""
    eps = np.zeros(n) if eps is None else np.asarray(eps); delta = np.zeros(2 * n) if delta is None else np.asarray(delta)
    dim = 2 ** n; plus = np.full(dim, 1 / np.sqrt(dim), complex); rho = np.outer(plus, plus.conj())
    dt = T / M; pairs = [(i, (i + 1) % n, -1.0) for i in range(n)] + [(i, (i + 2) % n, kappa) for i in range(n)]
    for m in range(M):
        s = (m + 0.5) / M; J = s; hs = h0 + (h - h0) * s
        for i in range(n):                                    # e^{+i hs dt X} = RX(-2 hs dt)
            rho = apply_1q(rho, rx(-2 * hs * dt * (1 + eps[i])), i, n)
        for k, (i, j, c) in enumerate(pairs):                 # e^{-i c J dt ZZ} = RZZ(2 c J dt)
            phi = 2 * c * J * dt * (1 + delta[k])
            if decompose_zz:
                rho = apply_cnot(rho, i, j, n); rho = depolarize(rho, p, j, n)
                v = np.exp(-0.5j * phi * _zb(n)[:, j]); rho = rho * np.outer(v, v.conj())
                rho = apply_cnot(rho, i, j, n); rho = depolarize(rho, p, j, n)
            else:
                rho = apply_rzz(rho, phi, i, j, n)
    return rho

def prepare_pure(n, kappa, h, eps=None, delta=None, T=8.0, M=32, h0=4.0):
    """Same circuit as prepare() with p = 0, on a statevector (RZZ applied directly; identical unitary)."""
    eps = np.zeros(n) if eps is None else np.asarray(eps); delta = np.zeros(2 * n) if delta is None else np.asarray(delta)
    dim = 2 ** n; psi = np.full(dim, 1 / np.sqrt(dim), complex); zb = _zb(n)
    dt = T / M; pairs = [(i, (i + 1) % n, -1.0) for i in range(n)] + [(i, (i + 2) % n, kappa) for i in range(n)]
    for m in range(M):
        s_ = (m + 0.5) / M; J = s_; hs = h0 + (h - h0) * s_
        for i in range(n):
            U = rx(-2 * hs * dt * (1 + eps[i])); r = psi.reshape(2 ** i, 2, 2 ** (n - 1 - i))
            psi = np.einsum("ab,xby->xay", U, r).reshape(dim)
        for k, (i, j, c) in enumerate(pairs):
            phi = 2 * c * J * dt * (1 + delta[k]); psi = psi * np.exp(-0.5j * phi * zb[:, i] * zb[:, j])
    return psi

# --------------------------------------------------------------------------- classification
def classify(obs, s_thr):
    """ordered if the antiphase string exceeds s_thr; ferro if next-nearest ZZ > 0 else antiphase."""
    if obs["antiphase_string"] < s_thr: return 2                 # paramagnetic
    return 0 if obs["zz_next"] > 0 else 1                         # ferro / antiphase

def ising_line(k):
    k = np.asarray(k, float); safe = np.maximum(k, 1e-9)
    inside = np.clip((1 - 3 * k + 4 * k ** 2) / np.maximum(1 - k, 1e-9), 0, None)
    return np.where(k < 0.5, (1 - k) * (1 - np.sqrt(inside)) / safe, np.nan)
def kt_line(k): k = np.asarray(k, float); return np.where(k > 0.5, 1.05 * np.sqrt(np.clip((k - 0.5) * (k - 0.1), 0, None)), np.nan)

# --------------------------------------------------------------------------- validation against PennyLane
def validate_against_pennylane(n=4, M=3, p=0.05, seed=0):
    import pennylane as qml
    rng = np.random.default_rng(seed); eps = rng.normal(0, 0.05, n); delta = rng.normal(0, 0.05, 2 * n)
    kappa, h, T, h0 = 0.4, 0.8, 2.0, 4.0; dt = T / M
    pairs = [(i, (i + 1) % n, -1.0) for i in range(n)] + [(i, (i + 2) % n, kappa) for i in range(n)]
    dev = qml.device("default.mixed", wires=n)
    @qml.qnode(dev)
    def circ():
        for i in range(n): qml.Hadamard(wires=i)
        for m in range(M):
            s = (m + 0.5) / M; J = s; hs = h0 + (h - h0) * s
            for i in range(n): qml.RX(-2 * hs * dt * (1 + eps[i]), wires=i)
            for k, (i, j, c) in enumerate(pairs):
                phi = 2 * c * J * dt * (1 + delta[k])
                qml.CNOT(wires=[i, j]); qml.DepolarizingChannel(p, wires=j)
                qml.RZ(phi, wires=j)
                qml.CNOT(wires=[i, j]); qml.DepolarizingChannel(p, wires=j)
        return qml.density_matrix(wires=range(n))
    rho_pl = np.array(circ()); rho_np = prepare(n, kappa, h, p=p, eps=eps, delta=delta, T=T, M=M, h0=h0)
    return float(np.max(np.abs(rho_pl - rho_np)))

if __name__ == "__main__":
    print("max |rho_numpy - rho_pennylane| (N=4, 3 steps, p=0.05, coherent errors on):", "%.2e" % validate_against_pennylane())
    n = 8; E, psi = exact_ground(n, 0.3, 0.6); o = observables_psi(psi, n)
    print("ED N=8 (0.3,0.6): E=%.4f zz1=%.3f zz2=%.3f x=%.3f string=%.3f" % (E, o["zz_nearest"], o["zz_next"], o["x_mean"], o["antiphase_string"]))
    import time; t = time.perf_counter(); rho = prepare(n, 0.3, 0.6); el = time.perf_counter() - t
    o2 = observables_rho(rho, n); H = annni_matrix(n, 0.3, 0.6); Ec = float(np.real(np.trace(rho @ H)))
    print("Trotter-adiabatic prep (T=8, M=32): E=%.4f (ED %.4f) zz1=%.3f zz2=%.3f x=%.3f string=%.3f | %.2f s/point" % (Ec, E, o2["zz_nearest"], o2["zz_next"], o2["x_mean"], o2["antiphase_string"], el))
