"""HVA state preparation for the ANNNI model: from |+>^N, layers of exp(-i g1 sum ZZ_nn) exp(-i g2 sum ZZ_nnn) exp(-i b sum X).
Adjoint gradients; L-BFGS; warm starts down the field. The same circuit, with coherent errors on its RX and RZZ angles
and depolarizing noise after every CNOT of each RZZ, is what the noise study runs."""
import os
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
import sys, numpy as np
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from sim import exact_ground, observables_psi, annni_matrix, _zb
N, LAYERS = 8, 4
KAPPAS = np.linspace(0.0, 1.0, 15); HS = np.linspace(0.05, 2.0, 15)
_ZB = _zb(N); _DIM = 2 ** N
_ZZ1 = sum(_ZB[:, i] * _ZB[:, (i + 1) % N] for i in range(N)).astype(float)     # diagonal of sum ZZ_nn
_ZZ2 = sum(_ZB[:, i] * _ZB[:, (i + 2) % N] for i in range(N)).astype(float)     # diagonal of sum ZZ_nnn
_PAIRS = [(i, (i + 1) % N) for i in range(N)] + [(i, (i + 2) % N) for i in range(N)]

def _rx_all(psi, angles):
    """Product of RX(angle_i) on qubit i."""
    for i in range(N):
        c, s = np.cos(angles[i] / 2), np.sin(angles[i] / 2); U = np.array([[c, -1j * s], [-1j * s, c]])
        r = psi.reshape(2 ** i, 2, 2 ** (N - 1 - i)); psi = np.einsum("ab,xby->xay", U, r).reshape(_DIM)
    return psi

def hva_state(params, eps=None, delta=None, kappa=None):
    """params: (LAYERS, 3) = (g1, g2, b). Coherent errors: RX angle 2b(1+eps_i); pair k angle scaled by (1+delta_k)."""
    p = np.asarray(params).reshape(LAYERS, 3); eps = np.zeros(N) if eps is None else np.asarray(eps)
    delta = np.zeros(2 * N) if delta is None else np.asarray(delta)
    psi = np.full(_DIM, 1 / np.sqrt(_DIM), complex)
    for l in range(LAYERS):
        g1, g2, b = p[l]
        if delta is None or not np.any(delta):
            psi = psi * np.exp(-1j * (g1 * _ZZ1 + g2 * _ZZ2))
        else:
            ph = np.zeros(_DIM)
            for k, (i, j) in enumerate(_PAIRS):
                g = g1 if k < N else g2; ph += g * (1 + delta[k]) * _ZB[:, i] * _ZB[:, j]
            psi = psi * np.exp(-1j * ph)
        psi = _rx_all(psi, 2 * b * (1 + eps))
    return psi

_X = np.array([[0, 1], [1, 0]], complex)
def energy_and_grad(params, H):
    p = np.asarray(params).reshape(LAYERS, 3); psi = np.full(_DIM, 1 / np.sqrt(_DIM), complex); states = []
    for l in range(LAYERS):
        g1, g2, b = p[l]; psi = psi * np.exp(-1j * (g1 * _ZZ1 + g2 * _ZZ2)); states.append(psi); psi = _rx_all(psi, 2 * b * np.ones(N))
    E = float(np.real(np.vdot(psi, H @ psi))); lam = H @ psi; phi = psi; g = np.zeros((LAYERS, 3))
    for l in reversed(range(LAYERS)):
        g1, g2, b = p[l]
        # d/db of exp(-i b sum X): 2 Im <lam| sum X_i |phi_after>
        sx = np.zeros(_DIM, complex)
        for i in range(N): sx += _rx_all_single(phi, _X, i)
        g[l, 2] = 2 * np.imag(np.vdot(lam, sx))
        phi = _rx_all(phi, -2 * b * np.ones(N)); lam = _rx_all(lam, -2 * b * np.ones(N))
        g[l, 0] = 2 * np.imag(np.vdot(lam, _ZZ1 * phi)); g[l, 1] = 2 * np.imag(np.vdot(lam, _ZZ2 * phi))
        ph = np.exp(1j * (g1 * _ZZ1 + g2 * _ZZ2)); phi = phi * ph; lam = lam * ph
    return E, g.reshape(-1)

def _rx_all_single(psi, U, i):
    r = psi.reshape(2 ** i, 2, 2 ** (N - 1 - i)); return np.einsum("ab,xby->xay", U, r).reshape(_DIM)

def vqe(H, x0, maxiter=400):
    from scipy.optimize import minimize
    r = minimize(lambda x: energy_and_grad(x, H), x0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-13, "gtol": 1e-9})
    return r.x, float(r.fun)

def solve_column(ki, rng, verbose=False):
    kappa = KAPPAS[ki]; prev = None; col = {}
    for hi in range(len(HS) - 1, -1, -1):
        h = HS[hi]; H = annni_matrix(N, kappa, h); Eg, gs = exact_ground(N, kappa, h)
        starts = [prev] if prev is not None else []
        starts += [rng.normal(0, 0.2, 3 * LAYERS) for _ in range(2 if prev is not None else 4)]
        best = (np.inf, None)
        for x0 in starts:
            x, E = vqe(H, x0)
            if E < best[0]: best = (E, x)
        E, x = best; prev = x; psi = hva_state(x); o = observables_psi(psi, N); og = observables_psi(gs, N)
        col[hi] = dict(kappa=kappa, h=h, params=x, E=E, Eg=Eg, F=abs(np.vdot(gs, psi)) ** 2, zz2=o["zz_next"], zz2_ed=og["zz_next"], zz1=o["zz_nearest"], zz1_ed=og["zz_nearest"], x=o["x_mean"])
        if verbose: print("  h=%.2f dE/E=%.4f zz2 %.3f (ED %.3f)" % (h, (E - Eg) / abs(Eg), o["zz_next"], og["zz_next"]))
    return col

if __name__ == "__main__":
    import pickle, time
    ki0, ki1 = int(sys.argv[1]), int(sys.argv[2]); rng = np.random.default_rng(ki0); out = {}; t0 = time.perf_counter()
    for ki in range(ki0, ki1):
        col = solve_column(ki, rng)
        for hi, c in col.items(): out[(ki, hi)] = c
        print("kappa=%.2f: max dE/E %.4f, max |zz2 err| %.3f (%.0f s)" % (KAPPAS[ki], max((c["E"] - c["Eg"]) / abs(c["Eg"]) for c in col.values()), max(abs(c["zz2"] - c["zz2_ed"]) for c in col.values()), time.perf_counter() - t0), flush=True)
    pickle.dump(out, open(os.path.join(DATA, "hva_%d_%d.pkl") % (ki0, ki1), "wb"))
