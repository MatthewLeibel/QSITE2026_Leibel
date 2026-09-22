"""HVA VQE at general N with a matrix-free Hamiltonian: H psi = diag * psi - h * sum_i psi[flip_i]."""
import os
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
import sys, numpy as np, scipy.sparse.linalg as sla
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class ANNNI:
    def __init__(self, n, layers=4):
        self.n, self.L = n, layers; dim = 2 ** n; self.dim = dim; idx = np.arange(dim)
        self.zb = 1 - 2 * ((idx[:, None] >> (n - 1 - np.arange(n))) & 1)
        self.zz1 = sum(self.zb[:, i] * self.zb[:, (i + 1) % n] for i in range(n)).astype(float)
        self.zz2 = sum(self.zb[:, i] * self.zb[:, (i + 2) % n] for i in range(n)).astype(float)
        self.flips = [idx ^ (1 << (n - 1 - i)) for i in range(n)]
        self.pairs = [(i, (i + 1) % n) for i in range(n)] + [(i, (i + 2) % n) for i in range(n)]
    def apply_h(self, psi, kappa, h):
        out = (-self.zz1 + kappa * self.zz2) * psi
        for f in self.flips: out -= h * psi[f]
        return out
    def ground(self, kappa, h):
        op = sla.LinearOperator((self.dim, self.dim), matvec=lambda v: self.apply_h(v, kappa, h), dtype=float)
        w, v = sla.eigsh(op, k=1, which="SA", tol=1e-10); return float(w[0]), v[:, 0].astype(complex)
    def rx_all(self, psi, angles):
        n, dim = self.n, self.dim
        for i in range(n):
            c, s = np.cos(angles[i] / 2), np.sin(angles[i] / 2); U = np.array([[c, -1j * s], [-1j * s, c]])
            psi = np.einsum("ab,xby->xay", U, psi.reshape(2 ** i, 2, 2 ** (n - 1 - i))).reshape(dim)
        return psi
    def x_all(self, psi, i):
        n, dim = self.n, self.dim; return psi[self.flips[i]]
    def state(self, params, eps=None, delta=None):
        n = self.n; p = np.asarray(params).reshape(self.L, 3); eps = np.zeros(n) if eps is None else np.asarray(eps)
        psi = np.full(self.dim, 1 / np.sqrt(self.dim), complex)
        for l in range(self.L):
            g1, g2, b = p[l]
            if delta is None or not np.any(delta): psi = psi * np.exp(-1j * (g1 * self.zz1 + g2 * self.zz2))
            else:
                ph = np.zeros(self.dim)
                for k, (i, j) in enumerate(self.pairs): ph += (g1 if k < n else g2) * (1 + delta[k]) * self.zb[:, i] * self.zb[:, j]
                psi = psi * np.exp(-1j * ph)
            psi = self.rx_all(psi, 2 * b * (1 + eps))
        return psi
    def energy_and_grad(self, params, kappa, h):
        n = self.n; p = np.asarray(params).reshape(self.L, 3); psi = np.full(self.dim, 1 / np.sqrt(self.dim), complex)
        for l in range(self.L):
            g1, g2, b = p[l]; psi = psi * np.exp(-1j * (g1 * self.zz1 + g2 * self.zz2)); psi = self.rx_all(psi, 2 * b * np.ones(n))
        hpsi = self.apply_h(psi, kappa, h); E = float(np.real(np.vdot(psi, hpsi))); lam = hpsi; phi = psi; g = np.zeros((self.L, 3))
        for l in reversed(range(self.L)):
            g1, g2, b = p[l]; sx = np.zeros(self.dim, complex)
            for i in range(n): sx += self.x_all(phi, i)
            g[l, 2] = 2 * np.imag(np.vdot(lam, sx)); phi = self.rx_all(phi, -2 * b * np.ones(n)); lam = self.rx_all(lam, -2 * b * np.ones(n))
            g[l, 0] = 2 * np.imag(np.vdot(lam, self.zz1 * phi)); g[l, 1] = 2 * np.imag(np.vdot(lam, self.zz2 * phi))
            ph = np.exp(1j * (g1 * self.zz1 + g2 * self.zz2)); phi = phi * ph; lam = lam * ph
        return E, g.reshape(-1)
    def vqe(self, kappa, h, x0, maxiter=400):
        from scipy.optimize import minimize
        r = minimize(lambda x: self.energy_and_grad(x, kappa, h), x0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-13, "gtol": 1e-9})
        return r.x, float(r.fun), r.nfev
    def zz2_of(self, psi):
        d = np.abs(psi) ** 2; return float(np.sum(d * self.zz2) / self.n)

def solve_column(m, kappa, hs, rng, warm=None):
    prev = None; col = {}
    for hi in range(len(hs) - 1, -1, -1):
        h = hs[hi]; starts = [prev] if prev is not None else []
        if warm is not None and hi in warm: starts.append(np.asarray(warm[hi]))          # the N=8 solution at the same (kappa, h)
        starts += [rng.normal(0, 0.2, 3 * m.L) for _ in range(1)]
        best = (np.inf, None, 0)
        for x0 in starts:
            x, E, nf = m.vqe(kappa, h, x0)
            if E < best[0]: best = (E, x, nf)
        E, x, nf = best; prev = x; Eg, gs = m.ground(kappa, h)
        col[hi] = dict(kappa=kappa, h=h, params=x, E=E, Eg=Eg, zz2=m.zz2_of(m.state(x)), zz2_ed=m.zz2_of(gs), nfev=nf)
    return col

if __name__ == "__main__":
    import pickle, time
    n = int(sys.argv[1]); cols = [float(v) for v in sys.argv[2].split(",")]; hs = np.linspace(0.05, 2.0, 15)
    m = ANNNI(n); rng = np.random.default_rng(n); out = {}; t0 = time.perf_counter()
    g8 = {}
    for f in ("0_4", "4_8", "8_12", "12_15"): g8.update(pickle.load(open(os.path.join(DATA, "hva_%s.pkl") % f, "rb")))
    K8 = np.linspace(0, 1, 15)
    for kappa in cols:
        ki = int(np.argmin(np.abs(K8 - kappa))); warm = {hi: g8[(ki, hi)]["params"] for hi in range(15)} if abs(K8[ki] - kappa) < 1e-6 else None
        col = solve_column(m, kappa, hs, rng, warm=warm); out[kappa] = col
        pickle.dump(col, open(os.path.join(DATA, "grid_N%d_k%.3f.pkl") % (n, kappa), "wb"))
        print("N=%d kappa=%.2f: max dE/E %.4f, max |zz2 err| %.3f, mean nfev %.0f (%.0f s)" % (n, kappa, max((c["E"] - c["Eg"]) / abs(c["Eg"]) for c in col.values()),
              max(abs(c["zz2"] - c["zz2_ed"]) for c in col.values()), np.mean([c["nfev"] for c in col.values()]), time.perf_counter() - t0), flush=True)
    print("saved per column")
