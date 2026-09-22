"""A compiled circuit only delivers what the compiler assumed if the device stays where it was characterized.
This experiment runs our routed circuits for many consecutive executions while every hardware edge's two-qubit
calibration and every qubit's single-qubit calibration drifts, and compares four ways of living with that:
  as-compiled        the fidelities the compiler saw (the number the compiler promised)
  free-running       nobody maintains the device
  engineer visit     an engineer recalibrates every K executions (scheduled maintenance, sites x shifts)
  maintained         the stability contract holds every channel in-band, one parallel acquisition per execution
Then the hardware graph is scaled to 10^2 .. 10^5 channels to measure the host cost of the maintained arm
against the memory an O(n^2) model-based reconstruction would need."""
import sys, math, time, random, json, numpy as np
sys.path.insert(0, "/tmp/qsite2/Computational Track"); sys.path.insert(0, "/home/claude/comp")
from starter_kit import benchmarks, hardware, scorer
import solver
from trueloop import StabilityContract
import networkx as nx

SHOTS = 4000; AMP = 3                      # odd rung: biased at the fringe's steepest point; monotone over |e| < 33%, so bounds x drift must stay inside
SIGN = float(np.sign(np.sin(AMP * np.pi / 2)))

def routed_ops(name, G, seed=1):
    prog = benchmarks.BENCHMARKS[name]; pl, routed = solver.solve(prog, G, restarts=60, seed=seed, reverse_passes=3)
    s = scorer.score_summary(prog, G, pl, routed); assert s["valid"]; return routed, s

class Device:
    """Per-edge two-qubit angle error eps_e and per-qubit single-qubit angle error eps_q, drifting: static offset plus
    a mean-reverting wander (lambda = 0.02 per execution), half common-mode and half independent. Gate infidelity from
    a coherent angle error eps is eps^2 / 4 on top of a fixed incoherent floor p0."""
    def __init__(self, G, rms, seed, p0_2q=0.006, p0_1q=0.0005, walk=0.002):
        """Freshly calibrated at compile time (errors zero at t = 0); afterwards a mean-reverting wander with stationary
        RMS `rms` (half common-mode, half independent) plus a slow non-reverting random walk of `walk` per execution."""
        self.G = G; self.edges = list(G.edges); self.nodes = list(G.nodes); self.rng = np.random.default_rng(seed)
        self.n = len(self.edges) + len(self.nodes); self.p0 = np.array([p0_2q] * len(self.edges) + [p0_1q] * len(self.nodes))
        s = rms / np.sqrt(2); self.static = np.zeros(self.n); self.u = np.zeros(self.n); self.walk = walk
        self.w = np.zeros(self.n); self.w_c = 0.0; self.step = s * np.sqrt(2 * 0.02); self.lam = 0.02
        self.edge_index = {tuple(sorted(e)): i for i, e in enumerate(self.edges)}; self.node_index = {q: len(self.edges) + i for i, q in enumerate(self.nodes)}
    def tick(self):
        self.w = (1 - self.lam) * self.w + self.step * self.rng.standard_normal(self.n); self.w_c = (1 - self.lam) * self.w_c + self.step * self.rng.standard_normal()
        self.u += self.walk * self.rng.standard_normal(self.n)
    def eps(self): return self.static + self.w + self.w_c + self.u
    def success(self, routed, correction):
        """Success probability of the routed circuit with effective angle errors eps*(1+c)-... : eps_eff = (1+eps)*a - 1."""
        e = (1 + self.eps()) * correction - 1; p = self.p0 + e ** 2 / 4
        logP = 0.0
        for op in routed:
            if op[0] == "SWAP": logP += 3 * math.log(1 - p[self.edge_index[tuple(sorted(op[1:]))]])      # a SWAP is three CNOTs
            elif op[0] == "2Q": logP += math.log(1 - p[self.edge_index[tuple(sorted(op[1:]))]])
        return math.exp(logP)
    def probe(self, correction, rng):
        e = (1 + self.eps()) * correction - 1
        px = 0.5 * (1 + np.cos(AMP * (np.pi / 2) * (1 + e)))                         # amplified parity / rotation probe per channel
        return rng.binomial(SHOTS, np.clip(px, 0, 1)) / SHOTS

def run(name, rms=0.04, executions=1000, engineer_every=200, seed=3):
    G = hardware.build_hardware_graph(); routed, s = routed_ops(name, G)
    arms = {}
    for arm in ("compiled", "free", "engineer", "maintained"):
        dev = Device(G, rms, seed); rng = np.random.default_rng(seed + 11); n = dev.n
        corr = np.ones(n); base = dev.eps().copy()
        contract = StabilityContract(n, x0=[1.0] * n, target=[0.5] * n, bounds=[[0.8, 1.2]] * n, measurement_noise=0.5 / np.sqrt(SHOTS), tolerance=0.01) if arm == "maintained" else None
        P = []; host = 0.0
        for t in range(executions):
            if arm == "compiled": P.append(dev.success(routed, 1.0 / (1 + dev.eps())))            # the compiler's promise: the characterized floor, no drift
            elif arm == "free": P.append(dev.success(routed, np.ones(n)))
            elif arm == "engineer":
                if t % engineer_every == 0: corr = 1.0 / (1 + dev.eps())                          # a full recalibration at the visit
                P.append(dev.success(routed, corr))
            else:
                P.append(dev.success(routed, corr)); m = dev.probe(corr, rng)
                t0 = time.perf_counter(); corr = np.array(contract.step(m.tolist())["config"]); host += time.perf_counter() - t0
            dev.tick()
        arms[arm] = dict(P=np.array(P), host_us=1e6 * host / executions if arm == "maintained" else 0.0)
    return routed, s, arms

def scaling(rms=0.04, executions=50):
    """Heavy-hex-like graphs with 10^2 .. 10^5 channels: host time per execution for the maintained arm, and the
    memory a dense O(n^2) model-based reconstruction would carry for the same channels."""
    rows = []
    for m in (4, 12, 40, 126):
        G = nx.convert_node_labels_to_integers(nx.hexagonal_lattice_graph(m, m)); dev = Device(G, rms, 1); n = dev.n
        contract = StabilityContract(n, x0=[1.0] * n, target=[0.5] * n, bounds=[[0.8, 1.2]] * n, measurement_noise=0.5 / np.sqrt(SHOTS), tolerance=0.01)
        corr = np.ones(n); rng = np.random.default_rng(2); host = []; res = []
        for t in range(executions):
            m_ = dev.probe(corr, rng); t0 = time.perf_counter(); corr = np.array(contract.step(m_.tolist())["config"]); host.append(time.perf_counter() - t0)
            res.append(np.sqrt(np.mean(((1 + dev.eps()) * corr - 1) ** 2))); dev.tick()
        free = np.sqrt(np.mean(dev.eps() ** 2))
        rows.append(dict(qubits=G.number_of_nodes(), edges=G.number_of_edges(), channels=n, host_us=1e6 * np.median(host[10:]), residual=np.mean(res[-10:]), free=free, recon_bytes=16 * n * n))
        print("  %6d qubits, %6d edges, %7d channels | host %8.0f us/execution | residual %.4f vs free %.4f | dense model %s" % (
            rows[-1]["qubits"], rows[-1]["edges"], n, rows[-1]["host_us"], rows[-1]["residual"], free, ("%.1f GB" % (16 * n * n / 1e9)) if 16 * n * n > 1e9 else ("%.0f MB" % (16 * n * n / 1e6))), flush=True)
    return rows

if __name__ == "__main__":
    out = {}
    for name in ("dense_random", "vqe_layers"):
        routed, s, arms = run(name)
        print("%s: %d SWAPs, depth %d, %d CNOT-equivalents on the routed circuit" % (name, s["swap_count"], s["depth"], sum(3 if op[0] == "SWAP" else 1 for op in routed)))
        for arm, r in arms.items():
            P = r["P"]; print("   %-11s success: start %.3f | mean %.3f | min %.3f | last-100 mean %.3f%s" % (arm, P[0], P.mean(), P.min(), P[-100:].mean(), ("   host %.0f us/execution" % r["host_us"]) if arm == "maintained" else ""))
        out[name] = {a: r["P"].tolist() for a, r in arms.items()}
    print("\nscaling the hardware graph (maintained arm):"); out["scaling"] = scaling()
    json.dump(out, open("/home/claude/comp/runtime_results.json", "w"))
