"""Required noisy diagrams computed in PennyLane: the optimized HVA circuit at every grid point on default.mixed with
DepolarizingChannel(p) on the CNOT target after every CNOT (the starter kit's convention). Chunked; results cached."""
import os
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
import sys, pickle, time, numpy as np, pennylane as qml
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from hva import _PAIRS, LAYERS, N
grid = {}
for f in ("0_4", "4_8", "8_12", "12_15"): grid.update(pickle.load(open(os.path.join(DATA, "hva_%s.pkl") % f, "rb")))
order = [(ki, hi) for ki in range(15) for hi in range(15)]
dev = qml.device("default.mixed", wires=N)

@qml.qnode(dev)
def zz_nnn(params, p):
    prm = np.asarray(params).reshape(LAYERS, 3)
    for i in range(N): qml.Hadamard(wires=i)
    for l in range(LAYERS):
        g1, g2, b = prm[l]
        for k, (i, j) in enumerate(_PAIRS):
            qml.CNOT(wires=[i, j])
            if p > 0: qml.DepolarizingChannel(p, wires=j)
            qml.RZ(2 * (g1 if k < N else g2), wires=j)
            qml.CNOT(wires=[i, j])
            if p > 0: qml.DepolarizingChannel(p, wires=j)
        for i in range(N): qml.RX(2 * b, wires=i)
    return [qml.expval(qml.PauliZ(i) @ qml.PauliZ((i + 2) % N)) for i in range(N)]

if __name__ == "__main__":
    c0, c1 = int(sys.argv[1]), int(sys.argv[2]); out = {}; t0 = time.perf_counter()
    for c in range(c0, c1):
        ki, hi = order[c]; prm = grid[(ki, hi)]["params"]
        out[(ki, hi)] = {p: float(np.mean(zz_nnn(prm, p))) for p in (0.0, 0.01, 0.05)}
    pickle.dump(out, open(os.path.join(DATA, "pl_depol_%d_%d.pkl") % (c0, c1), "wb"))
    print("points %d-%d done in %.0f s" % (c0, c1, time.perf_counter() - t0))
