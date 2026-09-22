"""solve(program, hardware_graph) -> (initial_placement, routed_program) for the QSITE 2026 Computational Track.

Rules of the game (from starter_kit/scorer.py): the routed program's 2Q gates must reproduce the original list exactly,
in order and orientation, so gates cannot be reordered; SWAPs may be inserted anywhere on hardware edges; depth is the
ASAP critical path over 2Q and SWAP ops; score = swaps + 0.5 * depth. Therefore everything is won in three places:
the initial placement, which qubit moves along which path for each gate, and SWAPs that do not extend the critical path.

Method: interaction-weighted placement by simulated annealing, sequential routing with a SABRE-style lookahead cost
that also charges each candidate SWAP by how much it lengthens the critical path, SABRE reverse passes to refine the
initial placement, and random restarts with the exact scorer in the loop."""
from __future__ import annotations
import math, random
import networkx as nx

# ----------------------------------------------------------------------------- distances
def all_pairs(graph):
    return dict(nx.all_pairs_shortest_path_length(graph))

# ----------------------------------------------------------------------------- placement
def interaction_weights(program, decay=0.985):
    """Weight of each logical pair: earlier gates count more (they are routed first), all gates count."""
    w = {}
    for k, op in enumerate(program):
        if op[0] != "2Q": continue
        a, b = sorted(op[1:]); w[(a, b)] = w.get((a, b), 0.0) + decay ** k
    return w

def placement_cost(placement, weights, D):
    return sum(wt * D[placement[a]][placement[b]] for (a, b), wt in weights.items())

def anneal_placement(logicals, graph, weights, D, rng, steps=4000, t0=2.0, t1=0.02):
    nodes = list(graph.nodes); physical = rng.sample(nodes, len(logicals))
    placement = dict(zip(logicals, physical)); free = [n for n in nodes if n not in physical]
    cost = placement_cost(placement, weights, D); best = (cost, dict(placement))
    for s in range(steps):
        T = t0 * (t1 / t0) ** (s / steps)
        a = rng.choice(logicals)
        if free and rng.random() < 0.4:                       # move a to a free physical qubit
            f = rng.choice(free); old = placement[a]; placement[a] = f
            new = placement_cost(placement, weights, D)
            if new <= cost or rng.random() < math.exp((cost - new) / T):
                cost = new; free.remove(f); free.append(old)
            else: placement[a] = old
        else:                                                   # swap the physical qubits of two logicals
            b = rng.choice(logicals)
            if a == b: continue
            placement[a], placement[b] = placement[b], placement[a]
            new = placement_cost(placement, weights, D)
            if new <= cost or rng.random() < math.exp((cost - new) / T): cost = new
            else: placement[a], placement[b] = placement[b], placement[a]
        if cost < best[0]: best = (cost, dict(placement))
    return best[1]

# ----------------------------------------------------------------------------- routing
def route(program, graph, placement, D, lookahead=8, decay=0.7, depth_weight=0.35):
    """Sequential routing. For each gate, while its qubits are not adjacent, pick the SWAP (incident to either qubit)
    minimizing: distance of the current gate + decayed distances of the next `lookahead` gates + a charge for how much
    the SWAP would extend the critical path (ASAP layer index it would land on)."""
    place = dict(placement); phys2log = {p: l for l, p in place.items()}
    routed = []; last = {}                                       # last[physical] = ASAP layer index of its last op
    gates = [op for op in program if op[0] == "2Q"]
    def layer_of(p, q): return 1 + max(last.get(p, 0), last.get(q, 0))
    def commit(kind, p, q):
        L = layer_of(p, q); last[p] = L; last[q] = L; routed.append((kind, p, q))
    for k, (_, a, b) in enumerate(gates):
        guard = 0
        while D[place[a]][place[b]] > 1:
            guard += 1
            if guard > 200: raise RuntimeError("routing stalled")
            pa, pb = place[a], place[b]; best = None; cur = D[pa][pb]
            future = gates[k + 1: k + 1 + lookahead]
            for anchor in (pa, pb):
                for nb in graph.neighbors(anchor):
                    # tentative swap of physical anchor<->nb
                    la, lb = phys2log.get(anchor), phys2log.get(nb)
                    def dist_after(x, y):
                        px, py = place[x], place[y]
                        m = {anchor: nb, nb: anchor}
                        return D[m.get(px, px)][m.get(py, py)]
                    cost = dist_after(a, b)
                    if cost != cur - 1: continue                   # every SWAP must bring the current gate one step closer
                    for j, (_, x, y) in enumerate(future):
                        cost += (decay ** (j + 1)) * dist_after(x, y)
                    cost += depth_weight * layer_of(anchor, nb)
                    if best is None or cost < best[0] - 1e-12: best = (cost, anchor, nb)
            _, p, q = best
            la, lb = phys2log.get(p), phys2log.get(q)
            if la is not None: place[la] = q
            if lb is not None: place[lb] = p
            phys2log[p], phys2log[q] = lb, la                 # after the SWAP, p holds lb and q holds la
            if lb is None: del phys2log[p]
            if la is None: del phys2log[q]
            commit("SWAP", p, q)
        commit("2Q", place[a], place[b])
    return routed, place

def score(routed):
    swaps = sum(1 for op in routed if op[0] == "SWAP"); last = {}; depth = 0
    for op in routed:
        p, q = op[1], op[2]; L = 1 + max(last.get(p, 0), last.get(q, 0)); last[p] = L; last[q] = L; depth = max(depth, L)
    return swaps + 0.5 * depth, swaps, depth

# ----------------------------------------------------------------------------- the entry point
def solve(program, hardware_graph, restarts=120, seed=None, reverse_passes=3, time_budget_s=None):
    """Entry point required by the track. Deterministic: with seed=None the best of seeds 0..2 is returned."""
    if seed is None:
        cands = [solve(program, hardware_graph, restarts, s, reverse_passes, time_budget_s) for s in range(3)]
        return min(cands, key=lambda pr: score(pr[1])[0])
    import time
    t_start = time.time()
    rng = random.Random(seed); D = all_pairs(hardware_graph)
    logicals = sorted({q for op in program for q in op[1:]})
    weights = interaction_weights(program); reversed_program = list(reversed(program))
    best = None
    for r in range(restarts):
        if time_budget_s and time.time() - t_start > time_budget_s: break
        placement = anneal_placement(logicals, hardware_graph, weights, D, rng, steps=3000 if r else 6000)
        # SABRE reverse passes: route forward, route the reversed program from the final layout, use its final layout as the start
        for _ in range(reverse_passes):
            _, final_layout = route(program, hardware_graph, placement, D)
            _, placement = route(reversed_program, hardware_graph, final_layout, D)
        for lookahead, decay, dw in ((8, 0.7, 0.35), (12, 0.8, 0.25), (5, 0.6, 0.5), (16, 0.85, 0.15)):
            routed, _ = route(program, hardware_graph, placement, D, lookahead=lookahead, decay=decay, depth_weight=dw)
            s, sw, dp = score(routed)
            if best is None or s < best[0]: best = (s, dict(placement), routed, sw, dp)
    return best[1], best[2]

if __name__ == "__main__":
    import sys, time; sys.path.insert(0, "..")  # run from a checkout of the track repository, or adjust to your starter_kit path
    from starter_kit import benchmarks, hardware, scorer, baseline_routing
    G = hardware.build_hardware_graph(); total = 0.0; base_total = 0.0
    print("%-16s | %8s | %6s | %5s | %6s | %s" % ("benchmark", "baseline", "ours", "swaps", "depth", "valid"))
    for name, prog in benchmarks.BENCHMARKS.items():
        t = time.time(); pl, routed = solve(prog, G, restarts=int(sys.argv[1]) if len(sys.argv) > 1 else 60)
        s = scorer.score_summary(prog, G, pl, routed); bpl, br = baseline_routing.solve(prog, G); bs = scorer.score_summary(prog, G, bpl, br)
        total += s["score"]; base_total += bs["score"]
        print("%-16s | %8.1f | %6.1f | %5d | %6d | %s  (%.1f s)" % (name, bs["score"], s["score"], s["swap_count"], s["depth"], s["valid"], time.time() - t))
    print("TOTAL baseline %.1f -> ours %.1f (%.0f%% lower)" % (base_total, total, 100 * (1 - total / base_total)))
