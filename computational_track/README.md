# QSITE 2026 Quantum Coalition, Computational Track submission
Matthew Stanley Leibel, TrueLoop Compute (NEOTECH Inc.), September 2026

| Deliverable | File |
|---|---|
| `solve(program, hardware_graph)` | `code/solution.py` (deterministic; best of 120 restarts over three seeds, three SABRE reverse passes) |
| Write-up (2 pages) | `writeup/writeup.pdf` (source `writeup/writeup.tex`) |
| Demo (4 slides, ~4 minutes) | `presentation/slides.pdf`, timed script in `presentation/speaker_script.md` |
| Run-time validity experiment | `code/runtime_validity.py`, results `code/runtime_results.json`, figure `figures/runtime_validity.png` |
| Final scores | `code/final_scores.json`: total 78.5 against the baseline 283.5, all valid |

## Running the solver
Place `code/solution.py` next to the track's `starter_kit/` (or set the `sys.path` line at the bottom of the file), then

    from solution import solve
    placement, routed = solve(program, hardware_graph)

`python code/solution.py` scores all six benchmarks against the provided baseline (about 30 s).

## Running the run-time experiment
Requires the maintenance layer's evaluation build shipped here: `pip install trueloop-3.1.0rc90-py3-none-any.whl`
(pure Python, no dependencies; 90-day evaluation from first use). Then `python code/runtime_validity.py` (about 4 minutes)
reproduces the four arms on dense_random and vqe_layers and the channel-scaling table.
Every number in the write-up and slides comes from `runtime_results.json` and `final_scores.json`.

## References and resources used
1. Quantum Coalition, QSITE 2026 Computational Track handout and starter kit: https://github.com/benmcdonough20/QSITE-2026-QuantumCoalition (hardware graph, benchmarks, scorer, baseline router).
2. G. Li, Y. Ding, Y. Xie, "Tackling the qubit mapping problem for NISQ-era quantum devices," ASPLOS 2019, arXiv:1809.02573 (the SABRE lookahead and reverse-pass heuristics adapted here).
3. NetworkX, A. Hagberg, D. Schult, P. Swart (2008).
4. M. S. Leibel, "A Stability Contract for Scalable Hybrid Computing: Minimal Sufficient Control of Component-Observable Hardware," Preprints 202609.1101 (2026), doi:10.20944/preprints202609.1101.v1, under review at IEEE Transactions on Computers. Method and bounds of the maintenance layer; the TrueLoop v3.1.0 evaluation build (NEOTECH Inc.) is included under its evaluation licence.
5. NumPy, Matplotlib.
