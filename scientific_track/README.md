# QSITE 2026 Quantum Coalition, Scientific Track submission

**The ANNNI phase diagram under two classes of error: what depolarizing noise and coherent drift each do to a
phase boundary, and what recovers it.** Matthew Stanley Leibel, TrueLoop Compute (NEOTECH Inc.), September 2026.

## Deliverables (as required by the track handout)

| Deliverable | File |
|---|---|
| Implementation notebook (executed, PennyLane) | `ANNNI_phase_diagram_two_error_classes.ipynb` |
| Clean phase diagram, p = 0 | `figures/phase_diagram_p0.png` |
| Noisy phase diagram, p = 0.01 | `figures/phase_diagram_p0.01.png` |
| Noisy phase diagram, p = 0.05 | `figures/phase_diagram_p0.05.png` |
| Write-up (3 pages) | `writeup/writeup.pdf` (source `writeup/writeup.tex`) |
| Presentation (9 slides, ~7 minutes) | `presentation/slides.pdf` with timed `presentation/speaker_script.md` |

Supporting figures: `figures/depolarizing_rescaled.png`, `figures/drift_free_vs_maintained.png`,
`figures/continuous_operation.png`, `figures/decoupling.png`.

## What is in the notebook
1. Hamiltonian from the starter kit, checked against an independent construction and exact diagonalization.
2. Hamiltonian-variational ansatz in PennyLane (128 CNOTs), equal to the fast numpy implementation to machine precision.
3. VQE with adjoint gradients: one point optimized live in PennyLane (relative error 1.7e-5); full grid cached.
4. Clean phase diagram (required image 1), boundaries against the analytic Ising and KT lines, thresholds fixed once.
5. Depolarizing noise on `default.mixed` with the starter kit's convention (required images 2 and 3), a live point
   recomputed and compared with the cached grid, amplitude-loss analysis, reference rescaling, robustness ranking.
6. Coherent calibration drift, its signed first-order effect, and in-band maintenance with TrueLoop v3.1.0.
7. Results table over all arms, including calibrate-once and per-point re-optimization, and the isolation test.
8. Continuous operation: ten passes with a calibration jump at N = 8 (re-run live) and N = 12 (36 channels).
9. Complexity decoupled from the state: the same scan at N = 8, 12, 16, 20 (24 to 60 channels, states of 256 to
   1,048,576 amplitudes) with free-running, engineer-every-50-cycles, and contract arms; eight-layer circuits
   (`code/push_n.py`, `code/continuous_n.py`, cached in `data/decoupling*.pkl`).
10. Interpretation, robustness explanation, floating-phase comment, limitations.

## Reproducing
```
pip install pennylane numpy scipy matplotlib nbformat nbconvert ipykernel
pip install trueloop/trueloop-3.1.0rc90-py3-none-any.whl     # the notebook does this itself if missing
jupyter nbconvert --to notebook --execute ANNNI_phase_diagram_two_error_classes.ipynb
```
Executed as shipped on Python 3.12 with PennyLane 0.45.1 in about 2 minutes (cached grids). The track environment
pins PennyLane 0.44.1; only standard APIs are used. Set `RECOMPUTE = True` in the first cell to regenerate every grid
from scratch: VQE about 6 minutes (`code/hva.py`), PennyLane noisy grids about 12 minutes (`code/pl_depol.py`).

`code/` holds the modules (simulator validated against `default.mixed` to 1e-12, HVA/VQE, noise study, analysis,
continuous operation, N-generic HVA). `data/` holds the cached VQE parameters, PennyLane grids, and every result the
notebook, write-up, and slides quote. `starter_kit/` is the track's kit, used for the Hamiltonian and the analytic lines.

## The maintenance layer
`trueloop/trueloop-3.1.0rc90-py3-none-any.whl` is the 90-day evaluation build of TrueLoop Compute, pure Python with
no dependencies, included so the maintenance results reproduce. Its term starts the first time it is used; nothing in
it phones home. Method and evidence: preprint doi 10.20944/preprints202609.1101.v1 (under review at IEEE Transactions
on Computers). Configuration note from this study: let the preflight learn link signs rather than declaring them when
the identification probe's shot noise is comparable to the drift (Section 9 of the notebook).

## References and resources used
1. Quantum Coalition, QSITE 2026 Scientific Track handout and starter kit: https://github.com/benmcdonough20/QSITE-2026-QuantumCoalition (Hamiltonian builder, observables, analytic lines; `starter_kit/` is a copy).
2. PennyLane demo "ANNNI Phase Detection": https://pennylane.ai/qml/demos/tutorial_annni (source of the Ising and Kosterlitz-Thouless boundary formulas used as reference lines).
3. PennyLane challenge "A Noisy Heisenberg Model": https://pennylane.ai/challenges/heisenberg_model (the depolarizing-after-every-CNOT convention).
4. V. Bergholm et al., "PennyLane: Automatic differentiation of hybrid quantum-classical computations," arXiv:1811.04968 (2018). PennyLane 0.45.1.
5. W. Selke, "The ANNNI model: theoretical analysis and experimental application," Physics Reports 170, 213 (1988).
6. M. S. Leibel, "A Stability Contract for Scalable Hybrid Computing: Minimal Sufficient Control of Component-Observable Hardware," Preprints 202609.1101 (2026), doi:10.20944/preprints202609.1101.v1, under review at IEEE Transactions on Computers. Method and bounds of the maintenance layer; the TrueLoop v3.1.0 evaluation build (NEOTECH Inc.) is included under its evaluation licence.
7. NumPy, SciPy, Matplotlib, nbformat/nbconvert.
