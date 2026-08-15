# docs/archive/

Superseded documents, kept for provenance rather than reference.

These were written against the pre-audit version of the project (Oct 2025 – Apr 2026).
They contain accuracy figures — Docker 94%, venvy 83% — that were measured on
training data and **should not be cited**. See `docs/EVAL_METHODOLOGY.md` for what
went wrong and `notes/PROGRESS.md` for the corrected direction.

| File | Was | Status |
|------|-----|--------|
| `TECHNICAL_ANALYSIS.md` | Initial feasibility study for the venvy proof-of-concept | Historical. Model recommendations (TinyLlama, Phi-2) are two generations stale. |
| `IMPLEMENTATION_PLAN.md` | Original build plan | Superseded by `notes/PROGRESS.md` |
| `TRAINING_GUIDE.md` | Gemma 3 1B training walkthrough | Superseded by the Colab notebook |
| `TRAINING_GUIDE_COLAB.md` | Colab variant of the above | Superseded by the Colab notebook |
| `COLAB_DOCKER_TRAINING_GUIDE.md` | Docker-specific Colab walkthrough | Superseded by the Colab notebook |
| `QUICK_START_DOCKER.md` | Docker quick start | Folded into README |
| `IMATRIX_FIX.md` | llama.cpp imatrix build failure + fix | Still technically useful; llama.cpp build steps drift fast |
| `IMATRIX_CMAKE_FIX.md` | CMake-era rewrite of the above | Still technically useful |

The two `IMATRIX_*` files document real debugging of the llama.cpp importance-matrix
build and are the most reusable thing in here.
