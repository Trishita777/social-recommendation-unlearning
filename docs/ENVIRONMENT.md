# Environment

Observed full-experiment environment on 9 November 2026:

| Component | Version |
| --- | --- |
| Python | 3.12.3 |
| PyTorch | 2.13.0+cu130 |
| NumPy | 2.5.4 |
| SciPy | 1.18.1 |
| scikit-learn | 1.9.1 |

These are recorded observations, not a guarantee that the same wheels are available for every platform. The package declares minimum API requirements, not a cross-platform exact lock. Install an appropriate PyTorch build for your CUDA driver. The source uses POSIX file locks (`fcntl`); Linux is the supported experiment platform.

Use `python -m pip freeze` to record the actual environment alongside reproduced results. Unit tests use small CPU graphs; full-data GPU memory and training time are substantially larger. Concurrent training affects wall-clock measurements, and CUDA nondeterminism may limit bitwise equality across hardware/software versions.
