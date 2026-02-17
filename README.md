# ionis-cuda

CUDA signature embedding engine for the IONIS project.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![COPR](https://img.shields.io/badge/COPR-ki7mt%2Fionis--ai-blue)](https://copr.fedorainfracloud.org/coprs/ki7mt/ionis-ai/)
[![Platform: EL9](https://img.shields.io/badge/Platform-EL9-green.svg)](https://rockylinux.org/)

## Overview

Generates float4 embeddings from WSPR spot data and solar indices using CUDA kernels on NVIDIA GPUs. The bulk-processor reads from `wspr.bronze` and `solar.bronze` in ClickHouse and writes embeddings to `wspr.silver`.

```
Pipeline:  wspr.bronze + solar.bronze  ──▶  bulk-processor (CUDA)  ──▶  wspr.silver
Output:    4.4B embeddings, 41 GiB
Hardware:  RTX PRO 6000 (96 GB VRAM) — single-pass processing
Wall time: ~45 min on Threadripper 9975WX
```

## Components

| Component | Description |
|-----------|-------------|
| `bulk-processor` | Main CUDA embedding generator — reads ClickHouse, writes silver table |
| `wspr-cuda-check` | Quick GPU capability check utility |
| `src/cuda/` | CUDA kernels for embedding computation |
| `src/engine/` | Processing engine and batch orchestration |
| `src/io/` | ClickHouse I/O with Maidenhead grid conversion |

## Requirements

- NVIDIA GPU with sufficient VRAM (tested on RTX PRO 6000, 96 GB)
- CUDA 12.8+ toolkit
- NVIDIA driver 570+
- CMake 3.28+
- ClickHouse with populated `wspr.bronze` and `solar.bronze`

## Building

```bash
cd build/cmake
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build

# Or use the top-level Makefile
make all
```

## Usage

```bash
# Run the bulk processor (default host: 192.168.1.90)
bulk-processor --host 192.168.1.90

# Environment variable override
CH_HOST=10.60.1.1 CH_PORT=9000 bulk-processor
```

## Installation

### From COPR (Recommended)

```bash
sudo dnf copr enable ki7mt/ionis-ai
sudo dnf install ionis-cuda
```

### Upgrading from ki7mt-ai-lab-cuda

The `ionis-cuda` package includes `Obsoletes: ki7mt-ai-lab-cuda` for seamless upgrade:

```bash
sudo dnf copr enable ki7mt/ionis-ai
sudo dnf upgrade --refresh
```

### From Source

```bash
git clone https://github.com/IONIS-AI/ionis-cuda.git
cd ionis-cuda
make all
sudo make install
```

## Related Repositories

| Repository | Purpose |
|------------|---------|
| [ionis-core](https://github.com/IONIS-AI/ionis-core) | DDL schemas, SQL scripts |
| [ionis-apps](https://github.com/IONIS-AI/ionis-apps) | Go data ingesters (WSPR, solar, contest, RBN) |
| [ionis-training](https://github.com/IONIS-AI/ionis-training) | PyTorch model training |
| [ionis-validate](https://github.com/IONIS-AI/ionis-validate) | Model validation suite (PyPI) |
| [ionis-docs](https://github.com/IONIS-AI/ionis-docs) | Documentation site |

## License

GPL-3.0-or-later — See [COPYING](COPYING)

## Author

Greg Beam, KI7MT

## Links

- **COPR:** https://copr.fedorainfracloud.org/coprs/ki7mt/ionis-ai/
- **Issues:** https://github.com/IONIS-AI/ionis-cuda/issues
