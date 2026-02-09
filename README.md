# ki7mt-ai-lab-cuda

CUDA signature embedding engine for the KI7MT Sovereign AI Lab.

Part of the [IONIS](https://github.com/KI7MT/ki7mt-ai-lab-docs) (Ionospheric Neural Inference System) project.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![COPR](https://img.shields.io/badge/COPR-ki7mt%2Fai--lab-blue)](https://copr.fedorainfracloud.org/coprs/ki7mt/ai-lab/)
[![Platform: EL9](https://img.shields.io/badge/Platform-EL9-green.svg)](https://rockylinux.org/)

## Overview

Generates float4 embeddings from WSPR spot data and solar indices using CUDA kernels on NVIDIA GPUs. The bulk-processor reads from `wspr.bronze` and `solar.bronze` in ClickHouse and writes embeddings to `wspr.silver`.

**Current version:** 2.3.1

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
sudo dnf copr enable ki7mt/ai-lab
sudo dnf install ki7mt-ai-lab-cuda
```

### From Source

```bash
git clone https://github.com/KI7MT/ki7mt-ai-lab-cuda.git
cd ki7mt-ai-lab-cuda
make all
sudo make install
```

## Related Repositories

| Repository | Purpose |
|------------|---------|
| [ki7mt-ai-lab-apps](https://github.com/KI7MT/ki7mt-ai-lab-apps) | Go data ingesters (WSPR, solar, contest, RBN) |
| [ki7mt-ai-lab-core](https://github.com/KI7MT/ki7mt-ai-lab-core) | DDL schemas, SQL scripts |
| [ki7mt-ai-lab-training](https://github.com/KI7MT/ki7mt-ai-lab-training) | PyTorch model training |
| [ki7mt-ai-lab-docs](https://github.com/KI7MT/ki7mt-ai-lab-docs) | Documentation site |

## License

GPL-3.0-or-later — See [COPYING](COPYING)

## Author

Greg Beam, KI7MT

## Links

- **COPR:** https://copr.fedorainfracloud.org/coprs/ki7mt/ai-lab/
- **Issues:** https://github.com/KI7MT/ki7mt-ai-lab-cuda/issues
