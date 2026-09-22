# ionis-cuda

CUDA signature embedding engine for the IONIS project.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![COPR](https://img.shields.io/badge/COPR-ki7mt%2Fionis--ai-blue)](https://copr.fedorainfracloud.org/coprs/ki7mt/ionis-ai/)
[![Platform: EL9](https://img.shields.io/badge/Platform-EL9-green.svg)](https://rockylinux.org/)

## Overview

Generates float4 embeddings from WSPR spot data and solar indices using CUDA kernels on NVIDIA GPUs.

```
Pipeline:  wspr.bronze + solar.bronze  ──▶  bulk-processor (CUDA)  ──▶  (destination table)
Hardware:  RTX PRO 6000 (96 GB VRAM) — single-pass processing
Wall time: ~45 min on Threadripper 9975WX
```

> ### Not currently wired into the IONIS pipeline
>
> `bulk-processor` wrote to `wspr.silver`, **which was dropped on 2026-09-22** holding zero
> rows. It was probably not always empty — a QA rebuild recorded 4.43B rows on 2026-02-07 —
> but ClickHouse's logs only retain back to 2026-09-06, so when it emptied cannot be
> established. **That a table could lose four billion rows unnoticed for seven months is the
> finding.** It could, because nothing read it: this tool is not packaged in any RPM, has no
> systemd unit, and runs only by hand, while all fourteen gold populate scripts read
> `wspr.bronze` directly. The medallion chain the docs described — `bronze → silver → gold` —
> was a design, not the build. The build is `bronze → gold`.
>
> **The CUDA engine itself is sound and is kept.** What it lacks is a consumer. Before
> running it again, decide where its output goes and what reads it; `sql/01-model_features.sql`
> still carries the original schema for reference, but creates nothing.
>
> Current lineage for every table in the lab:
> [`ionis-core/docs/DATA-DICTIONARY.md`](https://github.com/IONIS-AI/ionis-core/blob/main/docs/DATA-DICTIONARY.md)

## Components

| Component | Description |
|-----------|-------------|
| `bulk-processor` | Main CUDA embedding generator — reads ClickHouse, writes float4 embeddings. **No destination table at present — see above.** |
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
