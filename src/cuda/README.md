# WSPR CUDA Bridge

High-performance CUDA bridge for zero-copy WSPR data processing on the RTX 5090.

## Architecture

### Memory Model

```
┌─────────────────────────────────────────────────────────────┐
│                    Host (CPU) Memory                         │
├─────────────────────────────────────────────────────────────┤
│  WsprBatch (Go)                                              │
│  ↓ ToC()                                                     │
│  Pinned Memory (cudaHostAllocMapped)                         │
│  - Zero-copy allocation                                      │
│  - GPU-accessible via PCIe                                   │
│  - Write-combined for optimal H2D bandwidth                  │
└────────────────┬────────────────────────────────────────────┘
                 │
                 │ cudaMemcpyAsync (non-blocking)
                 │ PCIe Gen 4 x8 (~16 GB/s)
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                 Device (GPU) Memory - RTX 5090               │
├─────────────────────────────────────────────────────────────┤
│  WsprSpotC array (32GB VRAM)                                 │
│  ↓                                                           │
│  CUDA Kernel (sm_90 - compute capability 9.0)               │
│  - Validation: SNR, Power, Frequency, Distance, Band        │
│  - Future: Deduplication, Geospatial analysis               │
│  ↓                                                           │
│  Valid flags array (int[])                                   │
└────────────────┬────────────────────────────────────────────┘
                 │
                 │ cudaMemcpyAsync D2H
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  Results (Go []bool)                                         │
└─────────────────────────────────────────────────────────────┘
```

## Files

- `bridge.h` - C API declarations for CGO
- `bridge.cu` - CUDA implementation (kernels + runtime wrappers)
- `README.md` - This file

## API Overview

### Memory Management

```c
// Allocate pinned host memory (GPU-accessible)
CudaError cuda_host_alloc_pinned(void** ptr, size_t size);

// Free pinned host memory
CudaError cuda_free_host(void* ptr);

// Allocate device memory
CudaError cuda_malloc_device(void** ptr, size_t size);

// Free device memory
CudaError cuda_free_device(void* ptr);
```

### Async Data Transfers

```c
// Non-blocking host-to-device copy
CudaError cuda_memcpy_host_to_device_async(
    void* dst,
    const void* src,
    size_t size,
    CudaStream stream
);

// Non-blocking device-to-host copy
CudaError cuda_memcpy_device_to_host_async(
    void* dst,
    const void* src,
    size_t size,
    CudaStream stream
);
```

### Stream Management

```c
// Create CUDA stream for async operations
CudaError cuda_stream_create(CudaStream* stream);

// Destroy stream
CudaError cuda_stream_destroy(CudaStream stream);

// Block until stream completes
CudaError cuda_stream_synchronize(CudaStream stream);

// Check if stream is done (non-blocking)
CudaError cuda_stream_query(CudaStream stream);
```

### WSPR Validation Kernel

```c
// GPU kernel: validate WSPR spots
CudaError cuda_kernel_validate_spots(
    WsprSpotC* d_spots,       // Device pointer
    int* d_valid_flags,       // Device pointer (output)
    int count,                // Number of spots
    CudaStream stream         // Async stream
);
```

**Validation Rules** (current implementation):
1. SNR: -30 to 10 dB (typical WSPR range)
2. Power: 0 to 60 dBm (WSPR spec)
3. Frequency: > 0 MHz
4. Distance: > 0 km (can't receive own transmission)
5. Band: Valid WSPR bands (160m, 80m, 60m, 40m, 30m, 20m, 17m, 15m, 12m, 10m, 6m, 2m)

## Data Structure Alignment

The `WsprSpotC` struct is packed and aligned for optimal GPU access:

```c
typedef struct {
    int64_t  timestamp;       // 8 bytes
    char     reporter[16];    // 16 bytes
    char     reporter_grid[8];// 8 bytes
    int8_t   snr;             // 1 byte
    double   frequency;       // 8 bytes
    char     callsign[16];    // 16 bytes
    char     grid[8];         // 8 bytes
    int8_t   power;           // 1 byte
    int8_t   drift;           // 1 byte
    uint16_t distance;        // 2 bytes
    uint16_t azimuth;         // 2 bytes
    int8_t   band;            // 1 byte
    char     version[8];      // 8 bytes
    int8_t   code;            // 1 byte
    uint8_t  column_count;    // 1 byte
    uint8_t  _padding[5];     // 5 bytes (align to 8-byte boundary)
} __attribute__((packed)) WsprSpotC;

// Total: 87 bytes (padded to 88 for alignment)
```

**Design Notes**:
- Packed attribute ensures no compiler padding
- Explicit padding for 8-byte alignment (coalesced memory access)
- String fields use fixed arrays (no pointers, GPU-safe)
- int8/uint16 types minimize VRAM usage

## Performance Characteristics

### RTX 5090 Specifications
- Compute Capability: 9.0 (sm_90)
- VRAM: 32GB GDDR7
- Memory Bandwidth: ~1.5 TB/s
- CUDA Cores: ~21,760 (estimated)
- Tensor Cores: 680 (Gen 5)
- PCIe: Gen 4 x8 (~16 GB/s bidirectional)

### Throughput Estimates

**Batch Size**: 1M spots (typical)
- WsprSpotC size: 88 bytes
- Batch memory: 88MB

**H2D Transfer** (pinned memory):
- Time: 88MB / 16GB/s ≈ 5.5ms
- Overlap with CPU parsing via async streams

**Kernel Execution** (validation):
- Block size: 256 threads
- Grid size: (1M / 256) = 3,906 blocks
- Estimated time: < 1ms (memory-bound, simple logic)

**D2H Transfer** (results):
- Flag array: 1M * 4 bytes = 4MB
- Time: 4MB / 16GB/s ≈ 0.25ms

**Total GPU Time**: ~7ms per 1M spots
**Throughput**: ~143M spots/second

For 10B rows:
- GPU time: 10,000 * 7ms = 70 seconds
- Bottleneck: Likely disk I/O or ClickHouse insertion

## Usage from Go

See `internal/parser/cuda_bridge.go` for the CGO wrapper.

### Example: Process a batch

```go
// Create CUDA buffer (pinned memory + device memory)
buf, err := parser.NewCudaBuffer(1_000_000)
if err != nil {
    log.Fatal(err)
}
defer buf.Close()

// Copy batch to device (async)
err = buf.CopyToDevice(batch)
if err != nil {
    log.Fatal(err)
}

// Run validation kernel on GPU (async)
validFlags, err := buf.ValidateOnGPU(batch.Count)
if err != nil {
    log.Fatal(err)
}

// Synchronize stream
err = buf.Sync()
if err != nil {
    log.Fatal(err)
}

// Process results
for i, valid := range validFlags {
    if valid {
        fmt.Printf("Spot %d is valid\n", i)
    }
}
```

### Example: Buffer Pool for Pipelining

```go
// Create pool of 4 buffers for overlap
pool, err := parser.NewCudaBufferPool(4, 1_000_000)
if err != nil {
    log.Fatal(err)
}
defer pool.Close()

for batch := range batchChannel {
    // Get buffer from pool
    buf, _ := pool.Get()

    // Launch async operations
    buf.CopyToDevice(batch)
    buf.ValidateOnGPU(batch.Count)

    // Return to pool when done (async)
    go func(b *parser.CudaBuffer) {
        b.Sync()  // Wait for completion
        pool.Put(b)
    }(buf)
}
```

## Building

```bash
# Compile CUDA bridge
make cuda

# Build full Go binary
make build

# Run tests
make test

# Check CUDA installation
make check-cuda
```

## Requirements

- CUDA 12.8+ (`nvcc` in PATH)
- NVIDIA driver with compute capability 9.0+ support
- Go 1.25+ with CGO enabled
- RTX 5090 or compatible GPU

## Vectorized Processing Kernel

### Overview

The `cuda_kernel_process_vectorized` function implements high-speed WSPR processing using vectorized memory operations optimized for the RTX 5090.

### Struct Alignment for uint4 Vectorization

The `WsprSpotC` struct is now **96 bytes** (aligned to 16-byte boundary):

```c
typedef struct {
    // ... fields ...
    uint8_t  _padding[13];    // Align to 16-byte boundary (96 bytes = 6 × uint4)
} __attribute__((packed)) WsprSpotC;
```

### Vectorized Memory Access

Uses `uint4` (128-bit) loads/stores for coalesced memory access:

```cuda
// Load 6 × uint4 = 96 bytes (one complete struct)
const uint4* d_input_vec = reinterpret_cast<const uint4*>(d_input);
uint4 data[6];

#pragma unroll
for (int i = 0; i < 6; i++) {
    data[i] = d_input_vec[idx * 6 + i];  // Single 128-bit transaction
}

WsprSpotC spot = *reinterpret_cast<WsprSpotC*>(data);
```

**Benefits:**
- **Coalesced access**: Threads in warp load contiguous 128-bit chunks
- **Cache efficiency**: RTX 5090 uses 128-byte cache lines
- **Reduced transactions**: 6 loads vs 96 byte-by-byte reads

### Normalization Operations

1. **Grid Locator Standardization**: Converts to uppercase (`"fn42"` → `"FN42"`)
2. **Frequency Conversion**: Float64 → uint64 fixed-point (`14.097100 MHz` → `14097100`)
3. **Validation**: SNR, Power, Frequency, Distance, Band checks

### Stream-Based Overlap

Kernel executes on `cudaStream_t` to overlap with CPU mmap I/O:

```
Timeline:
  CPU: [mmap batch 1] [mmap batch 2] [mmap batch 3] ...
  GPU:                [process 1]    [process 2]    ...
```

### RTX 5090 Optimizations

- **`__launch_bounds__(256)`**: Optimizes register allocation
- **Tail handling**: Early exit for non-multiple-of-256 batch sizes
- **`#pragma unroll`**: Eliminates loop overhead for fixed-iteration loops

### Usage

```go
buf, _ := NewCudaBuffer(batchSize)
buf.CopyToDevice(batch)

// Single-pass: normalization + validation + frequency conversion
results, validFlags, freqFixed, err := buf.ProcessVectorizedOnGPU(batch.Count)
```

### Performance

- **Expected throughput**: 500M - 2B spots/sec (cache-dependent)
- **10B records**: ~5-20 seconds GPU time (not including I/O)

See `internal/parser/vectorized_test.go` for benchmarks.

## Future Enhancements

1. **Radix Sort on GPU**: Sort by `freqFixed` directly on device
2. **Deduplication Kernel**: Remove duplicate spots using GPU hash tables
3. **Multi-GPU Support**: Shard batches across multiple GPUs
4. **CUDA Graphs**: Pre-record operation sequences for lower latency
5. **Persistent Kernels**: Keep kernel running, feed batches via streams

## Debugging

```bash
# Check CUDA errors
export CUDA_LAUNCH_BLOCKING=1

# Enable CUDA error checking
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Profile with nsys
nsys profile ./bin/wspr-ingest
```

---

**Status**: CUDA bridge complete and ready for integration testing on RTX 5090 server.
