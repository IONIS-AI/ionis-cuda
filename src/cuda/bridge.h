#ifndef WSPR_CUDA_BRIDGE_H
#define WSPR_CUDA_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

// ============================================================================
// CUDA Error Codes
// ============================================================================

typedef enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_MEMORY_ALLOCATION = 2,
    CUDA_ERROR_INVALID_VALUE = 11,
    CUDA_ERROR_NOT_READY = 34
} CudaError;

// ============================================================================
// CUDA Stream (opaque pointer)
// ============================================================================

typedef void* CudaStream;

// ============================================================================
// WsprSpotC Structure (128 bytes, 16-byte aligned)
// ============================================================================
//
// CRITICAL: This struct layout MUST match internal/parser/cuda_bridge.go
// Total size: 128 bytes (optimized for RTX 5090 uint4 vectorization)
// Alignment: 16 bytes (8 x uint4 = 8 x 16 bytes)
//
// SCHEMA SUPPORT: 2013-era archives (SpotID present) and 2014+ (SpotID = 0)
//
// MEMORY LAYOUT:
//   Offset   0: spot_id        (uint64_t, 8 bytes)  <- WSPRnet database ID (2013 format)
//   Offset   8: timestamp      (int64_t,  8 bytes)
//   Offset  16: reporter       (char[16], 16 bytes)
//   Offset  32: reporter_grid  (char[8],  8 bytes)
//   Offset  40: snr            (int8_t,   1 byte)
//   Offset  41: _padding1      (char[7],  7 bytes)  <- Align frequency to 8-byte boundary
//   Offset  48: frequency      (double,   8 bytes)
//   Offset  56: callsign       (char[16], 16 bytes)
//   Offset  72: grid           (char[8],  8 bytes)
//   Offset  80: power          (int8_t,   1 byte)
//   Offset  81: drift          (int8_t,   1 byte)
//   Offset  82: distance       (uint16_t, 2 bytes)
//   Offset  84: azimuth        (uint16_t, 2 bytes)
//   Offset  86: _padding1b     (char[2],  2 bytes)  <- Align band to 4-byte boundary
//   Offset  88: band           (int32_t,  4 bytes)
//   Offset  92: _padding2      (char[4],  4 bytes)  <- Align mode to 8-byte boundary
//   Offset  96: mode           (char[8],  8 bytes)
//   Offset 104: version        (char[8],  8 bytes)
//   Offset 112: code           (int8_t,   1 byte)
//   Offset 113: column_count   (uint8_t,  1 byte)
//   Offset 114: _padding3      (char[14], 14 bytes) <- Align struct to 128 bytes (16-byte aligned)
//   Total: 128 bytes
//
// ============================================================================

typedef struct __attribute__((aligned(16))) WsprSpotC {
    uint64_t spot_id;           // Offset 0:   WSPRnet database ID (2013 format), 8 bytes
    int64_t  timestamp;         // Offset 8:   Unix timestamp (nanoseconds), 8 bytes
    char     reporter[16];      // Offset 16:  Receiving callsign, 16 bytes
    char     reporter_grid[8];  // Offset 32:  Receiver grid, 8 bytes
    int8_t   snr;               // Offset 40:  Signal-to-noise ratio, 1 byte
    char     _padding1[7];      // Offset 41:  Padding to align Frequency to 8-byte boundary
    double   frequency;         // Offset 48:  Frequency in MHz, 8 bytes (8-byte aligned for Blackwell)
    char     callsign[16];      // Offset 56:  Transmitting callsign, 16 bytes
    char     grid[8];           // Offset 72:  Transmitter grid, 8 bytes
    int8_t   power;             // Offset 80:  Transmit power, 1 byte
    int8_t   drift;             // Offset 81:  Frequency drift, 1 byte
    uint16_t distance;          // Offset 82:  Distance in km, 2 bytes
    uint16_t azimuth;           // Offset 84:  Bearing in degrees, 2 bytes
    char     _padding1b[2];     // Offset 86:  Padding for int32 alignment
    int32_t  band;              // Offset 88:  ADIF band ID, 4 bytes
    char     _padding2[4];      // Offset 92:  Padding to align Mode to 8-byte boundary
    char     mode[8];           // Offset 96:  WSPR mode, 8 bytes
    char     version[8];        // Offset 104: WSPR version, 8 bytes
    int8_t   code;              // Offset 112: Status code, 1 byte
    uint8_t  column_count;      // Offset 113: Column count, 1 byte
    char     _padding3[14];     // Offset 114: Final padding to reach 128 bytes (16-byte aligned)
} WsprSpotC;

// ============================================================================
// CUDA Memory Management
// ============================================================================

// Allocate pinned host memory (zero-copy capable)
CudaError cuda_host_alloc_pinned(void** ptr, size_t size);

// Free pinned host memory
CudaError cuda_free_host(void* ptr);

// Allocate device memory (GPU VRAM)
CudaError cuda_malloc_device(void** ptr, size_t size);

// Free device memory
CudaError cuda_free_device(void* ptr);

// ============================================================================
// CUDA Memory Transfers (Async)
// ============================================================================

// Async copy from host to device
CudaError cuda_memcpy_host_to_device_async(void* dst, const void* src, size_t size, CudaStream stream);

// Async copy from device to host
CudaError cuda_memcpy_device_to_host_async(void* dst, const void* src, size_t size, CudaStream stream);

// ============================================================================
// CUDA Stream Management
// ============================================================================

// Create CUDA stream for async operations
CudaError cuda_stream_create(CudaStream* stream);

// Destroy CUDA stream
CudaError cuda_stream_destroy(CudaStream stream);

// Synchronize stream (blocking wait for all operations to complete)
CudaError cuda_stream_synchronize(CudaStream stream);

// Synchronize device (blocking wait for all device operations to complete)
// BLACKWELL FIX: Use this instead of stream synchronize on RTX 5090 to avoid race conditions
CudaError cuda_device_synchronize();

// Query stream status (non-blocking check if all operations complete)
CudaError cuda_stream_query(CudaStream stream);

// ============================================================================
// CUDA Device Information
// ============================================================================

// Get device properties (compute capability, memory, SM count)
CudaError cuda_get_device_properties(
    int* compute_capability_major,
    int* compute_capability_minor,
    size_t* total_memory,
    int* multiprocessor_count
);

// Get current GPU memory usage
CudaError cuda_get_memory_info(size_t* free_bytes, size_t* total_bytes);

// Get last CUDA error string (for debugging)
const char* cuda_get_last_error_string(void);

// ============================================================================
// CUDA Kernels
// ============================================================================

// Validation kernel: Check if WSPR spots meet validity criteria
CudaError cuda_kernel_validate_spots(
    WsprSpotC* d_spots,
    int* d_valid_flags,
    int count,
    CudaStream stream
);

// Vectorized processing kernel: Normalize + validate + convert frequency
CudaError cuda_kernel_process_vectorized(
    const WsprSpotC* d_input,
    WsprSpotC* d_output,
    int* d_valid_flags,
    uint64_t* d_freq_fixed,
    int count,
    CudaStream stream
);

// Deduplication kernel: Remove duplicate WSPR spots on GPU
CudaError cuda_kernel_deduplicate(
    const WsprSpotC* d_input,
    WsprSpotC* d_temp,
    WsprSpotC* d_output,
    int* d_unique_flags,
    int* d_scan,
    int* d_unique_count,
    int count,
    CudaStream stream
);

// Callsign sanitizer kernel: Strip unwanted characters from callsign/reporter fields
CudaError cuda_kernel_sanitize_callsigns(
    WsprSpotC* d_spots,
    int* d_modified_flags,
    int count,
    CudaStream stream
);

#ifdef __cplusplus
}
#endif

#endif // WSPR_CUDA_BRIDGE_H
