/**
 * @file bulk_kernels.cu
 * @brief CUDA Bulk Kernels for Structure-of-Arrays (SoA) WSPR Processing
 *
 * These kernels operate on flat column arrays instead of struct arrays,
 * enabling efficient bulk data transfer and coalesced memory access.
 *
 * Optimized for RTX 5090 Blackwell (sm_120):
 *   - 170 SMs, 128 threads per warp
 *   - 32GB VRAM with high bandwidth
 *   - Coalesced access patterns for maximum throughput
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>

// Kernel configuration for Blackwell
#define BLOCK_SIZE 256
#define WARP_SIZE 32

// String field sizes
#define CALLSIGN_SIZE 16
#define REPORTER_SIZE 16

// Characters to strip from callsigns/reporters
__device__ __constant__ char g_strip_chars[] = "\"'`<>[]{}()";
#define NUM_STRIP_CHARS 10

/**
 * @brief Check if character should be stripped
 */
__device__ __forceinline__ bool should_strip(char c) {
    #pragma unroll
    for (int i = 0; i < NUM_STRIP_CHARS; i++) {
        if (c == g_strip_chars[i]) return true;
    }
    return false;
}

/**
 * @brief Bulk sanitize callsigns and reporters (SoA layout)
 *
 * This kernel processes callsign and reporter columns in parallel,
 * stripping unwanted characters (quotes, brackets, etc.).
 *
 * Memory Layout:
 *   callsign[N * 16]: Flat array of callsigns, 16 bytes per spot
 *   reporter[N * 16]: Flat array of reporters, 16 bytes per spot
 *
 * Each thread handles one spot's callsign AND reporter for locality.
 *
 * @param d_callsign  Flat callsign array [N][16]
 * @param d_reporter  Flat reporter array [N][16]
 * @param d_modified  Per-spot modification flags [N]
 * @param count       Number of spots to process
 */
__global__ void kernel_bulk_sanitize_callsigns(
    char* __restrict__ d_callsign,
    char* __restrict__ d_reporter,
    int* __restrict__ d_modified,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    int modified = 0;

    // Process callsign for this spot
    char* callsign = &d_callsign[idx * CALLSIGN_SIZE];
    int write_pos = 0;

    #pragma unroll
    for (int i = 0; i < CALLSIGN_SIZE; i++) {
        char c = callsign[i];
        if (c == '\0') break;
        if (!should_strip(c)) {
            callsign[write_pos++] = c;
        } else {
            modified = 1;
        }
    }
    // Null-pad remainder
    while (write_pos < CALLSIGN_SIZE) {
        callsign[write_pos++] = '\0';
    }

    // Process reporter for this spot
    char* reporter = &d_reporter[idx * REPORTER_SIZE];
    write_pos = 0;

    #pragma unroll
    for (int i = 0; i < REPORTER_SIZE; i++) {
        char c = reporter[i];
        if (c == '\0') break;
        if (!should_strip(c)) {
            reporter[write_pos++] = c;
        } else {
            modified = 1;
        }
    }
    // Null-pad remainder
    while (write_pos < REPORTER_SIZE) {
        reporter[write_pos++] = '\0';
    }

    d_modified[idx] = modified;
}

/**
 * @brief Bulk validate WSPR spot data (SoA layout)
 *
 * Validates multiple columns in parallel:
 *   - SNR: -50 to +50 dB
 *   - Power: -30 to +60 dBm
 *   - Frequency: 1 MHz to 500 MHz (in Hz)
 *   - Distance: 0 to 50000 km
 *   - Band: valid ADIF band ID (1-200)
 *
 * @param d_snr       SNR array [N]
 * @param d_power     Power array [N]
 * @param d_freq      Frequency array [N] (Hz)
 * @param d_distance  Distance array [N] (km)
 * @param d_band      Band array [N]
 * @param d_valid     Output validation flags [N]
 * @param count       Number of spots to validate
 */
__global__ void kernel_bulk_validate(
    const int8_t* __restrict__ d_snr,
    const int8_t* __restrict__ d_power,
    const uint64_t* __restrict__ d_freq,
    const uint32_t* __restrict__ d_distance,
    const int32_t* __restrict__ d_band,
    int* __restrict__ d_valid,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    int valid = 1;

    // Validate SNR: -50 to +50 dB
    int8_t snr = d_snr[idx];
    if (snr < -50 || snr > 50) {
        valid = 0;
    }

    // Validate Power: -30 to +60 dBm
    int8_t power = d_power[idx];
    if (power < -30 || power > 60) {
        valid = 0;
    }

    // Validate Frequency: 1 MHz to 500 MHz (in Hz)
    uint64_t freq = d_freq[idx];
    if (freq < 1000000ULL || freq > 500000000ULL) {
        valid = 0;
    }

    // Validate Distance: 0 to 50000 km
    uint32_t distance = d_distance[idx];
    if (distance > 50000) {
        valid = 0;
    }

    // Validate Band: 1 to 200 (ADIF band IDs)
    int32_t band = d_band[idx];
    if (band < 1 || band > 200) {
        valid = 0;
    }

    d_valid[idx] = valid;
}

// =============================================================================
// C Interface for CGO
// =============================================================================

extern "C" {

/**
 * @brief Launch bulk sanitize kernel
 */
int cuda_kernel_bulk_sanitize_callsigns(
    char* d_callsign,
    char* d_reporter,
    int* d_modified,
    int count,
    cudaStream_t stream
) {
    if (count <= 0) return 0;

    int num_blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;

    kernel_bulk_sanitize_callsigns<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        d_callsign,
        d_reporter,
        d_modified,
        count
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "bulk_sanitize kernel launch failed: %s\n", cudaGetErrorString(err));
        return (int)err;
    }

    return 0;
}

/**
 * @brief Launch bulk validate kernel
 */
int cuda_kernel_bulk_validate(
    const int8_t* d_snr,
    const int8_t* d_power,
    const uint64_t* d_freq,
    const uint32_t* d_distance,
    const int32_t* d_band,
    int* d_valid,
    int count,
    cudaStream_t stream
) {
    if (count <= 0) return 0;

    int num_blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;

    kernel_bulk_validate<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        d_snr,
        d_power,
        d_freq,
        d_distance,
        d_band,
        d_valid,
        count
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "bulk_validate kernel launch failed: %s\n", cudaGetErrorString(err));
        return (int)err;
    }

    return 0;
}

} // extern "C"
