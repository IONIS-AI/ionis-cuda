/**
 * @file bulk_kernels.h
 * @brief CUDA Bulk Kernel Interface for Structure-of-Arrays (SoA) Processing
 *
 * These functions provide the C interface to CUDA kernels that operate on
 * flat column arrays, enabling efficient bulk data transfer from Go/CGO.
 */

#ifndef BULK_KERNELS_H
#define BULK_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/**
 * @brief Bulk sanitize callsigns and reporters
 *
 * Strips unwanted characters (quotes, brackets) from callsign and reporter
 * columns. Operates on flat byte arrays where each spot's data is at
 * offset [idx * field_size].
 *
 * @param d_callsign  Device pointer to callsign array [N * 16 bytes]
 * @param d_reporter  Device pointer to reporter array [N * 16 bytes]
 * @param d_modified  Device pointer to modification flags [N * 4 bytes]
 * @param count       Number of spots to process
 * @param stream      CUDA stream for async execution
 * @return 0 on success, CUDA error code on failure
 */
int cuda_kernel_bulk_sanitize_callsigns(
    char* d_callsign,
    char* d_reporter,
    int* d_modified,
    int count,
    void* stream
);

/**
 * @brief Bulk validate WSPR spot columns
 *
 * Validates SNR, power, frequency, distance, and band columns in parallel.
 * Each thread validates one spot across all columns.
 *
 * @param d_snr       Device pointer to SNR array [N bytes]
 * @param d_power     Device pointer to power array [N bytes]
 * @param d_freq      Device pointer to frequency array [N * 8 bytes] (Hz)
 * @param d_distance  Device pointer to distance array [N * 4 bytes] (km)
 * @param d_band      Device pointer to band array [N * 4 bytes]
 * @param d_valid     Device pointer to validation flags [N * 4 bytes] (output)
 * @param count       Number of spots to validate
 * @param stream      CUDA stream for async execution
 * @return 0 on success, CUDA error code on failure
 */
int cuda_kernel_bulk_validate(
    const int8_t* d_snr,
    const int8_t* d_power,
    const uint64_t* d_freq,
    const uint32_t* d_distance,
    const int32_t* d_band,
    int* d_valid,
    int count,
    void* stream
);

#ifdef __cplusplus
}
#endif

#endif // BULK_KERNELS_H
