/*
 * wspr-cuda-check - CUDA HAL Self-Test Utility
 *
 * Purpose: Verify CUDA toolkit handshake and GPU compute capability
 * Part of: ki7mt-ai-lab-cuda (Sovereign CUDA HAL)
 *
 * Usage: wspr-cuda-check [--verbose] [--help] [--version]
 *
 * Exit codes:
 *   0 - Success (GPU detected and kernel executed)
 *   1 - No CUDA-capable GPU detected
 *   2 - Kernel execution failed
 *   3 - Library initialization failed
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bridge.h"

/* Version info (substituted by build) */
#ifndef VERSION
#define VERSION "1.1.6"
#endif

/* ANSI color codes */
#define COLOR_GREEN  "\033[32m"
#define COLOR_RED    "\033[31m"
#define COLOR_YELLOW "\033[33m"
#define COLOR_RESET  "\033[0m"

/* Compute capability names */
static const char* get_arch_name(int major, int minor) {
    int sm = major * 10 + minor;
    switch (sm) {
        case 80: return "Ampere (A100)";
        case 86: return "Ampere (RTX 30xx)";
        case 87: return "Ampere (Jetson Orin)";
        case 89: return "Ada Lovelace (RTX 40xx)";
        case 90: return "Hopper (H100)";
        case 100: return "Blackwell (B100/B200)";
        case 120: return "Blackwell (RTX 5090)";
        default:
            if (major >= 10) return "Blackwell (Unknown)";
            return "Unknown";
    }
}

/* Format bytes as human-readable */
static void format_bytes(size_t bytes, char* buf, size_t buf_size) {
    if (bytes >= 1024ULL * 1024 * 1024) {
        snprintf(buf, buf_size, "%.1f GB", (double)bytes / (1024.0 * 1024.0 * 1024.0));
    } else if (bytes >= 1024ULL * 1024) {
        snprintf(buf, buf_size, "%.1f MB", (double)bytes / (1024.0 * 1024.0));
    } else {
        snprintf(buf, buf_size, "%.1f KB", (double)bytes / 1024.0);
    }
}

int main(int argc, char* argv[]) {
    int verbose = 0;
    CudaError err;

    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0) {
            verbose = 1;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("wspr-cuda-check v%s - CUDA HAL Self-Test Utility\n", VERSION);
            printf("\n");
            printf("Usage: %s [OPTIONS]\n", argv[0]);
            printf("\n");
            printf("Options:\n");
            printf("  -v, --verbose  Show detailed GPU information\n");
            printf("  -h, --help     Show this help message\n");
            printf("  -V, --version  Show version information\n");
            printf("\n");
            printf("Exit codes:\n");
            printf("  0 - Success (GPU detected and kernel executed)\n");
            printf("  1 - No CUDA-capable GPU detected\n");
            printf("  2 - Kernel execution failed\n");
            printf("  3 - Library initialization failed\n");
            return 0;
        } else if (strcmp(argv[i], "--version") == 0 || strcmp(argv[i], "-V") == 0) {
            printf("wspr-cuda-check v%s\n", VERSION);
            return 0;
        }
    }

    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│  wspr-cuda-check v%s                                      │\n", VERSION);
    printf("│  ki7mt-ai-lab-cuda Sovereign HAL Self-Test                  │\n");
    printf("└─────────────────────────────────────────────────────────────┘\n");
    printf("\n");

    /* Step 1: Get device properties */
    printf("[1/3] Detecting CUDA device...\n");

    int cc_major = 0, cc_minor = 0;
    size_t total_mem = 0;
    int sm_count = 0;

    err = cuda_get_device_properties(&cc_major, &cc_minor, &total_mem, &sm_count);

    if (err != CUDA_SUCCESS) {
        printf("      " COLOR_RED "FAILED" COLOR_RESET ": %s\n", cuda_get_last_error_string());
        printf("\n");
        printf("No CUDA-capable GPU detected.\n");
        printf("Ensure NVIDIA driver is installed and GPU is present.\n");
        return 1;
    }

    char mem_str[32];
    format_bytes(total_mem, mem_str, sizeof(mem_str));

    printf("      " COLOR_GREEN "OK" COLOR_RESET ": GPU detected\n");
    printf("\n");
    printf("      Compute Capability: sm_%d%d (%s)\n", cc_major, cc_minor, get_arch_name(cc_major, cc_minor));
    printf("      Total Memory:       %s\n", mem_str);
    printf("      Multiprocessors:    %d SMs\n", sm_count);

    if (verbose) {
        size_t free_mem = 0, total_mem_info = 0;
        err = cuda_get_memory_info(&free_mem, &total_mem_info);
        if (err == CUDA_SUCCESS) {
            char free_str[32];
            format_bytes(free_mem, free_str, sizeof(free_str));
            printf("      Free Memory:        %s\n", free_str);
        }
    }
    printf("\n");

    /* Step 2: Create CUDA stream */
    printf("[2/3] Creating CUDA stream...\n");

    CudaStream stream = NULL;
    err = cuda_stream_create(&stream);

    if (err != CUDA_SUCCESS) {
        printf("      " COLOR_RED "FAILED" COLOR_RESET ": %s\n", cuda_get_last_error_string());
        return 3;
    }

    printf("      " COLOR_GREEN "OK" COLOR_RESET ": Stream created\n");
    printf("\n");

    /* Step 3: Allocate and run dummy kernel */
    printf("[3/3] Running validation kernel (1 block, 256 threads)...\n");

    /* Allocate small test buffer */
    const int test_count = 256;
    const size_t spot_size = sizeof(WsprSpotC);
    const size_t buffer_size = test_count * spot_size;

    void* d_spots = NULL;
    void* d_flags = NULL;

    err = cuda_malloc_device(&d_spots, buffer_size);
    if (err != CUDA_SUCCESS) {
        printf("      " COLOR_RED "FAILED" COLOR_RESET ": Device malloc (spots): %s\n", cuda_get_last_error_string());
        cuda_stream_destroy(stream);
        return 2;
    }

    err = cuda_malloc_device(&d_flags, test_count * sizeof(int));
    if (err != CUDA_SUCCESS) {
        printf("      " COLOR_RED "FAILED" COLOR_RESET ": Device malloc (flags): %s\n", cuda_get_last_error_string());
        cuda_free_device(d_spots);
        cuda_stream_destroy(stream);
        return 2;
    }

    /* Run validation kernel (will process zeroed memory) */
    err = cuda_kernel_validate_spots((WsprSpotC*)d_spots, (int*)d_flags, test_count, stream);

    if (err != CUDA_SUCCESS) {
        printf("      " COLOR_RED "FAILED" COLOR_RESET ": Kernel launch: %s\n", cuda_get_last_error_string());
        cuda_free_device(d_spots);
        cuda_free_device(d_flags);
        cuda_stream_destroy(stream);
        return 2;
    }

    /* Synchronize and check for errors */
    err = cuda_device_synchronize();

    if (err != CUDA_SUCCESS) {
        printf("      " COLOR_RED "FAILED" COLOR_RESET ": Synchronize: %s\n", cuda_get_last_error_string());
        cuda_free_device(d_spots);
        cuda_free_device(d_flags);
        cuda_stream_destroy(stream);
        return 2;
    }

    printf("      " COLOR_GREEN "OK" COLOR_RESET ": Kernel executed successfully\n");
    printf("\n");

    /* Cleanup */
    cuda_free_device(d_spots);
    cuda_free_device(d_flags);
    cuda_stream_destroy(stream);

    /* Summary */
    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│  " COLOR_GREEN "PASS" COLOR_RESET ": CUDA HAL handshake successful                      │\n");
    printf("└─────────────────────────────────────────────────────────────┘\n");
    printf("\n");
    printf("GPU:     sm_%d%d (%s)\n", cc_major, cc_minor, get_arch_name(cc_major, cc_minor));
    printf("Memory:  %s\n", mem_str);
    printf("SMs:     %d\n", sm_count);
    printf("\n");

    /* Check if supported architecture */
    int sm = cc_major * 10 + cc_minor;
    if (sm < 80) {
        printf(COLOR_YELLOW "WARNING" COLOR_RESET ": GPU compute capability sm_%d is below minimum (sm_80).\n", sm);
        printf("         Some kernels may not be available.\n");
    } else if (sm >= 100) {
        printf(COLOR_GREEN "OPTIMAL" COLOR_RESET ": Blackwell (sm_%d) detected - full performance available.\n", sm);
    }

    return 0;
}
