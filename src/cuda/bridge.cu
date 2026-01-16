#include "bridge.h"
#include <cuda_runtime.h>
#include <stdio.h>

// Convert CUDA runtime error to our enum
static CudaError convert_cuda_error(cudaError_t err) {
    switch (err) {
        case cudaSuccess:
            return CUDA_SUCCESS;
        case cudaErrorMemoryAllocation:
            return CUDA_ERROR_MEMORY_ALLOCATION;
        case cudaErrorInvalidValue:
            return CUDA_ERROR_INVALID_VALUE;
        case cudaErrorNotReady:
            return CUDA_ERROR_NOT_READY;
        default:
            return (CudaError)err;
    }
}

// Pinned memory allocation with cudaHostAllocMapped for zero-copy access
CudaError cuda_host_alloc_pinned(void** ptr, size_t size) {
    if (ptr == NULL || size == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    // cudaHostAllocMapped allows GPU to access host memory directly
    // cudaHostAllocWriteCombined optimizes host-to-device transfers
    unsigned int flags = cudaHostAllocMapped | cudaHostAllocWriteCombined;
    cudaError_t err = cudaHostAlloc(ptr, size, flags);

    if (err != cudaSuccess) {
        fprintf(stderr, "cudaHostAlloc failed: %s\n", cudaGetErrorString(err));
        *ptr = NULL;
    }

    return convert_cuda_error(err);
}

// Free pinned host memory
CudaError cuda_free_host(void* ptr) {
    if (ptr == NULL) {
        return CUDA_SUCCESS;
    }

    cudaError_t err = cudaFreeHost(ptr);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaFreeHost failed: %s\n", cudaGetErrorString(err));
    }

    return convert_cuda_error(err);
}

// Allocate device memory
CudaError cuda_malloc_device(void** ptr, size_t size) {
    if (ptr == NULL || size == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    cudaError_t err = cudaMalloc(ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        *ptr = NULL;
    }

    return convert_cuda_error(err);
}

// Free device memory
CudaError cuda_free_device(void* ptr) {
    if (ptr == NULL) {
        return CUDA_SUCCESS;
    }

    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaFree failed: %s\n", cudaGetErrorString(err));
    }

    return convert_cuda_error(err);
}

// Async copy from host to device
CudaError cuda_memcpy_host_to_device_async(void* dst, const void* src, size_t size, CudaStream stream) {
    if (dst == NULL || src == NULL || size == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, (cudaStream_t)stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyAsync H2D failed: %s\n", cudaGetErrorString(err));
    }

    return convert_cuda_error(err);
}

// Async copy from device to host
CudaError cuda_memcpy_device_to_host_async(void* dst, const void* src, size_t size, CudaStream stream) {
    if (dst == NULL || src == NULL || size == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, (cudaStream_t)stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyAsync D2H failed: %s\n", cudaGetErrorString(err));
    }

    return convert_cuda_error(err);
}

// Create CUDA stream
CudaError cuda_stream_create(CudaStream* stream) {
    if (stream == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    cudaError_t err = cudaStreamCreate((cudaStream_t*)stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaStreamCreate failed: %s\n", cudaGetErrorString(err));
        *stream = NULL;
    }

    return convert_cuda_error(err);
}

// Destroy CUDA stream
CudaError cuda_stream_destroy(CudaStream stream) {
    if (stream == NULL) {
        return CUDA_SUCCESS;
    }

    cudaError_t err = cudaStreamDestroy((cudaStream_t)stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaStreamDestroy failed: %s\n", cudaGetErrorString(err));
    }

    return convert_cuda_error(err);
}

// Synchronize stream
CudaError cuda_stream_synchronize(CudaStream stream) {
    cudaError_t err = cudaStreamSynchronize((cudaStream_t)stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaStreamSynchronize failed: %s\n", cudaGetErrorString(err));
    }

    return convert_cuda_error(err);
}

// Synchronize device (BLACKWELL FIX for RTX 5090)
// Forces all device work to complete before returning
CudaError cuda_device_synchronize() {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
    }

    return convert_cuda_error(err);
}

// Query stream status (non-blocking)
CudaError cuda_stream_query(CudaStream stream) {
    cudaError_t err = cudaStreamQuery((cudaStream_t)stream);
    return convert_cuda_error(err);
}

// Get device properties
CudaError cuda_get_device_properties(
    int* compute_capability_major,
    int* compute_capability_minor,
    size_t* total_memory,
    int* multiprocessor_count
) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);

    if (err == cudaSuccess) {
        if (compute_capability_major) *compute_capability_major = prop.major;
        if (compute_capability_minor) *compute_capability_minor = prop.minor;
        if (total_memory) *total_memory = prop.totalGlobalMem;
        if (multiprocessor_count) *multiprocessor_count = prop.multiProcessorCount;
    } else {
        fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
    }

    return convert_cuda_error(err);
}

// Get current GPU memory usage
CudaError cuda_get_memory_info(size_t* free_bytes, size_t* total_bytes) {
    cudaError_t err = cudaMemGetInfo(free_bytes, total_bytes);

    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemGetInfo failed: %s\n", cudaGetErrorString(err));
        if (free_bytes) *free_bytes = 0;
        if (total_bytes) *total_bytes = 0;
    }

    return convert_cuda_error(err);
}

// Get last error string
const char* cuda_get_last_error_string() {
    return cudaGetErrorString(cudaGetLastError());
}

// ============================================================================
// CUDA KERNEL: Placeholder WSPR Validation
// ============================================================================

__global__ void kernel_validate_spots(
    const WsprSpotC* spots,
    int* valid_flags,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= count) {
        return;
    }

    const WsprSpotC* spot = &spots[idx];

    // Placeholder validation logic:
    // 1. SNR must be >= -30 dB and <= 10 dB (typical WSPR range)
    // 2. Power must be 0-60 dBm (WSPR spec allows 0-60 dBm, typically 23-37)
    // 3. Frequency must be positive
    // 4. Distance must be > 0 (can't receive your own transmission in typical setup)

    int is_valid = 1;

    // Validate SNR
    if (spot->snr < -30 || spot->snr > 10) {
        is_valid = 0;
    }

    // Validate Power
    if (spot->power < 0 || spot->power > 60) {
        is_valid = 0;
    }

    // Validate Frequency (must be positive)
    if (spot->frequency <= 0.0) {
        is_valid = 0;
    }

    // Validate Distance (must be > 0)
    if (spot->distance == 0) {
        is_valid = 0;
    }

    // Validate Band (ADIF-MCP v.3.1.6 full spectrum)
    // Valid band ranges:
    //   0 = Unknown (invalid)
    //   1-11 = ITU classifications (ELF to Infrared)
    //   100-111 = HF amateur bands (2190m-10m)
    //   200-203 = VHF amateur bands (6m-1.25m)
    //   300-303 = UHF amateur bands (70cm-13cm)
    //   400-403 = SHF amateur bands (9cm-1.2cm)
    if (spot->band < 0) {
        is_valid = 0;  // Negative band IDs are invalid
    }
    // Note: Band = 0 is allowed (unknown frequency, will be filtered in post-processing)

    valid_flags[idx] = is_valid;
}

// Kernel launcher
CudaError cuda_kernel_validate_spots(
    WsprSpotC* d_spots,
    int* d_valid_flags,
    int count,
    CudaStream stream
) {
    if (d_spots == NULL || d_valid_flags == NULL || count <= 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    // Optimal block size for RTX 5090 (high occupancy)
    int block_size = 256;
    int grid_size = (count + block_size - 1) / block_size;

    kernel_validate_spots<<<grid_size, block_size, 0, (cudaStream_t)stream>>>(
        d_spots,
        d_valid_flags,
        count
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "kernel_validate_spots launch failed: %s\n", cudaGetErrorString(err));
    }

    return convert_cuda_error(err);
}

// ============================================================================
// CUDA KERNEL: Vectorized WSPR Processing
// ============================================================================
//
// OPTIMIZATION STRATEGY for RTX 5090:
//
// 1. VECTORIZED MEMORY ACCESS (uint4 loads/stores):
//    - WsprSpotC struct is 96 bytes (aligned to 16-byte boundary)
//    - Each struct = 6 x uint4 (6 x 16 bytes)
//    - Using uint4 enables 128-bit coalesced transactions
//    - RTX 5090 has 128-byte cache lines, loading 96 bytes efficiently
//
// 2. STREAM-BASED OVERLAP:
//    - Kernel executes on cudaStream_t for async operation
//    - CPU can prepare next mmap batch while GPU processes current batch
//    - Enables overlap: CPU mmap I/O || GPU kernel execution
//
// 3. REGISTER OPTIMIZATION:
//    - __launch_bounds__(256) hints optimal register allocation
//    - RTX 5090 has 65536 registers per SM
//    - 256 threads/block ensures high occupancy with complex kernels
//
// 4. TAIL HANDLING:
//    - Early exit (idx >= count) handles non-multiple-of-256 batch sizes
//    - No warp divergence penalty for small tails
//
// ============================================================================

// Device helper: Convert character to uppercase (for grid locator normalization)
__device__ __forceinline__ char to_upper(char c) {
    return (c >= 'a' && c <= 'z') ? (c - 32) : c;
}

// Device helper: Normalize grid locator to uppercase
__device__ void normalize_grid(char* grid, int len) {
    #pragma unroll
    for (int i = 0; i < len; i++) {
        grid[i] = to_upper(grid[i]);
    }
}

// Device helper: Convert frequency (MHz) to fixed-point for efficient sorting
// Multiply by 1,000,000 to preserve 6 decimal places
__device__ __forceinline__ uint64_t freq_to_fixed(double freq_mhz) {
    return (uint64_t)(freq_mhz * 1000000.0);
}

// Device helper: Validate WSPR spot (same logic as kernel_validate_spots)
__device__ int validate_spot(const WsprSpotC* spot) {
    // SNR range: -30 to 10 dB
    if (spot->snr < -30 || spot->snr > 10) {
        return 0;
    }

    // Power range: 0 to 60 dBm
    if (spot->power < 0 || spot->power > 60) {
        return 0;
    }

    // Frequency must be positive
    if (spot->frequency <= 0.0) {
        return 0;
    }

    // Distance must be > 0
    if (spot->distance == 0) {
        return 0;
    }

    // Valid ADIF-MCP v.3.1.6 band ranges
    // Allow all non-negative band IDs (full spectrum coverage)
    // 0 = Unknown, 1-11 = ITU, 100+ = Amateur bands
    if (spot->band < 0) {
        return 0;  // Negative band IDs are invalid
    }

    return 1;
}

// Vectorized processing kernel optimized for RTX 5090
// Uses __launch_bounds__ to optimize register usage
__global__ __launch_bounds__(256)
void kernel_process_vectorized(
    const WsprSpotC* __restrict__ d_input,
    WsprSpotC* __restrict__ d_output,
    int* __restrict__ d_valid_flags,
    uint64_t* __restrict__ d_freq_fixed,
    int count
) {
    // Global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Early exit for out-of-bounds threads (handles tail)
    if (idx >= count) {
        return;
    }

    // Vectorized read using uint4 (16-byte aligned loads)
    // WsprSpotC is 96 bytes = 6 x uint4 (6 x 16 bytes)
    // This enables coalesced memory access on RTX 5090
    // Each uint4 load reads 16 bytes in a single 128-bit transaction
    const uint4* d_input_vec = reinterpret_cast<const uint4*>(d_input);
    uint4 data[6];  // 6 x 16 bytes = 96 bytes (full WsprSpotC struct)

    // Load 6 uint4 vectors to read complete WsprSpotC struct
    #pragma unroll
    for (int i = 0; i < 6; i++) {
        data[i] = d_input_vec[idx * 6 + i];
    }

    // Reinterpret vectorized data as WsprSpotC struct
    WsprSpotC spot = *reinterpret_cast<WsprSpotC*>(data);

    // ========================================================================
    // NORMALIZATION PASS
    // ========================================================================

    // 1. Normalize grid locators to uppercase (standardization)
    //    This ensures "fn42" and "FN42" are treated identically
    normalize_grid(spot.reporter_grid, 8);
    normalize_grid(spot.grid, 8);

    // 2. Convert frequency to fixed-point for efficient sorting downstream
    //    Converts MHz (14.097100) to uint64 (14097100)
    uint64_t freq_fixed = freq_to_fixed(spot.frequency);

    // 3. Validate spot using same rules as validation kernel
    int is_valid = validate_spot(&spot);

    // ========================================================================
    // WRITE RESULTS
    // ========================================================================

    // Vectorized write using uint4 (16-byte aligned stores)
    // Write normalized spot back using vectorized stores for coalesced memory access
    uint4* d_output_vec = reinterpret_cast<uint4*>(d_output);
    const uint4* spot_vec = reinterpret_cast<const uint4*>(&spot);

    #pragma unroll
    for (int i = 0; i < 6; i++) {
        d_output_vec[idx * 6 + i] = spot_vec[i];
    }

    // Write validation result and fixed-point frequency
    d_valid_flags[idx] = is_valid;     // Validation result
    d_freq_fixed[idx] = freq_fixed;    // Fixed-point frequency for sorting
}

// Kernel launcher for vectorized processing
CudaError cuda_kernel_process_vectorized(
    const WsprSpotC* d_input,
    WsprSpotC* d_output,
    int* d_valid_flags,
    uint64_t* d_freq_fixed,
    int count,
    CudaStream stream
) {
    if (d_input == NULL || d_output == NULL || d_valid_flags == NULL ||
        d_freq_fixed == NULL || count <= 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    // Optimal block size for RTX 5090 with __launch_bounds__(256)
    // This ensures high occupancy and optimal register allocation
    int block_size = 256;
    int grid_size = (count + block_size - 1) / block_size;

    // Launch vectorized kernel on specified stream (enables overlap with CPU)
    kernel_process_vectorized<<<grid_size, block_size, 0, (cudaStream_t)stream>>>(
        d_input,
        d_output,
        d_valid_flags,
        d_freq_fixed,
        count
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "kernel_process_vectorized launch failed: %s\n", cudaGetErrorString(err));
    }

    return convert_cuda_error(err);
}

// ============================================================================
// CUDA KERNEL: On-GPU Deduplication with Shared Memory
// ============================================================================
//
// DEDUPLICATION STRATEGY:
//
// 1. HASH-BASED DUPLICATE DETECTION:
//    - Hash key: (callsign, reporter, timestamp_minute)
//    - Collisions are resolved by full comparison
//
// 2. SHARED MEMORY TILE PROCESSING:
//    - Each block loads a tile of spots into shared memory
//    - Block size: 256 threads, processes 256 spots per iteration
//    - Shared memory usage: ~24KB per block (within 48KB limit)
//
// 3. ATOMIC ELECTION:
//    - Use atomicCAS to elect one thread per unique spot
//    - Winner writes to global output buffer
//
// 4. STREAM COMPACTION:
//    - Use prefix sum to pack unique spots without holes
//    - Return exact count of unique spots
//
// ============================================================================

// Device helper: Compute hash of WSPR spot for deduplication
// Hash key: callsign + reporter + timestamp (rounded to minute)
__device__ __forceinline__ uint32_t compute_spot_hash(const WsprSpotC* spot) {
    uint32_t hash = 2166136261u; // FNV-1a offset basis

    // Hash callsign (16 bytes)
    for (int i = 0; i < 16 && spot->callsign[i] != 0; i++) {
        hash ^= (uint32_t)spot->callsign[i];
        hash *= 16777619u; // FNV-1a prime
    }

    // Hash reporter (16 bytes)
    for (int i = 0; i < 16 && spot->reporter[i] != 0; i++) {
        hash ^= (uint32_t)spot->reporter[i];
        hash *= 16777619u;
    }

    // Hash timestamp rounded to minute (60 seconds * 1e9 nanoseconds)
    int64_t timestamp_minute = spot->timestamp / 60000000000LL;
    hash ^= (uint32_t)(timestamp_minute & 0xFFFFFFFF);
    hash *= 16777619u;
    hash ^= (uint32_t)(timestamp_minute >> 32);
    hash *= 16777619u;

    return hash;
}

// Device helper: Compare two spots for equality (deduplication)
__device__ bool spots_equal(const WsprSpotC* a, const WsprSpotC* b) {
    // Compare timestamps (rounded to minute)
    int64_t a_minute = a->timestamp / 60000000000LL;
    int64_t b_minute = b->timestamp / 60000000000LL;
    if (a_minute != b_minute) return false;

    // Compare callsigns
    for (int i = 0; i < 16; i++) {
        if (a->callsign[i] != b->callsign[i]) return false;
        if (a->callsign[i] == 0) break; // Early exit on null terminator
    }

    // Compare reporters
    for (int i = 0; i < 16; i++) {
        if (a->reporter[i] != b->reporter[i]) return false;
        if (a->reporter[i] == 0) break;
    }

    return true;
}

// Shared memory tile-based deduplication kernel
// Uses __launch_bounds__(256) for optimal occupancy on RTX 5090
__global__ __launch_bounds__(256)
void kernel_deduplicate_shared(
    const WsprSpotC* __restrict__ d_input,
    WsprSpotC* __restrict__ d_temp,
    int* __restrict__ d_unique_flags,  // 1 if unique, 0 if duplicate
    int count
) {
    // Shared memory for tile processing
    // 256 spots * 96 bytes = 24,576 bytes (24KB, well under 48KB limit)
    extern __shared__ char shared_mem[];
    WsprSpotC* s_tile = reinterpret_cast<WsprSpotC*>(shared_mem);

    // Hash table follows tile data
    // 512 slots * 4 bytes = 2KB
    uint32_t* s_hash_table = reinterpret_cast<uint32_t*>(shared_mem + 256 * sizeof(WsprSpotC));
    int* s_hash_owners = reinterpret_cast<int*>(shared_mem + 256 * sizeof(WsprSpotC) + 512 * sizeof(uint32_t));

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize hash table (use 2 iterations for 256 threads to init 512 slots)
    s_hash_table[tid] = 0xFFFFFFFF; // Empty marker
    s_hash_owners[tid] = -1;
    s_hash_table[tid + 256] = 0xFFFFFFFF;
    s_hash_owners[tid + 256] = -1;

    __syncthreads();

    // Load spot into shared memory using vectorized read (only if in bounds)
    bool is_valid = (gid < count);

    if (is_valid) {
        const uint4* d_input_vec = reinterpret_cast<const uint4*>(d_input);
        uint4 data[6];

        #pragma unroll
        for (int i = 0; i < 6; i++) {
            data[i] = d_input_vec[gid * 6 + i];
        }

        s_tile[tid] = *reinterpret_cast<WsprSpotC*>(data);
    }

    __syncthreads();

    // Only process valid threads
    if (is_valid) {
        // Compute hash for this spot
        uint32_t hash = compute_spot_hash(&s_tile[tid]);

        // Linear probing to find slot in hash table
        // Avoid bank conflicts by using sequential access pattern
        int slot = hash % 512;
        bool is_unique = false;
        int probe_count = 0;

        while (probe_count < 512) {
            uint32_t old_hash = atomicCAS(&s_hash_table[slot], 0xFFFFFFFF, hash);

            if (old_hash == 0xFFFFFFFF) {
                // We claimed an empty slot - this is a unique spot in this tile
                s_hash_owners[slot] = tid;
                is_unique = true;
                break;
            } else if (old_hash == hash) {
                // Hash collision - check if it's the same spot
                int owner_tid = s_hash_owners[slot];
                if (owner_tid >= 0 && owner_tid < 256) {
                    if (spots_equal(&s_tile[tid], &s_tile[owner_tid])) {
                        // Duplicate spot - not unique
                        is_unique = false;
                        break;
                    }
                }
                // Hash collision but different spot - continue probing
            }

            // Linear probing: move to next slot
            slot = (slot + 1) % 512;
            probe_count++;
        }

        // If we exhausted probes, assume unique (hash table full)
        if (probe_count >= 512) {
            is_unique = true;
        }

        // Mark unique flag
        d_unique_flags[gid] = is_unique ? 1 : 0;

        // If unique, write to temp buffer (will be compacted later)
        if (is_unique) {
            // Vectorized write
            uint4* d_temp_vec = reinterpret_cast<uint4*>(d_temp);
            const uint4* spot_vec = reinterpret_cast<const uint4*>(&s_tile[tid]);

            #pragma unroll
            for (int i = 0; i < 6; i++) {
                d_temp_vec[gid * 6 + i] = spot_vec[i];
            }
        }
    }
    // Out-of-bounds threads do nothing (they only participated in __syncthreads)
}

// Prefix sum kernel for stream compaction (Blelloch scan algorithm)
// Computes exclusive prefix sum for compaction
__global__ void kernel_prefix_sum(
    const int* __restrict__ d_input,
    int* __restrict__ d_output,
    int count
) {
    extern __shared__ int s_data[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load data into shared memory
    s_data[tid] = (gid < count) ? d_input[gid] : 0;
    s_data[tid + blockDim.x] = (gid + blockDim.x < count) ? d_input[gid + blockDim.x] : 0;
    __syncthreads();

    // Up-sweep (reduction) phase
    int offset = 1;
    for (int d = blockDim.x; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            s_data[bi] += s_data[ai];
        }
        offset *= 2;
    }

    // Clear last element
    if (tid == 0) {
        s_data[blockDim.x * 2 - 1] = 0;
    }
    __syncthreads();

    // Down-sweep phase
    for (int d = 1; d < blockDim.x * 2; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;

            int temp = s_data[ai];
            s_data[ai] = s_data[bi];
            s_data[bi] += temp;
        }
    }
    __syncthreads();

    // Write results
    if (gid < count) {
        d_output[gid] = s_data[tid];
    }
    if (gid + blockDim.x < count) {
        d_output[gid + blockDim.x] = s_data[tid + blockDim.x];
    }
}

// Stream compaction kernel: pack unique spots into contiguous array
__global__ void kernel_compact(
    const WsprSpotC* __restrict__ d_temp,
    WsprSpotC* __restrict__ d_output,
    const int* __restrict__ d_unique_flags,
    const int* __restrict__ d_scan,  // Prefix sum of unique_flags
    int count
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid >= count) {
        return;
    }

    // If this spot is unique, copy it to compacted position
    if (d_unique_flags[gid]) {
        int output_idx = d_scan[gid];

        // Vectorized copy
        const uint4* d_temp_vec = reinterpret_cast<const uint4*>(d_temp);
        uint4* d_output_vec = reinterpret_cast<uint4*>(d_output);

        #pragma unroll
        for (int i = 0; i < 6; i++) {
            d_output_vec[output_idx * 6 + i] = d_temp_vec[gid * 6 + i];
        }
    }
}

// Kernel launcher: Deduplication with shared memory
CudaError cuda_kernel_deduplicate(
    const WsprSpotC* d_input,
    WsprSpotC* d_temp,          // Temporary buffer (same size as input)
    WsprSpotC* d_output,        // Compacted output
    int* d_unique_flags,        // Temporary: 1 if unique, 0 if duplicate
    int* d_scan,                // Temporary: prefix sum buffer
    int* d_unique_count,        // Output: number of unique spots (on device)
    int count,
    CudaStream stream
) {
    if (d_input == NULL || d_temp == NULL || d_output == NULL ||
        d_unique_flags == NULL || d_scan == NULL || d_unique_count == NULL || count <= 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    const int block_size = 256;
    const int grid_size = (count + block_size - 1) / block_size;

    // Shared memory size: 256 spots (24KB) + 512 hash slots (2KB) + 512 owners (2KB) = 28KB
    const size_t shared_mem_size = 256 * sizeof(WsprSpotC) + 512 * sizeof(uint32_t) + 512 * sizeof(int);

    // Step 1: Deduplicate within blocks using shared memory
    kernel_deduplicate_shared<<<grid_size, block_size, shared_mem_size, (cudaStream_t)stream>>>(
        d_input,
        d_temp,
        d_unique_flags,
        count
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "kernel_deduplicate_shared launch failed: %s\n", cudaGetErrorString(err));
        return convert_cuda_error(err);
    }

    // Step 2: Prefix sum for stream compaction
    const int scan_block_size = 128;
    const int scan_grid_size = (count + scan_block_size * 2 - 1) / (scan_block_size * 2);
    const size_t scan_shared_mem = scan_block_size * 2 * sizeof(int);

    kernel_prefix_sum<<<scan_grid_size, scan_block_size, scan_shared_mem, (cudaStream_t)stream>>>(
        d_unique_flags,
        d_scan,
        count
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "kernel_prefix_sum launch failed: %s\n", cudaGetErrorString(err));
        return convert_cuda_error(err);
    }

    // Step 3: Compact unique spots
    kernel_compact<<<grid_size, block_size, 0, (cudaStream_t)stream>>>(
        d_temp,
        d_output,
        d_unique_flags,
        d_scan,
        count
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "kernel_compact launch failed: %s\n", cudaGetErrorString(err));
        return convert_cuda_error(err);
    }

    // Step 4: Compute unique count (last scan value + last flag value)
    // Copy last elements to compute total
    cudaMemcpyAsync(d_unique_count, &d_scan[count - 1], sizeof(int), cudaMemcpyDeviceToDevice, (cudaStream_t)stream);

    int last_flag;
    cudaMemcpyAsync(&last_flag, &d_unique_flags[count - 1], sizeof(int), cudaMemcpyDeviceToHost, (cudaStream_t)stream);

    // Total unique = last_scan + last_flag
    int host_unique_count;
    cudaMemcpyAsync(&host_unique_count, d_unique_count, sizeof(int), cudaMemcpyDeviceToHost, (cudaStream_t)stream);

    // BLACKWELL FIX: Use single cudaDeviceSynchronize instead of multiple cudaStreamSynchronize
    // This forces the driver to retire all work before returning to Go bridge
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        return convert_cuda_error(err);
    }

    host_unique_count += last_flag;
    cudaMemcpyAsync(d_unique_count, &host_unique_count, sizeof(int), cudaMemcpyHostToDevice, (cudaStream_t)stream);

    return CUDA_SUCCESS;
}

// ============================================================================
// CUDA KERNEL: Sovereign Callsign Sanitizer
// ============================================================================
//
// SANITIZATION STRATEGY for RTX 5090 (170 SMs):
//
// 1. IN-PLACE STRING CLEANING:
//    - Each thread processes one WsprSpotC struct
//    - Sanitizes both callsign[16] and reporter[16] fields
//    - Strips: " (quote), ' (apostrophe), \ (backslash)
//    - Preserves: / (forward slash) and all alphanumeric characters
//
// 2. PARALLEL PROCESSING:
//    - 170 SMs × multiple blocks = massive parallelism
//    - Each thread operates independently (no synchronization needed)
//    - 100k spots processed simultaneously across GPU
//
// 3. ALIGNMENT PRESERVATION:
//    - WsprSpotC is 112 bytes (16-byte aligned)
//    - String fields are char arrays (no pointers)
//    - In-place modification preserves struct layout
//
// 4. MODIFICATION TRACKING:
//    - Returns bitmask flags indicating which spots were modified
//    - Go side uses flags to log audit trail to stderr
//
// ============================================================================

// Device helper: Check if character should be stripped
__device__ __forceinline__ bool should_strip_char(char c) {
    return (c == '"' ||   // Double quote
            c == '\'' ||  // Single quote/apostrophe
            c == '\\');   // Backslash
}

// Device helper: Sanitize a single callsign/reporter field (16 bytes)
// Returns true if any modifications were made
__device__ bool sanitize_callsign_field(char* field, int max_len) {
    bool modified = false;
    int write_pos = 0;

    // Scan through field and remove unwanted characters
    for (int read_pos = 0; read_pos < max_len; read_pos++) {
        char c = field[read_pos];

        // Stop at null terminator
        if (c == '\0') {
            // Null-terminate at write position
            field[write_pos] = '\0';

            // Clear remaining bytes to maintain consistent struct layout
            for (int i = write_pos + 1; i < max_len; i++) {
                field[i] = '\0';
            }
            break;
        }

        // Strip unwanted characters
        if (should_strip_char(c)) {
            modified = true;
            continue;  // Skip this character
        }

        // Keep this character
        if (write_pos != read_pos) {
            field[write_pos] = c;
        }
        write_pos++;
    }

    return modified;
}

// Callsign sanitizer kernel optimized for RTX 5090
// Processes callsign and reporter fields in parallel across 170 SMs
__global__ __launch_bounds__(256)
void kernel_sanitize_callsigns(
    WsprSpotC* __restrict__ d_spots,
    int* __restrict__ d_modified_flags,  // Output: 1 if spot was modified, 0 otherwise
    int count
) {
    // Global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Early exit for out-of-bounds threads
    if (idx >= count) {
        return;
    }

    WsprSpotC* spot = &d_spots[idx];
    bool modified = false;

    // Sanitize reporter field (offset 8, 16 bytes)
    if (sanitize_callsign_field(spot->reporter, 16)) {
        modified = true;
    }

    // Sanitize callsign field (offset 48, 16 bytes)
    if (sanitize_callsign_field(spot->callsign, 16)) {
        modified = true;
    }

    // Record if this spot was modified
    d_modified_flags[idx] = modified ? 1 : 0;
}

// Kernel launcher for callsign sanitization
CudaError cuda_kernel_sanitize_callsigns(
    WsprSpotC* d_spots,
    int* d_modified_flags,
    int count,
    CudaStream stream
) {
    if (d_spots == NULL || d_modified_flags == NULL || count <= 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    // Optimal block size for RTX 5090 (170 SMs, high occupancy)
    // 256 threads/block ensures good SM utilization
    int block_size = 256;
    int grid_size = (count + block_size - 1) / block_size;

    // Launch sanitizer kernel on specified stream
    kernel_sanitize_callsigns<<<grid_size, block_size, 0, (cudaStream_t)stream>>>(
        d_spots,
        d_modified_flags,
        count
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "kernel_sanitize_callsigns launch failed: %s\n", cudaGetErrorString(err));
    }

    return convert_cuda_error(err);
}


