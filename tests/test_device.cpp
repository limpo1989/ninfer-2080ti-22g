#include "core/device.h"

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <utility>

namespace {

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int expect_throws_device(int device_id) {
    try {
        ninfer::DeviceContext invalid(device_id);
    } catch (const std::runtime_error&) { return 0; }
    std::cerr << "DeviceContext(" << device_id << ") did not throw\n";
    return 1;
}

int check_context(const ninfer::DeviceContext& ctx, const char* label) {
    int failures = 0;
    if (ctx.stream == nullptr) {
        std::cerr << label << " compute stream is null\n";
        ++failures;
    }
    if (ctx.load_stream == nullptr) {
        std::cerr << label << " load stream is null\n";
        ++failures;
    }
    if (ctx.sm() <= 0) {
        std::cerr << label << " sm is not positive\n";
        ++failures;
    }
    if (ctx.total_vram() == 0) {
        std::cerr << label << " total_vram is zero\n";
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }
    if (count == 0) {
        std::cout << "SKIP: no CUDA devices\n";
        return 77;
    }

    int failures = 0;

    ninfer::DeviceContext ctx(0);
    if (ctx.device != 0) {
        ++failures;
        std::cerr << "ctx.device expected 0, got " << ctx.device << '\n';
    }
    failures += check_context(ctx, "ctx");
    ctx.synchronize();

    const cudaStream_t original_stream = ctx.stream;
    ninfer::DeviceContext moved(std::move(ctx));
    if (ctx.stream != nullptr || ctx.load_stream != nullptr) {
        ++failures;
        std::cerr << "move construction did not null source streams\n";
    }
    if (moved.stream != original_stream) {
        ++failures;
        std::cerr << "move construction did not transfer compute stream\n";
    }
    failures += check_context(moved, "moved");

    // synchronize() must make asynchronous stream results visible before returning,
    // under both the default blocking policy and the opt-in spin policy.
    constexpr std::size_t bytes = 4096;
    unsigned char* host = nullptr;
    unsigned char* gpu  = nullptr;
    CUDA_CHECK(cudaMallocHost(&host, bytes));
    CUDA_CHECK(cudaMalloc(&gpu, bytes));
    for (const unsigned char value : {0x5a, 0xa5}) {
        CUDA_CHECK(cudaMemsetAsync(gpu, value, bytes, moved.stream));
        CUDA_CHECK(cudaMemcpyAsync(host, gpu, bytes, cudaMemcpyDeviceToHost, moved.stream));
        moved.synchronize();
        for (std::size_t i = 0; i < bytes; ++i) {
            if (host[i] != value) {
                ++failures;
                std::cerr << "stream result was not visible after synchronize\n";
                break;
            }
        }
    }
    CUDA_CHECK(cudaFree(gpu));
    CUDA_CHECK(cudaFreeHost(host));

    failures += expect_throws_device(count);

    ninfer::CudaEventTimer timer(moved);
    timer.start();
    moved.synchronize();
    const float elapsed_ms = timer.stop_ms();
    if (elapsed_ms < 0.0f) {
        ++failures;
        std::cerr << "timer elapsed time was negative\n";
    }

    return failures == 0 ? 0 : fail("device test failed");
}
