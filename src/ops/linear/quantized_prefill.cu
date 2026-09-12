#include "ninfer/ops/quantized_prefill.h"
#include "core/device.h"
#include "core/layout.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/attn_input_proj.h"
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#if defined(NINFER_SM75)
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#endif

namespace ninfer::ops {
namespace {
constexpr std::size_t kDequantBytes = 128U << 20;
constexpr std::size_t kGemmBytes = 8U << 20;
int slice_rows(int rows, int k) {
    return static_cast<int>(std::min<std::size_t>(rows, kDequantBytes / (std::size_t(k) * 2))) / 64 * 64;
}
struct Scratch { DeviceSpan weight, input, product, gemm, input_scale; };
template<class Arena>
Scratch allocate(Arena& arena, int rows, int k, int tokens, bool half_product) {
    return {arena.alloc_bytes(std::size_t(slice_rows(rows, k)) * k * 2),
            arena.alloc_bytes(std::size_t(k) * tokens * 2),
            arena.alloc_bytes(std::size_t(rows) * tokens * (half_product ? 2 : 4)),
            arena.alloc_bytes(kGemmBytes), arena.alloc_bytes(std::size_t(tokens) * sizeof(float))};
}
#if defined(NINFER_SM75)
inline void prefill_cublas_check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("prefill cuBLAS status " + std::to_string(static_cast<int>(status)));
    }
}

// A power-of-two scale per column preserves the useful BF16 exponent range in FP16 GEMM.
static __global__ void prefill_input_half(const __nv_bfloat16* input, __half* output,
                                         float* scales, int k) {
    const int token = blockIdx.x, tid = threadIdx.x;
    const std::size_t base = std::size_t(token) * k;
    float peak = 0;
    for (int i = tid; i < k; i += 256) { peak = fmaxf(peak, fabsf(__bfloat162float(input[base+i]))); }
    for (int d = 16; d; d >>= 1) { peak = fmaxf(peak, __shfl_down_sync(0xffffffffu, peak, d)); }
    __shared__ float maxima[8];
    if ((tid & 31) == 0) { maxima[tid / 32] = peak; }
    __syncthreads();
    if (tid < 32) {
        peak = tid < 8 ? maxima[tid] : 0;
        for (int d = 16; d; d >>= 1) { peak = fmaxf(peak, __shfl_down_sync(0xffffffffu, peak, d)); }
        if (tid == 0) {
            const unsigned exponent = __float_as_uint(peak) & 0x7f800000u;
            const float scale = peak == 0 ? 1 : (exponent ? __uint_as_float(exponent) : peak);
            maxima[0] = scale;
            scales[token] = scale;
        }
    }
    __syncthreads();
    const float scale = maxima[0];
    for (int i = tid; i < k; i += 256) {
        output[base+i] = __float2half(__bfloat162float(input[base+i]) / scale);
    }
}

template <bool Q5>
static __global__ void prefill_dequant_half(const std::uint8_t* codes,
                                           const std::uint8_t* high,
                                           const std::uint16_t* scales, __half2* output,
                                           int k, int begin_row, std::size_t pairs) {
    const auto pair = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair >= pairs) { return; }
    const auto global_pair = pair + static_cast<std::size_t>(begin_row) * (k / 2);
    const auto group = global_pair / 32;
    const int lane = global_pair % 32;
    const int packed = codes[global_pair];
    int a = packed & 15, b = packed >> 4;
    if constexpr (Q5) {
        const int bits = high[group * 8 + lane / 4];
        a |= ((bits >> (2 * (lane % 4))) & 1) << 4;
        b |= ((bits >> (2 * (lane % 4) + 1)) & 1) << 4;
        a = (a ^ 16) - 16;
        b = (b ^ 16) - 16;
    } else {
        a = (a ^ 8) - 8;
        b = (b ^ 8) - 8;
    }
    const float scale = __half2float(__ushort_as_half(scales[group]));
    output[pair] = __floats2half2_rn(a * scale, b * scale);
}

template <bool SwiGlu, bool HalfProduct = false>
static __global__ void prefill_epilogue(const void* product,
                                               __nv_bfloat16* output, const float* scales, int rows,
                                               std::size_t elements) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= elements) { return; }
    const auto read_product = [&](std::size_t j) {
        if constexpr (HalfProduct) { return __half2float(static_cast<const __half*>(product)[j]); }
        else { return static_cast<const float*>(product)[j]; }
    };
    float value;
    if constexpr (SwiGlu) {
        const auto token = i / rows;
        const auto row = i % rows;
        const float gate = read_product(token * (2 * rows) + row) * scales[token];
        const float up = read_product(token * (2 * rows) + rows + row) * scales[token];
        value = (gate / (1.0f + expf(-gate))) * up;
    } else {
        value = read_product(i) * scales[i / rows] + __bfloat162float(output[i]);
    }
    output[i] = __float2bfloat16(value);
}

void project(cublasHandle_t handle, const Weight& w, int tokens, const Scratch& scratch,
             bool half_product, cudaStream_t stream) {
    const int rows = w.n, k = w.k;
    const bool q4 = w.qtype == QType::Q4G64_F16S;
    const float one = 1, zero = 0;
    const int slice = slice_rows(rows, k);
    for (int begin = 0; begin < rows; begin += slice) {
        const int count = std::min(slice, rows - begin);
        const auto pairs = std::size_t(count) * k / 2;
        const auto dequant = q4 ? prefill_dequant_half<false> : prefill_dequant_half<true>;
        dequant<<<(pairs + 255) / 256, 256, 0, stream>>>(
            static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint8_t*>(w.qhigh),
            static_cast<const std::uint16_t*>(w.scales), static_cast<__half2*>(scratch.weight.data),
            k, begin, pairs);
        if (half_product) {
            const __half one16 = __float2half(1), zero16 = __float2half(0);
            prefill_cublas_check(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                count, tokens, k, &one16, scratch.weight.data, CUDA_R_16F, k,
                scratch.input.data, CUDA_R_16F, k, &zero16,
                static_cast<__half*>(scratch.product.data) + begin, CUDA_R_16F, rows,
                CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        } else {
            prefill_cublas_check(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                count, tokens, k, &one, scratch.weight.data, CUDA_R_16F, k,
                scratch.input.data, CUDA_R_16F, k, &zero,
                static_cast<float*>(scratch.product.data) + begin, CUDA_R_32F, rows,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
    }
}

static __global__ void projection_store(const float* product, const float* scale,
    __nv_bfloat16* output, int source_rows, int source_begin, int destination_rows,
    int destination_begin, int count, std::size_t elements) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= elements) { return; }
    const auto t = i / count, r = i % count;
    output[t * destination_rows + destination_begin + r] =
        __float2bfloat16(product[t * source_rows + source_begin + r] * scale[t]);
}
void store_projection(const Scratch& scratch, int source_rows, int source_begin,
                      Tensor& output, int begin, int count, cudaStream_t stream) {
    const auto elements = std::size_t(count) * output.ne[1];
    projection_store<<<(elements+255)/256, 256, 0, stream>>>(
        static_cast<const float*>(scratch.product.data), static_cast<const float*>(scratch.input_scale.data),
        static_cast<__nv_bfloat16*>(output.data), source_rows, source_begin, output.ne[0], begin,
        count, elements);
}
#endif
} // namespace

struct QuantizedPrefillContext::Impl {
#if defined(NINFER_SM75)
    cublasHandle_t handle = nullptr;
    Impl() {
        prefill_cublas_check(cublasCreate(&handle));
        try {
            prefill_cublas_check(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
            prefill_cublas_check(cublasSetMathMode(handle, static_cast<cublasMath_t>(
                CUBLAS_TENSOR_OP_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
        } catch (...) { cublasDestroy(handle); throw; }
    }
    ~Impl() { cublasDestroy(handle); }
#endif
};

bool QuantizedPrefillContext::enabled() {
#if defined(NINFER_SM75)
    static const bool value = [] {
        const char* env = std::getenv("NINFER_QUANTIZED_PREFILL");
        return env == nullptr || std::string_view(env) == "1";
    }();
    return value;
#else
    return false;
#endif
}
QuantizedPrefillContext::QuantizedPrefillContext() {
    if (enabled()) { impl_ = std::make_unique<Impl>(); }
}
QuantizedPrefillContext::~QuantizedPrefillContext() = default;
bool QuantizedPrefillContext::admits(QType type, int rows, int k, int tokens) {
    return enabled() && tokens >= 256 &&
        ((type == QType::Q4G64_F16S && rows == 34816 && k == 5120) ||
         (type == QType::Q5G64_F16S && rows == 5120 && (k == 6144 || k == 17408)));
}
std::size_t QuantizedPrefillContext::workspace_capacity_bytes(QType type, int rows, int k,
                                                            int max_tokens) {
    if (!admits(type, rows, k, max_tokens)) { return 0; }
    WorkspaceLayoutBuilder layout;
    (void)allocate(layout, rows, k, max_tokens, type == QType::Q4G64_F16S);
    return layout.peak_bytes(1);
}
void QuantizedPrefillContext::run(const Tensor& x, const Weight& w, Tensor& output,
                                 WorkspaceArena& workspace, cudaStream_t stream) {
#if defined(NINFER_SM75)
    const int tokens = x.ne[1], rows = w.n, k = w.k;
    const bool swiglu = w.qtype == QType::Q4G64_F16S;
    const int output_rows = swiglu ? rows / 2 : rows;
    if (!impl_ || !admits(w.qtype, rows, k, tokens) || x.ne[0] != k ||
        w.padded_shape[1] != k || output.ne[0] != output_rows || output.ne[1] != tokens ||
        x.dtype != DType::BF16 || output.dtype != DType::BF16 ||
        !x.is_contiguous() || !output.is_contiguous()) {
        throw std::invalid_argument("quantized prefill: unsupported input");
    }
    auto scope = workspace.scope();
    const bool half_product = swiglu;
    const Scratch scratch = allocate(workspace, rows, k, tokens, half_product);
    auto handle = impl_->handle;
    prefill_cublas_check(cublasSetStream(handle, stream));
    prefill_cublas_check(cublasSetWorkspace(handle, scratch.gemm.data, scratch.gemm.bytes));
    prefill_input_half<<<tokens, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<__half*>(scratch.input.data),
        static_cast<float*>(scratch.input_scale.data), k);
    project(handle, w, tokens, scratch, half_product, stream);
    const auto elements = std::size_t(output_rows) * tokens;
    const auto epilogue = swiglu ? prefill_epilogue<true, true> : prefill_epilogue<false>;
    epilogue<<<(elements + 255) / 256, 256, 0, stream>>>(
        scratch.product.data, static_cast<__nv_bfloat16*>(output.data),
        static_cast<const float*>(scratch.input_scale.data), output_rows, elements);
    CUDA_CHECK(cudaGetLastError());
#else
    (void)x; (void)w; (void)output; (void)workspace; (void)stream;
    throw std::logic_error("quantized prefill requires SM75");
#endif
}
std::size_t QuantizedPrefillContext::gdn_workspace_capacity_bytes(int tokens) {
    if (!enabled() || tokens < 256) { return 0; }
    WorkspaceLayoutBuilder layout;
    (void)allocate(layout, 12288, 5120, tokens, false);
    return layout.peak_bytes(1);
}
std::size_t QuantizedPrefillContext::attention_workspace_capacity_bytes(int tokens) {
    if (!enabled() || tokens < 256) { return 0; }
    WorkspaceLayoutBuilder layout;
    (void)allocate(layout, 7168, 5120, tokens, false);
    return layout.peak_bytes(1);
}
void QuantizedPrefillContext::gdn_input_proj(const Tensor& x, const Weight& qk, const Weight& vz,
    Tensor& qkv, Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
    if (!enabled() || x.ne[1] < 256) { ops::gdn_input_proj(x, qk, vz, qkv, z, stream); return; }
#if defined(NINFER_SM75)
    const int tokens = x.ne[1];
    if (x.ne[0] != 5120 || x.dtype != DType::BF16 || !x.is_contiguous() ||
        qk.qtype != QType::Q4G64_F16S || qk.n != 4096 || qk.k != 5120 || qk.padded_shape[1] != 5120 ||
        vz.qtype != QType::Q5G64_F16S || vz.n != 12288 || vz.k != 5120 || vz.padded_shape[1] != 5120 ||
        qkv.ne[0] != 10240 || qkv.ne[1] != tokens || qkv.dtype != DType::BF16 || !qkv.is_contiguous() ||
        z.ne[0] != 6144 || z.ne[1] != tokens || z.dtype != DType::BF16 || !z.is_contiguous()) {
        throw std::invalid_argument("bounded GDN input projection shape");
    }
    auto scope = workspace.scope();
    const auto scratch = allocate(workspace, 12288, 5120, tokens, false);
    const auto handle = impl_->handle;
    prefill_cublas_check(cublasSetStream(handle, stream));
    prefill_cublas_check(cublasSetWorkspace(handle, scratch.gemm.data, scratch.gemm.bytes));
    prefill_input_half<<<tokens, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
        static_cast<__half*>(scratch.input.data), static_cast<float*>(scratch.input_scale.data), 5120);
    project(handle, qk, tokens, scratch, false, stream);
    store_projection(scratch, 4096, 0, qkv, 0, 4096, stream);
    project(handle, vz, tokens, scratch, false, stream);
    store_projection(scratch, 12288, 0, qkv, 4096, 6144, stream);
    store_projection(scratch, 12288, 6144, z, 0, 6144, stream);
    CUDA_CHECK(cudaGetLastError());
#endif
}
void QuantizedPrefillContext::attention_input_proj(const Tensor& x, const Weight& qk, const Weight& gv,
    Tensor& q, Tensor& gate, Tensor& k, Tensor& v, WorkspaceArena& workspace, cudaStream_t stream) {
    if (!enabled() || x.ne[1] < 256) { ops::attn_input_proj(x, qk, gv, q, gate, k, v, stream); return; }
#if defined(NINFER_SM75)
    const int tokens = x.ne[1];
    if (x.ne[0] != 5120 || x.dtype != DType::BF16 || !x.is_contiguous() ||
        qk.qtype != QType::Q4G64_F16S || qk.n != 7168 || qk.k != 5120 || qk.padded_shape[1] != 5120 ||
        gv.qtype != QType::Q5G64_F16S || gv.n != 7168 || gv.k != 5120 || gv.padded_shape[1] != 5120) {
        throw std::invalid_argument("bounded attention input projection shape");
    }
    for (const Tensor* output : {&q, &gate, &k, &v}) {
        const int rows = (output == &q || output == &gate)?6144:1024;
        if (output->ne[0] != rows || output->ne[1] != tokens || output->dtype != DType::BF16 ||
            !output->is_contiguous()) { throw std::invalid_argument("bounded attention output shape"); }
    }
    auto scope = workspace.scope();
    const auto scratch = allocate(workspace, 7168, 5120, tokens, false);
    const auto handle = impl_->handle;
    prefill_cublas_check(cublasSetStream(handle, stream));
    prefill_cublas_check(cublasSetWorkspace(handle, scratch.gemm.data, scratch.gemm.bytes));
    prefill_input_half<<<tokens, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
        static_cast<__half*>(scratch.input.data), static_cast<float*>(scratch.input_scale.data), 5120);
    project(handle, qk, tokens, scratch, false, stream);
    store_projection(scratch, 7168, 0, q, 0, 6144, stream);
    store_projection(scratch, 7168, 6144, k, 0, 1024, stream);
    project(handle, gv, tokens, scratch, false, stream);
    store_projection(scratch, 7168, 0, gate, 0, 6144, stream);
    store_projection(scratch, 7168, 6144, v, 0, 1024, stream);
    CUDA_CHECK(cudaGetLastError());
#endif
}
} // namespace ninfer::ops
