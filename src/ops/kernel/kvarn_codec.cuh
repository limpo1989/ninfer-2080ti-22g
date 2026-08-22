#pragma once

// ninfer::ops - KVarN record addressing, bit packing, and dequantization.
//
// One record covers exactly one paged-KV page of one KV head. Field offsets mirror
// ninfer::kvarn_record_layout(); the wrapper asserts the two agree before any launch.

#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops {

constexpr std::int64_t kvarn_align_up(std::int64_t value, std::int64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

template <int HeadDimValue, int GroupValue, int KeyBitsValue, int ValueBitsValue>
struct KvarnRecordSpec {
    static_assert(GroupValue == kPagedKVPageSize, "A KVarN record spans exactly one KV page");
    static_assert(8 % KeyBitsValue == 0 && 8 % ValueBitsValue == 0,
                  "KVarN codes must tile a byte evenly");

    static constexpr int HeadDim   = HeadDimValue;
    static constexpr int Group     = GroupValue;
    static constexpr int KeyBits   = KeyBitsValue;
    static constexpr int ValueBits = ValueBitsValue;
    static constexpr int KeyPack   = 8 / KeyBits;
    static constexpr int ValuePack = 8 / ValueBits;
    static constexpr int KeyMask   = (1 << KeyBits) - 1;
    static constexpr int ValueMask = (1 << ValueBits) - 1;

    static constexpr std::int64_t kElements = static_cast<std::int64_t>(HeadDim) * Group;

    static constexpr std::int64_t KPayloadOff   = 0;
    static constexpr std::int64_t KPayloadBytes = kElements * KeyBits / 8;
    static constexpr std::int64_t KScaleOff     = KPayloadOff + KPayloadBytes;   // [HeadDim] fp16
    static constexpr std::int64_t KZeroOff      = KScaleOff + HeadDim * 2;       // [HeadDim] fp16
    static constexpr std::int64_t KTokenOff     = KZeroOff + HeadDim * 2;        // [Group]   fp16
    static constexpr std::int64_t VPayloadOff   = KTokenOff + Group * 2;
    static constexpr std::int64_t VPayloadBytes = kElements * ValueBits / 8;
    static constexpr std::int64_t VChannelOff   = VPayloadOff + VPayloadBytes;   // [HeadDim] fp16
    static constexpr std::int64_t VScaleOff     = VChannelOff + HeadDim * 2;     // [Group]   fp16
    static constexpr std::int64_t VZeroOff      = VScaleOff + Group * 2;         // [Group]   fp16

    static constexpr std::int64_t RawBytes    = VZeroOff + Group * 2;
    // The per-token slot is the KV plane's leading extent, so the record must divide by Group.
    static constexpr std::int64_t RecordBytes = kvarn_align_up(RawBytes, Group % 8 == 0 ? Group : 8 * Group);
    static constexpr std::int64_t SlotBytes   = RecordBytes / Group;

    static_assert(RecordBytes % Group == 0);
};

using KvarnK4V2 = KvarnRecordSpec<256, kPagedKVPageSize, 4, 2>;
using KvarnK4V4 = KvarnRecordSpec<256, kPagedKVPageSize, 4, 4>;

// Byte offset of the record owning (physical_page, kv_head) inside one layer's record plane.
template <typename Spec, int KVHeads>
__device__ __forceinline__ std::int64_t kvarn_record_offset(std::int32_t physical_page,
                                                            std::int32_t kv_head) {
    return Spec::RecordBytes * (static_cast<std::int64_t>(kv_head) +
                                static_cast<std::int64_t>(KVHeads) * physical_page);
}

__device__ __forceinline__ float kvarn_load_f16(const std::uint8_t* record, std::int64_t offset,
                                                int index) {
    const __half* values = reinterpret_cast<const __half*>(record + offset);
    return __half2float(values[index]);
}

__device__ __forceinline__ void kvarn_store_f16(std::uint8_t* record, std::int64_t offset,
                                                int index, float value) {
    reinterpret_cast<__half*>(record + offset)[index] = __float2half(value);
}

// Codes are packed low-order value first: value i lives in bits [(i % pack) * bits, ...) of
// byte i / pack.
template <int Bits>
__device__ __forceinline__ int kvarn_unpack(const std::uint8_t* payload, std::int64_t index) {
    constexpr int pack = 8 / Bits;
    constexpr int mask = (1 << Bits) - 1;
    return (payload[index / pack] >> ((index % pack) * Bits)) & mask;
}

// Orthonormal symmetric Walsh-Hadamard transform of one Width-element vector held in shared
// memory, one element per thread. Its own inverse, so the same call rotates Q into the record
// frame and rotates the attention output back out of it.
template <int Width>
__device__ __forceinline__ void kvarn_hadamard_shared(float* values, int lane) {
    static_assert(Width > 0 && (Width & (Width - 1)) == 0);
    __syncthreads();
#pragma unroll
    for (int stride = 1; stride < Width; stride *= 2) {
        const float mine  = values[lane];
        const float other = values[lane ^ stride];
        __syncthreads();
        values[lane] = (lane & stride) != 0 ? other - mine : mine + other;
        __syncthreads();
    }
    values[lane] *= rsqrtf(static_cast<float>(Width));
    __syncthreads();
}

} // namespace ninfer::ops
