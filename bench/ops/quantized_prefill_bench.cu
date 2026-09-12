#include "ninfer/ops/quantized_prefill.h"
#include "quantized_weight.cuh"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/attn_input_proj.h"

#include <cstdlib>
#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    using namespace ninfer;
    if (argc != 3) {
        std::fprintf(stderr, "usage: ninfer_quantized_prefill_bench q4|q5|q5gdn|gdn-input|attn-input TOKENS\n");
        return 2;
    }
    const std::string mode(argv[1]);
    const int requested_tokens=std::atoi(argv[2]);
    if ((mode=="gdn-input" || mode=="attn-input") && requested_tokens>=256) {
        const bool gdn=mode=="gdn-input";
        const int t=requested_tokens, k=5120;
        DeviceContext device(0);
        auto qk=bench::make_row_split_weight(QType::Q4G64_F16S,gdn?4096:7168,k,k);
        auto vz=bench::make_row_split_weight(QType::Q5G64_F16S,gdn?12288:7168,k,k);
        auto input=bench::make_bf16(std::size_t(k)*t);
        Tensor x(input.p,DType::BF16,{k,t});
        DeviceBuffer a(std::size_t(gdn?10240:6144)*t*2), b(std::size_t(6144)*t*2),
                     c(std::size_t(1024)*t*2),d(std::size_t(1024)*t*2),flush(256U<<20);
        Tensor y(a.p,DType::BF16,{gdn?10240:6144,t}),z(b.p,DType::BF16,{6144,t}),
               key(c.p,DType::BF16,{1024,t}),value(d.p,DType::BF16,{1024,t});
        ops::QuantizedPrefillContext context;
        if (!context.enabled()) { return 2; }
        WorkspaceArena workspace(gdn?context.gdn_workspace_capacity_bytes(t)
                                    :context.attention_workspace_capacity_bytes(t));
        for (int candidate : {0,1}) {
            const auto result=bench::measure_cold_launch([&](cudaStream_t stream) {
                if (gdn) {
                    if(candidate) context.gdn_input_proj(x,qk.weight,vz.weight,y,z,workspace,stream);
                    else ops::gdn_input_proj(x,qk.weight,vz.weight,y,z,stream);
                } else {
                    if(candidate) context.attention_input_proj(x,qk.weight,vz.weight,y,z,key,value,workspace,stream);
                    else ops::attn_input_proj(x,qk.weight,vz.weight,y,z,key,value,stream);
                }
            },flush,device.stream,5,20);
            std::printf("%s T=%d route=%s median_us=%.3f\n",argv[1],t,
                        candidate?"bounded_fp32":"native",result.median_us);
        }
        return 0;
    }
    const bool swiglu = std::string(argv[1]) == "q4";
    const int tokens = std::atoi(argv[2]);
    const bool gdn = std::string(argv[1]) == "q5gdn";
    if ((std::string(argv[1]) != "q5" && !swiglu && !gdn) || tokens < 256) { return 2; }
    const int n = swiglu ? 34816 : 5120, k = swiglu ? 5120 : (gdn ? 6144 : 17408);
    const int output_rows = swiglu ? n / 2 : n;
    const QType type = swiglu ? QType::Q4G64_F16S : QType::Q5G64_F16S;
    DeviceContext device(0);
    auto weight = bench::make_row_split_weight(type, n, k, k);
    auto input = bench::make_bf16(std::size_t(k) * tokens);
    auto output = bench::make_bf16(std::size_t(output_rows) * tokens);
    DeviceBuffer flush(256U << 20);
    Tensor x(input.p, DType::BF16, {k, tokens});
    Tensor y(output.p, DType::BF16, {output_rows, tokens});
    const auto bytes = swiglu
        ? ops::linear_swiglu_workspace_capacity_bytes(type, n, k, ops::LinearPolicy::A16Only, tokens, tokens)
        : ops::linear_add_workspace_capacity_bytes(type, n, k, tokens, tokens);
    WorkspaceArena workspace(std::max(bytes,
        ops::QuantizedPrefillContext::workspace_capacity_bytes(type,n,k,tokens)));
    ops::QuantizedPrefillContext control;
    if (!ops::QuantizedPrefillContext::enabled()) { return 2; }
    for (int alternative : {0, 1}) {
        const auto result = bench::measure_cold_launch([&](cudaStream_t stream) {
            if (alternative) { control.run(x, weight.weight, y, workspace, stream); }
            else if (swiglu) { ops::linear_swiglu(x, weight.weight, y, workspace, stream); }
            else { ops::linear_add(x, weight.weight, y, workspace, stream); }
        }, flush, device.stream, 10, 30);
        std::printf("%s T=%d route=%s median_us=%.3f\n", argv[1], tokens,
                    alternative ? (swiglu ? "bounded_fp16" : "bounded_fp32") : "native", result.median_us);
    }
}
