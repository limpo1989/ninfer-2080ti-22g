#include "ops/linear_add/linear_add_test_common.h"
#include "ops/linear_swiglu/linear_swiglu_test_common.h"
#include "ninfer/ops/quantized_prefill.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "core/device.h"
#include "core/decode_graph.h"
#include <algorithm>
#include <array>
#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
    const bool q4_only = argc == 2 && std::string_view(argv[1]) == "--q4-only";
    if (argc > 1 && !q4_only) { return 2; }
    using namespace ninfer;
    if (!test::linear_add::cuda_available()) { return 77; }
    try {
        if (!ops::QuantizedPrefillContext::enabled()) {
            throw std::runtime_error("test requires NINFER_QUANTIZED_PREFILL=1");
        }
        DeviceContext device(0);
        ops::QuantizedPrefillContext context;
        int failures = 0;
        const auto run = [&](const Tensor& x, const Weight& w, Tensor& y) {
            const bool alternative = ops::QuantizedPrefillContext::admits(w.qtype, w.n, w.k, x.ne[1]);
            const bool swiglu = w.qtype == QType::Q4G64_F16S;
            const auto bytes = alternative
                ? ops::QuantizedPrefillContext::workspace_capacity_bytes(w.qtype, w.n, w.k, x.ne[1])
                : (swiglu ? ops::linear_swiglu_workspace_capacity_bytes(w.qtype,w.n,w.k,x.ne[1],x.ne[1])
                           : ops::linear_add_workspace_capacity_bytes(w.qtype,w.n,w.k,x.ne[1],x.ne[1]));
            WorkspaceArena workspace(std::max<std::size_t>(bytes,256));
            const auto launch = [&] {
                if (alternative) { context.run(x,w,y,workspace,device.stream); }
                else if (swiglu) { ops::linear_swiglu(x,w,y,workspace,device.stream); }
                else { ops::linear_add(x,w,y,workspace,device.stream); }
            };
            if (x.ne[1] == 257) {
                DecodeGraphDefinition definition;
                definition.capture(device.stream, launch);
                DecodeGraphExecutable graph;
                graph.instantiate(definition);
                graph.launch(device.stream);
                device.synchronize();
            } else { launch(); device.synchronize(); }
            if (workspace.used() != 0 || workspace.peak_used() != bytes) {
                throw std::runtime_error("quantized prefill workspace extent mismatch");
            }
        };
        constexpr std::array<int,1> starts{256};
        constexpr std::array<int,2> interiors{1024,2048};
        if (!q4_only) for (int k : {6144,17408}) {
            failures += test::linear_add::run_shape("Q5 bounded prefill", test::linear_add::WeightFormat::Q5G64F16S,
                {5120,k,409U,starts,interiors},
                [&](const Tensor& x, const Weight& w, Tensor& y, cudaStream_t) { run(x,w,y); });
        }
        constexpr std::array<int,6> tokens{1,255,256,257,1024,2048};
        failures += test::linear_swiglu::run_profile("Q4 bounded prefill",
            {QType::Q4G64_F16S,34816,5120,17408,1401U,test::linear_swiglu::ActivationCompute::A16},tokens,
            [&](const Tensor& x, const Weight& w, Tensor& y, cudaStream_t) { run(x,w,y); });
        std::cout << (failures ? "FAIL" : "OK") << " bounded quantized prefill (direct + graph)\n";
        return failures ? 1 : 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
