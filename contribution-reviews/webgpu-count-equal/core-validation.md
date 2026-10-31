> Historical evidence for the original `e3ded50` implementation, before the small-input fast path. See [current validation and benchmarks](small-input-optimization/README.md).

# WebGPU COUNT_EQUAL 本地实现与验证

用户本轮已授权修改个人 fork。实现现已提交并推送到 LeON-Nie-code/llama.cpp 的 preview/webgpu-count-equal 分支，提交 e3ded50f4b2dea3e50d5f6f3f7ce594f5047096f。没有向官方仓库创建 PR、issue 或发表评论。

- 工作仓库：`/tmp/llama-count-equal-work`
- 本地分支：`preview/webgpu-count-equal`
- origin：`https://github.com/LeON-Nie-code/llama.cpp.git`
- upstream：`https://github.com/ggml-org/llama.cpp.git`
- 基线：`0ec4027a2a0183de87c007069adddc06e665f7de`
- 本次分支独立于 ARANGE PR，没有包含 ARANGE 改动。
- 网页 diff：https://github.com/LeON-Nie-code/llama.cpp/compare/0ec4027a2a0183de87c007069adddc06e665f7de...preview/webgpu-count-equal
- 差异：6 文件，217 行增加、3 行删除；包含新 shader。GitHub compare API 已核实只比基线多 1 个提交。

## 实现与适用范围

支持相同形状、连续 I32 输入，返回 I64 标量。连续 view 的偏移、相同输入、重叠输入复用后端现有 merged binding helper。先用现有 memset pipeline 清零两个 32 位字，再进行整数比较、workgroup 归约和一次 atomicAdd/workgroup。两次 dispatch 都经由 build_multi 编码，不单独发起 CPU 端清零。

支持空输入：只清零输出。非连续输入和非 I32 输入不宣称支持。除现有设备 storage-binding 限制外，输入元素数限制为 UINT32_MAX / sizeof(int32_t)，从而让比较索引及计数保持在 32 位范围内，I64 高位保持零。使用固定 256 线程 workgroup；这是 WebGPU 保证支持的工作组大小，与硬件 warp 大小无关。

真实调用位于 ggml/src/ggml-opt.cpp 的准确率统计。没有宣称生成模型推理整体加速，也没有沿用 ARANGE 的 TTS 数据。

## 本机结果

NVIDIA L4，驱动 550.163.01；native Dawn（复用此前安装的 v20260908.214631）；Release；GCC 14.2；CMake 3.31.6。

- 原始上游：两个 COUNT_EQUAL 用例都显示 not supported（baseline-test.log）。
- 修改后 WebGPU：23/23，与 CPU reference 比较（webgpu-test.log）。
- CPU：23/23（cpu-test.log）。
- ARGMAX / SUM / SUM_ROWS 回归：27/27（reduction-regression.log）。
- 精确 I64 / 重复执行 / 2D dispatch：全部通过（exact-replay-test.log）。该程序直接读取 int64_t，不依赖框架的 float 比较，能检查 2^24 以上奇数计数和非零高位残留。
- 独立验证程序对 N=0、1、257、3084、16777217、67108872 分别运行 4 个阶段；每阶段两个 COUNT_EQUAL 节点，覆盖全匹配、全不匹配、每三项匹配以及输入更新后的重新执行。输出每次预填 -1，并检查两个输入的前缀 sentinel。
- 3084 个元素使用 [257,3,2,2] 四维输入；大尺寸超过单轴 65535 个 workgroup 的上限，走二维网格。另验证相同输入和部分重叠 view，各连续运行三次。
- 独立能力检查确认拒绝 F32、非连续输入、超限大小；拒绝大输入发生于分配前。
- clang-format 22.1.0、EditorConfig 3.0.3、git diff --check 通过。
- 浏览器、其他 GPU、反向传播训练未测试。完整模型推理/前向验证结果另见 fullmodel 目录；没有声称完整项目 CI 已全部运行。

性能为仓库 perf 工具的批量算子测量，包含清零与归约，不等于一次 CPU 同步调用的端到端延迟，也不代表模型整体加速：

| 元素数 | WebGPU us/run | CPU us/run |
|---:|---:|---:|
| 256 | 14.60 | 1.52 |
| 4096 | 14.32 | 4.81 |
| 65536 | 14.31 | 57.03 |
| 1048576 | 23.12 | 927.73 |

CPU 行仅作本机参考，不作为端到端加速比例；小输入 CPU 本身很便宜，是否值得留在 GPU 取决于图中数据位置和调度开销。原始输出见 webgpu-perf.log / cpu-perf.log。

## 构建与复现

```sh
cmake -S /tmp/llama-count-equal-work -B /tmp/llama-count-equal-build \
  -DGGML_WEBGPU=ON -DCMAKE_PREFIX_PATH=/tmp/llama-dawn \
  -DDawn_DIR=/tmp/llama-dawn/lib64/cmake/Dawn \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/llama-count-equal-build --target test-backend-ops -j 6
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json XDG_RUNTIME_DIR=/tmp \
  /tmp/llama-count-equal-build/bin/test-backend-ops test -b WebGPU -o COUNT_EQUAL
/tmp/llama-count-equal-build/bin/test-backend-ops test -b CPU -o COUNT_EQUAL
c++ -O2 -std=c++17 validate-count-equal.cpp \
  -I/tmp/llama-count-equal-work/ggml/include \
  -L/tmp/llama-count-equal-build/bin -Wl,-rpath,/tmp/llama-count-equal-build/bin \
  -lggml-webgpu -lggml-base -o /tmp/validate-count-equal
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json XDG_RUNTIME_DIR=/tmp \
  /tmp/validate-count-equal
```

本机默认沙箱不能直接访问 NVIDIA 驱动。GPU 验证均在获准访问显卡的执行环境中完成，日志显示 NVIDIA L4，并非 llvmpipe。

## 提交前仍需由你处理

实现已做过本地自审，没有发现当前改动的阻断性问题。仍需你逐行理解、人工审阅代码并用自己的文字撰写最终正文；本地自动检查不能代替维护者 review。

1. 仓库 CONTRIBUTING.md 要求新贡献者同时只开一个 PR。此前 ARANGE #31455 当前仍 open，建议先等其处理。
2. 仓库限制 AI 生成的公开帖子、PR 正文，要求 AI 使用披露；pr-body-preview.md 是按你的要求准备的私下审阅参考，不应直接自动发布。没有替你勾选贡献者协议或声称你已完成人工审阅。
3. 当前未找到专门要求 WebGPU COUNT_EQUAL 的 maintainer issue。旧 WebGPU 总跟踪 issue #7773 作者为 ggerganov，author_association=MEMBER，但已关闭，不能拿它冒充本项功能的明确认可。代码正确和测试通过不保证维护者愿意接收该选题；没有替你开新 issue 或联系维护者。
4. 发现上游 CPU COUNT_EQUAL 高维行索引公式可疑，初版 [257,3,2,2] 对照测试在 CPU 对 CPU 时亦失败。因此提交用例限制为可靠的二维形状，高维 GPU 结果用独立闭式计数验证。CPU 实现未改动，该问题不混入此 PR；也未公开报告。

PR 结构参考你之前的 #31455，以及已合并的 WebGPU PR #30914、#31162，强调范围、实测数据和明确的支持限制。参考内容原始记录见 merged-pr-examples.json。
