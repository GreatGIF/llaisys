# 项目1：优化CPU推理

主要是优化了linear算子的优化推理，参考了pytorch中的策略分发，并且用到了openmp，openblas，MKL，OneDNN四个库。

在xmake编译过程中通过 `xmake f --openmp=y/n --cpu-blas=y/n --cpu-mkl=y/n --cpu-onednn=y/n` 启用支持。

**优先级从低到高有以下实现：**

1. **NAIVE_SMALL**: 单线程朴素实现，用于极小规模运算以减少线程调度开销。
2. **OMP_PARALLEL**: 朴素的 OpenMP 并行实现。
3. **OMP_TILED**: 带有分块优化（Tiling）的 OpenMP 并行实现。
4. **BLAS_GEMM**: 使用通用的 OpenBLAS（openblas基本只支持fp32不支持fp16和bf16，所以只实现了fp32）。
5. **MKL_GEMM**: 使用 Intel MKL (Math Kernel Library) 的 `cblas_sgemm`、`cblas_gemm_bf16bf16f32` 等接口。
6. **ONEDNN_GEMM**: 优先使用 oneDNN 库（Intel 推出的深度学习抽象库）。

> 只有在 M, N, K 小的时候 omp 多线程启动有开销，会启用 naive，其余根据环境选择优先级最高的加速。其他算子的 CPU 实现也都用 openmp 加速了一下。

---

# 项目2：在LLAISYS中集成CUDA

通过cuda支持了nvidia和沐曦的GPU。沐曦的sdk通过对cuda的api起别名来实现cuda的支持，因此不用对cuda代码进行什么修改，只需要在xmake中使用沐曦的mxcc编译器即可。使用 `xmake f --nv-gpu=y/n --mx-gpu=y/n` 来启用支持。沐曦GPU也是使用--device nvidia

**踩坑记录：**
xmake检测到 .cu 文件会自动去寻找 cuda sdk 和 nvcc，但是沐曦的环境显然并没有。所以一开始选择的是偷懒将 .cu 文件复制改成 .cpp 文件避免检测 nvcc，但还是太不美观，所以花了一天研究怎么复用 .cu 文件：
- 首先需要先用 xmake 内置的 `sourcekind`，将 .cu 文件伪装成 cxx。
- 然后用 xmake 内置的 toolchain 添加 mxcc。
- xmake 会自动添加很多编译选项，而一些编译选项 mxcc 又不支持，但 xmake 又去不掉。所以得写一个包装脚本，xmake 输出到脚本过滤一遍然后再输出给 mxcc。

**项目体验：**
xmake 相比 cmake 的优势在于易用性以及内置包管理器，但是其易用性来自于定制化和包装，但也正是这定制化与包装，导致灵活性不如 cmake，为了实现 cmake 简单就能实现的一些功能要用到一些奇淫巧计。

**核心算子：**
主要介绍采样算子，其他算子要不调库要不是简单的 Element-wise 实现即可。
- **Top-K**: 两阶段流水 stage1 先做分块 top-k 候选筛选，stage2 再做归并 + 采样。
- **Top-P**: 先排序再做累计概率截断采样，参考了 FasterTransformer 和 TensorRT-LLM。

---

# 项目3：构建 AI 聊天机器人

启动安装 fastapi，uvicorn，requests 三个库。

### 服务端启动 (`python server.py`)
| 参数 | 说明 |
| :--- | :--- |
| `--model MODEL` | 本地模型路径或 Hugging Face model ID |
| `--backend {pytorch,llaisys}` | 推理后端 (默认: pytorch) |
| `--device {cpu,nvidia,mx}` | 使用平台 (默认: cpu) |
| `--temperature TEMPERATURE` | 采样温度 (默认: 0.8) |
| `--top-p TOP_P` | Top-p 采样参数 (默认: 0.8) |
| `--top-k TOP_K` | Top-k 采样参数 (默认: 50) |
| `--seed SEED` | 采样种子 (默认: 0, 随机) |
| `--host HOST` | 服务器 host (默认: 0.0.0.0) |
| `--port PORT` | 服务器端口 (默认: 8000) |
| `--workers WORKERS` | worker 数量 (默认: 1) |

### 客户端交互 (`python client.py`)
| 参数 | 说明 |
| :--- | :--- |
| `--base-url BASE_URL` | 服务器基础 URL |
| `--message MESSAGE` | 要发送的消息 |
| `--temperature` | 采样温度 |
| `--top-p` | Top-p 参数 |
| `--top-k` | Top-k 参数 |
| `--max-tokens` | 最大生成长度 |
| `--seed SEED` | 采样种子 |
| `--mode {health,models,stream,non-stream,interactive,all}` | 操作模式 |
