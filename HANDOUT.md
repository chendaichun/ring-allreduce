# L6：手写 Ring AllReduce —— 实验指导

> 版本 1.1 · 2026-10-20 · 机器：gpu01（`ssh ustcc`）· 预计用时 10–14 小时（不含选做）
>
> 本文档参照 Stanford CS336 作业 handout 的写法：每一节先给背景和规格，再给带分值的 **Problem**，每个 Problem 写明 **Deliverable**（要交什么）。分值只用来提示工作量和重要性。
>
> **框架已经提供**：构建（`Makefile`）、基准测试主程序（`ring.cu`：计时、正确性校验、表格与 CSV 输出、NCCL 基线）、公共工具（`common.h/.cu`）、画图（`plot.py`）、链路探测骨架（`probe.cu`）和报告模板（`README.md`）。**AllReduce 由你自己写**：`allreduce.cu` 里的 r1–r4（选做 r5）、`probe.cu` 的 (b)–(e)，以及实验报告。

**目录**

1. 实验概览
2. 背景与约定
3. Problem 0：环境与构建
4. Problem 1：先认识你的“链路”
5. Problem 2：读懂框架与 NCCL 基线
6. Problem 3：r1 朴素实现
7. Problem 4：r2 Ring + 全体同步
8. Problem 5：r3 Ring + event 只和邻居同步
9. Problem 6：r4 Ring + 流水线
10. Problem 7：全尺寸测试与绘图
11. Problem 8：α-β 分析
12. Problem 9（选做）：r5 手动 pinned host 中转
13. 选做探索

附录 A 完成标准与报告结构 · 附录 B 常见坑 · 附录 C 参考数据（剧透）

---

## 1 实验概览

### 1.1 这个实验要回答的问题

你已经知道：在“p 条互相独立的链路”这个模型下，Ring AllReduce 是带宽最优的；NCCL 用 channel、8 格 FIFO + head/tail、Simple/LL protocol 来实现它。这个实验让你在一台链路很特殊的机器上亲手实现一遍。gpu01 的 4 张 RTX 5090 之间没有 P2P，GPU 之间的所有数据都要经过 CPU 内存中转。做完之后，你应该能用实测数据和 α-β 模型回答下面三个问题：

1. 在这台机器上，Ring 相对朴素的“汇总到一张卡”，到底赢在哪里、输在哪里？
2. 你的实现比 NCCL 慢多少？差距在 α（固定开销）还是 β（带宽）？
3. r2 → r3 → r4 的每一步改动，分别改变了模型里的哪一项？

### 1.2 你要实现什么

一个单进程、单线程控制 4 张 GPU 的基准测试程序 `ring`，包含下面几个版本：

| 版本 | 做法 | 相对上一版唯一的改动 |
|---|---|---|
| r1 | 朴素：其他卡把整份 buf 拷到 GPU0 → GPU0 一个 kernel 累加 → 结果拷回各卡 | 起点 |
| r2 | Ring（ReduceScatter p−1 步 + AllGather p−1 步），只用 host 端全体同步表达依赖 | 算法：集中式 → 环 |
| r3 | 同样的 Ring，每卡一个 stream，用 cudaEvent 只和邻居同步，中途不做 host 同步 | 同步：全局屏障 → 邻居依赖 |
| r4 | r3 + 流水线：每块切成 K 个小块，tmp 做成 S 槽 FIFO，拷贝与加法分属两个 stream | 粒度：整块 → 小块流水 |
| nccl | 同一程序里调用 `ncclAllReduce`，作为基线 | — |
| r5（选做） | r4 的流水线，但用手动 pinned host 中转（D2H + H2D）代替 `cudaMemcpyPeerAsync` | 传输原语 |

框架已经把版本登记、校验、计时和输出都接好了。`allreduce.cu` 里每个版本现在都是写着 `NOT_IMPLEMENTED(...)` 的空函数，框架运行时会跳过它们，并提示对应的 Problem；你每实现一个版本，它就会自动出现在表格、汇总表和 CSV 里。除此之外，你还要补全链路探测程序 `probe.cu`，并完成实验报告 `README.md`。

### 1.3 规则

- **自己写**：除 nccl 基线外，各版本只能使用 CUDA runtime API（`cudaMemcpyPeerAsync`、`cudaMemcpyAsync`、stream、event）和你自己写的 kernel。不调用 NCCL/MPI 的通信接口，也不引入 thrust/cub 等额外的库。GPU 间拷贝统一用 `cudaMemcpyPeerAsync`（r5 除外），无 P2P 时由驱动自动经主机中转。
- **问题设定**：单进程控制 4 张 GPU，float32 求和，原地（in-place）计算：`buf[g]` 既是输入，也是结果。
- **代码风格**：每个版本一个独立函数，用中文注释写清每一个依赖。每个版本只做它要求的那一个改动，不要顺手优化，否则版本之间的差异无法归因。
- **共享机器**：开始任何测量前先用 `nvidia-smi` 确认 4 张卡空闲，长时间测量前再确认一次。不用 sudo，不动别人的进程。
- **只读目录**：不修改 `~/nccl-tests` 和 `~/pmpp/pmpp-a1-gemm` 里的任何文件，也不往 `pmpp-a1-gemm/.venv` 里装包。

### 1.4 目录与命令行

```
~/pmpp/ring-allreduce/
├── HANDOUT.md        本文档
├── AI_HANDOFF.md     给协助你的 AI 看的交接说明（含实测结果，相当于剧透）
├── Makefile          [提供] 构建与常用命令
├── common.h/.cu      [提供] 公共工具：CK、mod、now_us、sync_devices、init/add kernel、NUMA 工具
├── allreduce.h       [提供] 框架与你的实现之间的接口：Ctx、Params、各版本的函数声明
├── allreduce.cu      [你写] r1–r4（选做 r5）的实现，现在全是 NOT_IMPLEMENTED 的空函数
├── ring.cu           [提供] 基准测试主程序 + NCCL 基线（一般不需要改）
├── probe.cu          [部分] 计时工具和 (a) 已写好，(b)–(e) 由你来写（P1）
├── plot.py           [提供] 画 busbw / time / K 扫描图（P6、P7）
├── README.md         [你写] 实验报告，已按 Problem 编号给好了结构
└── results/          表格、CSV、图
```

常用命令（都在 `~/pmpp/ring-allreduce` 下执行）：

| 命令 | 作用 |
|---|---|
| `make` | 编译 `ring` 和 `probe` |
| `make check` | 小尺寸正确性检查（1K–4M，`--check-reps 3`）；任何已实现版本 `#wrong ≠ 0` 时失败 |
| `make run` | 完整测试 1K–1G → `results/allreduce.txt`、`results/allreduce.csv` |
| `make ksweep` | r4 的 K 扫描（K = 1…32）→ `results/k_sweep.csv` |
| `make venv` / `make plot` | 建 Python 虚拟环境 / 画图 → `results/*.png` |
| `./probe [abcde]` | 链路探测，可以只跑其中几节 |

`ring` 的参数：

```bash
./ring --versions r1,r2,r3,r4,nccl --min 1K --max 1G --k 8 \
       [--slots 8] [--warmup 5] [--iters 20] [--timing sync|batch] [--check-reps 1] \
       [--r5-numa send|recv|off] [--csv results/x.csv]
```

### 1.5 分值与提交物

| Problem | 内容 | 分值 | Deliverable |
|---|---|---:|---|
| P0 `environment` | 环境与构建 | 2 | 环境信息 |
| P1 `link_probe` | 认识链路 | 12 | probe.cu 的 (b)–(e) + 表格 + 简答 |
| P2 `harness` | 读懂框架与 NCCL 基线 | 8 | 运行结果 + 表格 + 简答 |
| P3 `r1_naive` | 朴素实现 | 10 | `ar_r1` + 预测与实测对照 |
| P4 `r2_ring_bsp` | Ring + 全体同步 | 12 | `ar_r2` + 证明 + 简答 |
| P5 `r3_ring_events` | Ring + event | 16 | `ar_r3` + 证明 + 分析 |
| P6 `r4_pipeline` | 流水线 | 18 | `ar_r4` + K/S 扫描 + 分析 |
| P7 `full_benchmark` | 全尺寸测试与绘图 | 6 | results/ 下的表、CSV、图 |
| P8 `alpha_beta` | α-β 分析 | 16 | README 中的分析 |
| P9 `r5_staging`（选做） | 手动 pinned 中转 | +10 | `ar_r5` + 对比 + 分析 |

所有简答、证明和分析都写进 `README.md`，按 Problem 编号组织。建议每完成一个 Problem 就 `git commit` 一次（例如 `P4: r2 ring + barrier`）。

### 1.6 建议的工作顺序

- 顺序：P0 → P1 → P2（读懂框架，先只跑 nccl）→ P3 → P4 → P5 → P6 → P7 → P8 →（P9）。
- 每写完一个版本，先跑 `make check`，或者只测这一个版本：`./ring --versions rX --max 64K --warmup 1 --iters 3 --check-reps 3`。正确之后再跑大尺寸。
- 调试阶段把尺寸限制在 `--max 64M` 以内；1 GB 的完整测试留到 P7 一次跑完，因为机器是共享的。

---

## 2 背景与约定

### 2.1 记号

- p = 4，环为 0→1→2→3→0；next = (g+1) mod p，prev = (g−1) mod p。
- n：每张卡的元素数；S = 4n：每张卡的字节数。下文的“消息大小”都指 S。
- 第 c 块（chunk）指元素 [c·n/p, (c+1)·n/p)。r4 再把每块切成 K 个小块，每个小块有 n/(pK) 个元素。

### 2.2 Ring 调度（本实验统一使用）

| 阶段 | 第 s 步（s = 0..p−2），GPU g 发出 | 写到哪里 | 接收方 next 做什么 |
|---|---|---|---|
| ReduceScatter | 第 (g − s) mod p 块 | next 的 tmp | 把 tmp 加到自己 buf 的同一块 |
| AllGather | 第 (g + 1 − s) mod p 块 | next 的 buf 的同一位置 | 无（直接覆盖） |

C++ 的 `%` 对负数会返回负值，例如 `(0 - 1) % 4 == -1`。框架在 `common.h` 里提供了 `mod(a, p)`，环上的下标请写成 `mod(g - s, p)`。

### 2.3 带宽口径

- algbw = S / t，busbw = algbw × 2(p−1)/p，GB 按 10⁹ 字节计。
- 所有版本都用这个公式，包括不是 ring 的 r1。busbw 是 AllReduce 这个操作的口径，与算法无关，nccl-tests 也是这样算的，所以不同版本的数字可以直接比较。

### 2.4 α-β 模型

传一次 m 字节的代价为 T(m) = α + m/β；设 γ 为加法 kernel 的有效带宽。

- **Ring（不流水）**：T ≈ 2(p−1)·α_step + 2(p−1)/p · S/β + (p−1)/p · S/γ
- **朴素（汇总到 GPU0 再广播）**：T ≈ k·α + (p−1)·S/β_in + (p−1)·S/β_out。β_in、β_out 是 GPU0 一张卡的收、发带宽（瓶颈链路），k 是“轮数”。
- **Ring + K 段流水**：T ≈ 2(p−1)·K·α_piece + 2(p−1)/p · S/β + 流水线的填充/排空项

**α 在本实验里的含义。** α 不只是“链路延迟”，而是所有与消息大小无关的开销之和：host 端下发 API、驱动中转、跨卡 event 信号、kernel launch、host 同步往返……单线程依次给 4 张卡下发时，哪些开销会串行累加、谁占大头，需要你自己测出来（P1(d)、P5(d)、P8(a)）。

### 2.5 gpu01 的拓扑

| 项目 | 配置 |
|---|---|
| GPU | 4 × RTX 5090（sm_120，32 GB），每张 PCIe Gen5 x16 |
| CPU / 内存 | 2 × Xeon Gold 6526Y，64 个逻辑核，503 GB |
| NUMA | GPU0/1 → node0（CPU 0–15, 32–47）；GPU2/3 → node1（CPU 16–31, 48–63） |
| `nvidia-smi topo -m` | GPU0–GPU1、GPU2–GPU3 为 NODE；跨两组为 SYS |
| P2P | `nvidia-smi topo -p2p r` 全部为 CNS（chipset not supported） |
| 软件 | CUDA 12.9（/usr/local/cuda），driver 580.126.09，NCCL 2.30.7（~/nccl_local） |

由此可以得到几个事实：

- 没有 P2P 时，`cudaMemcpyPeerAsync` 由驱动经主机内存中转；NCCL 走 SHM transport，同样经过主机内存。
- 环 0→1→2→3→0 中，1→2 和 3→0 两条边跨 NUMA 节点，要经过 UPI。
- GPU 空闲时 PCIe 链路会降速：空闲时 `nvidia-smi --query-gpu=pcie.link.gen.current --format=csv` 显示 1。所以任何测量之前都要 warmup。

### 2.6 已有的 NCCL 基线

下表节选自 `~/nccl-tests/nccl_matrix/summary.csv`（`all_reduce_perf -b 1K -e 1G -f 2 -g 4`，out-of-place）：

| size | Ring_Simple time (µs) | default_default busbw | Ring_Simple busbw | Tree_Simple busbw |
|---:|---:|---:|---:|---:|
| 1 KB | 28.56 | 0.05 | 0.05 | 0.05 |
| 32 KB | 32.36 | 1.80 | 1.52 | 1.29 |
| 256 KB | 43.08 | 5.44 | 9.13 | 5.18 |
| 1 MB | 82.22 | 9.59 | 19.13 | 7.69 |
| 4 MB | 222.81 | 11.25 | 28.24 | 13.31 |
| 32 MB | 2356.5 | 21.20 | 21.36 | 22.25 |
| 256 MB | 18378.9 | 21.93 | 21.91 | 24.65 |
| 1 GB | 71805.6 | 22.44 | 22.43 | 25.05 |

nccl-tests 的计时方法是：1 个 host 线程控制 4 张卡，warmup 1 次，然后把 20 次 AllReduce 背靠背发出去，最后只同步一次，总时间除以 20。这和 P2 规定的“每次迭代前后都同步”不同，P2(b) 会让你量化这个差别。

---

## 3 Problem 0：环境与构建

```bash
ssh ustcc
nvidia-smi                          # 确认 4 张卡空闲
cd ~/pmpp/ring-allreduce
make                                # 编译 ring 和 probe
./ring --versions nccl --max 64K --iters 3   # 只跑 NCCL 基线，确认框架能用
make venv                           # 建 .venv 并装 matplotlib（uv 在 /snap/bin/uv，已配置 USTC 镜像）
```

`pmpp-a1-gemm/.venv` 和系统 python3 都没有 matplotlib，而前者不允许改动，所以框架用实验目录自己的 `.venv`（已写进 `.gitignore`）。

> **注意**：系统里还有另一份 NCCL（`/lib/x86_64-linux-gnu/libnccl.so.2`）。Makefile 用 rpath 和 `LD_LIBRARY_PATH` 保证加载的是 `~/nccl_local` 里的 2.30.7。确认方法：`ldd ./ring | grep nccl` 应指向 `~/nccl_local/lib`，`ring` 启动信息里的 NCCL 版本应为 23007。

> **Problem（`environment`）：环境与构建（2 分）**
>
> 按上面的步骤编译并运行框架。`ring` 启动时会打印每张 GPU 的名字、PCI bus id、NUMA 节点、P2P 可访问的对数、CUDA driver/runtime 版本和 NCCL 版本。确认这些信息和 2.5 节一致。
>
> **Deliverable**：启动信息贴进 README 的“环境”一节；如果和 2.5 节有出入，写明哪里不同。

---

## 4 Problem 1：先认识你的“链路”

Ring 带宽最优的论证有一个前提：p 条链路互相独立，并且能同时满速工作。在 gpu01 上，每条“链路”实际上是 GPU → 主机内存 → GPU，中转由驱动完成。所以在写任何 ring 之前，先把这个传输原语的性质测清楚，后面每一个分析都要用到这些数字。

测量要求：

- 用 host 端计时（`std::chrono::steady_clock`）；每次测量前对所有卡 `cudaDeviceSynchronize`；先 warmup；重复足够多次取平均。
- 区分两个时间：**调用返回时间**（API 调用本身让 host 阻塞了多久）和**完成时间**（直到 `cudaStreamSynchronize` 返回）。
- 尺寸：1K、64K、1M、4M、16M、64M、256M。

框架：`probe.cu` 已经分配好每卡的 `src`/`dst` 缓冲、一块 pinned 的 `host` 缓冲和每卡两个 stream（`st[g]`、`st2[g]`），并做了 warmup。计时工具 `measure(R, setup, op)` 按上面的要求返回调用返回时间和完成时间；`setup` 不计时，可以用来制造前序工作。(a) 已经写成一个完整的例子，(b)–(e) 的函数里写好了要测什么，由你来实现。`./probe a` 只跑 (a)，`./probe bd` 只跑 (b) 和 (d)。

> **Problem（`link_probe`）：认识链路（12 分）**
>
> **(a) 单条拷贝（3 分）** GPU0→GPU1 的 `cudaMemcpyPeerAsync`（用 GPU0 的 stream）：框架里的 `section_a()` 已经测好了 size | 调用返回 (µs) | 完成 (µs) | GB/s，运行它并读懂代码。回答：它对 host 真的是异步的吗？调用返回时间怎样随大小变化？
>
> **(b) 并发（3 分）** 测三种情况：(i) ring 的 4 条边 g→g+1 同时拷贝，各用发送方的 stream；(ii) GPU0 用两个 stream 同时发 0→1 和 0→2；(iii) 1→0、2→0、3→0 同时扇入 GPU0。回答：这些拷贝真的并发了吗？整个环同时工作时，每条边能拿到多少带宽？这个数对 r2–r4 的大消息 busbw 意味着什么？(iii) 对 r1 又意味着什么？
>
> **(c) 参考上限（2 分）** 测每张卡 pinned 内存的 D2H、H2D 单向带宽，以及同一张卡 D2H 与 H2D 同时进行时每个方向的带宽。再测一次“手动中转”：D2H 到 pinned 缓冲，完成后再 H2D（串行）。回答：和 (a) 相比，驱动内部有没有把中转做成流水？GPU0/1 和 GPU2/3 的数字一样吗？为什么？
>
> **(d) 小开销（2 分）** 测 host 端的单次开销：kernel launch、`cudaEventRecord`、`cudaStreamWaitEvent`（同卡 / 跨卡）、空 stream 上的 `cudaStreamSynchronize`、1 KB 的 `cudaMemcpyPeerAsync`。再测一条依赖链的 GPU 端延迟，例如 1 KB 拷贝沿 0→1→2→3→0… 一跳一跳传，每跳都等上一跳的 event。这些数字是你在 P8 里拼出 α 的“原料”。
>
> **(e) NUMA（2 分）** 让 4 张卡同时做 D2H / H2D / 两者同时进行，host 缓冲分别放在：全在 node0、全在 node1、各自的本地节点、各自的远端节点。回答：什么限制了经过主机内存的并发流量？据此猜测驱动给 `cudaMemcpyPeerAsync` 用的中转缓冲放在哪里，并用这个猜测解释 (b)。
>
> **Deliverable**：`probe.cu` 的 `section_b()` … `section_e()`；每个子问题一张表，加 1–3 句回答。

> **提示**
>
> - 想知道调用会不会被“前面还没执行完的工作”阻塞，可以在 `measure` 的 `setup` 里先 `launch_spin(2'000'000, st[0])`（GPU 空转 2 ms），或者让 stream 先等一个跨卡 event，然后只测拷贝调用本身的耗时。
> - 测 GPU 端延迟时，如果“下发耗时 ≈ 执行耗时”，你测到的其实是 host 端开销。可以先用一个 spin kernel 把 stream “挡住”，趁这段时间把整条链都下发完，再看链本身的执行时间。
> - 如果调用本身会阻塞 host，4 条拷贝的启动时刻就是错开的。计算 (b) 的带宽时，要想清楚分母应该取什么。
> - (e) 要控制 pinned 内存放在哪个 NUMA 节点，直接用框架的 `host_alloc_on_node(bytes, node)`。它的原理是：节点 N 的 CPU 列表在 `/sys/devices/system/node/nodeN/cpulist`；分配前用 `sched_setaffinity` 把当前线程绑到该节点，`cudaHostAlloc` 锁页时会触碰页面，按 first-touch 原则落在当前节点上；分配完再恢复原来的 affinity。GPU 所在的节点用 `gpu_numa_node(g)` 查。
> - 源缓冲和目的缓冲分开分配；扇入测试中，各发送方写到目的缓冲的不同偏移。

---

## 5 Problem 2：读懂框架与 NCCL 基线

测试框架（`ring.cu`）和 NCCL 基线已经写好。这一节说明框架是怎么工作的：5.1 是你的代码要遵守的接口，5.2–5.7 是框架已经实现的行为，读懂即可。框架代码可以改（比如加一个参数），改了请在 README 里说明。

### 5.1 接口（`allreduce.h`）

```cpp
struct Ctx {                    // 框架分配和初始化，你的函数直接使用
  int p;                        // GPU 数（gpu01 上为 4）
  float* buf[MAXG];             // 每卡 n 个 float：输入，结束时为结果（in-place）
  float* tmp[MAXG];             // 每卡的接收区（device 内存），容量 tmp_elems[g] 个 float，见 5.5
  size_t tmp_elems[MAXG];
  cudaStream_t s[MAXG];         // 每卡一个主 stream（cudaStreamNonBlocking）
};
struct Params { int K, S; size_t max_n; std::string r5_numa; };   // 来自命令行

void ar_r1(Ctx& c, size_t n);                 // P3
void ar_r2(Ctx& c, size_t n);                 // P4
void r3_setup(Ctx& c, const Params& prm);     // P5：测试开始前调用一次，创建 event 等
void ar_r3(Ctx& c, size_t n);
void r4_setup(Ctx& c, const Params& prm);     // P6
void ar_r4(Ctx& c, size_t n);
void r5_setup(Ctx& c, const Params& prm);     // P9（选做）
void ar_r5(Ctx& c, size_t n);
```

需要额外状态（event、辅助 stream、FIFO 序号、host 槽位）的版本，把状态放在 `allreduce.cu` 里对应的 `namespace r3 { … }` 等位置，在 `rX_setup` 里创建。

语义：函数返回时，所有工作都已下发（不一定已执行完）；调用者对所有卡同步之后，`buf` 就是结果。对 r3、r4（和 r5）还有一条更强的要求：**连续调用多次、中间不做任何 host 同步，结果也必须正确**（见 5.3 的 `--check-reps`），就像 stream-ordered 的 `ncclAllReduce` 一样。

还没实现的函数保留 `NOT_IMPLEMENTED(...)`：框架捕获这个异常后跳过该版本，并在输出里注明。

### 5.2 尺寸

- 每张卡 1 KB 到 1 GB，每次 ×2。
- 大小必须能被 p × K × 4 字节整除（K 是 r4 的小块数）。不能整除的尺寸跳过，并打印一行说明。所有版本使用同一组尺寸。
- 外层循环尺寸，内层循环版本。这样同一尺寸的各版本在时间上挨着测，能减少共享机器上环境波动的影响。

### 5.3 正确性

- 用一个 kernel 初始化：`buf[g][i] = g + 1 + (i % 7)`。
- 一次 AllReduce 之后，每张卡的期望值为 E[i] = Σ_g (g + 1 + i%7) = p(p+1)/2 + p·(i%7)。
- `#wrong` = 所有卡上与期望值不相等的元素总数，直接用 `==` 比较。
- `--check-reps m`：初始化一次，背靠背调用 m 次（中间不做 host 同步），此时期望值为 p^(m−1)·E[i]。

### 5.4 计时

- 对每个（版本, 尺寸）：先做一次校验，再 warmup 5 次，然后计时 20 次取平均。
- 每次计时迭代的步骤：对所有卡同步 → 记 t0 → 调用一次 → 记 t_issue → 对所有卡同步 → 记 t1。报告平均的 `time = t1 − t0`，以及平均的**下发时间** `issue = t_issue − t0`。如果 `issue ≈ time`，说明这个版本卡在 host 端。这一列在 P5、P8 里非常有用。
- “对所有卡同步”指对每张卡执行 `cudaSetDevice` + `cudaDeviceSynchronize`，这样能覆盖辅助 stream 和驱动内部的中转操作。
- `--timing batch`：同步 → t0 → 背靠背调用 20 次 → 同步 → t1，再除以 20，也就是 nccl-tests 的方法。

### 5.5 显存与内存预算（以 1 GB 为例）

| 用途 | 大小 |
|---|---|
| buf[g] | n 个 float，每卡 1 GB |
| r1：GPU0 的 tmp（接收其余 p−1 张卡的数据） | (p−1)·n，3 GB |
| r2/r3 的 tmp | n/p，每卡 256 MB |
| r4 的 tmp（S 个槽） | S·n/(pK) |
| host 端校验缓冲 | 1 GB（pageable 即可） |

框架按最大尺寸把所有缓冲分配一次，各尺寸复用；32 GB 显存绰绰有余。`tmp` 的容量保证是：每张卡 ≥ max(cap/p, S·cap/(pK))；测试包含 r1 时，GPU0 ≥ (p−1)·cap（cap 为最大尺寸的元素数）。r5 的 pinned host 槽位要你在 `r5_setup` 里自己分配。

### 5.6 输出格式（模仿 nccl-tests）

每个版本一张表：

```
# r3  Ring + cudaEvent 只和邻居同步
#       size         count      time   algbw   busbw  #wrong      issue
#        (B)    (elements)      (us)  (GB/s)  (GB/s)              (us)
        1024           256       ...     ...     ...       0        ...
```

再输出一张汇总表，每个尺寸一行，各版本的 busbw 并排：

```
# 汇总 busbw (GB/s)
#    size(B)        r1        r2        r3        r4      nccl
```

CSV 用长表格式，每行对应一个（版本, 尺寸）：

```
version,K,slots,timing,size_bytes,count,time_us,issue_us,algbw_GBps,busbw_GBps,wrong
```

### 5.7 NCCL 基线（`ring.cu` 里的 `ar_nccl`）

- 用 `ncclCommInitAll(comm, 4, {0,1,2,3})` 初始化。
- 在 `ncclGroupStart()` / `ncclGroupEnd()` 之间，对每张卡调用 in-place 的 `ncclAllReduce(buf[g], buf[g], n, ncclFloat, ncclSum, comm[g], s[g])`。
- 与你自己的版本使用同一组 stream、同一套校验和计时代码。

> **Problem（`harness`）：读懂框架与 NCCL 基线（8 分）**
>
> **(a)（2 分）** 跑 nccl 的全部尺寸：`./ring --versions nccl --csv results/nccl_only.csv | tee results/nccl_only.txt`，确认 `#wrong = 0`。读 `ring.cu` 的 `verify` 和 `time_us`，用 5–8 句话说明一次（版本, 尺寸）测量的完整流程：校验做了什么、warmup 几次、计时的起止点在哪里、issue 列是怎么来的。
>
> **(b)（3 分）** 把你测到的 nccl 结果和 nccl-tests 的 default_default 对比。用 `--timing sync` 和 `--timing batch` 各跑一次 nccl，在 1 KB、1 MB、256 MB 三个尺寸上列表比较 time、issue 和 busbw。解释差异来自哪里：计时方法、in-place 与 out-of-place、NCCL 选择的 algorithm/protocol。
>
> **(c)（1 分）** 为什么这里可以用 `==` 精确比较浮点数？这种初始化方式有哪些错误发现不了？块长满足什么条件时，“数据写错了块”这类错误会被漏掉？
>
> **(d)（2 分）** 为什么背靠背调用（`--check-reps > 1`）能发现“每次调用前后都全体同步”时发现不了的错误？举一个 ring 实现里的具体例子。
>
> **Deliverable**：`results/nccl_only.txt`；(a) 的流程说明；(b) 的表格；(b)–(d) 的回答。

> **提示**：用 `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING`（或 `COLL`）可以看到每次调用选中的 algorithm / protocol。也可以用 `NCCL_ALGO=Ring NCCL_PROTO=Simple` 把它固定下来，再与 nccl-tests 的 Ring_Simple 列对比。

---

## 6 Problem 3：r1 朴素实现

规格：

1. 每张 GPU g（g ≠ 0）把整份 `buf[g]` 拷到 GPU0 的 tmp 第 g−1 段，各用自己的 stream。
2. 全体同步；GPU0 用一个 kernel 计算 `buf[0][i] += Σ_j tmp[0][j·n + i]`。
3. GPU0 把 `buf[0]` 拷回其余每张卡，最后全体同步。

阶段之间和 r2 一样只用 host 同步，这样 r1 → r2 的差别就只在算法上。

> **Problem（`r1_naive`）：朴素实现（10 分）**
>
> **(a)（5 分）** 在 `allreduce.cu` 里实现 `ar_r1`，`#wrong = 0`。
>
> **(b)（3 分）** 先预测，再测量。用 α-β 模型和 P1 的数据指出 r1 的瓶颈链路，预测 1 GB 时的 busbw，以及 1 KB 时的耗时（数一数有几轮拷贝、几次同步）；然后与实测对比。
>
> **(c)（2 分）** 广播阶段的 3 次拷贝：放在 GPU0 的同一个 stream 上，和分别放到 GPU0 的 3 个 stream 上（kernel 之后先做一次全体同步）相比，有没有区别？用 1 KB、64 MB、1 GB 的数据和 P1(b)(ii) 解释。
>
> **Deliverable**：代码；预测与实测对照表；(b)(c) 的回答。

---

## 7 Problem 4：r2 Ring + 全体同步

规格：使用 2.2 节的调度。**只允许用 host 同步表达依赖，不使用任何 event**：只要后一个操作依赖前一个操作的结果，或者会覆盖前一个操作还在使用的数据，就对所有卡的 stream 做 `cudaStreamSynchronize`。

> **提示**：先写一个 30 行左右的 CPU 模拟器：p 个 host 数组，同样的下标公式，同样的执行顺序，用 p = 2..8 验证结果。下标写错在 CPU 上很好查；到了 4 张 GPU 上，你能看到的就只剩一个 `#wrong` 数字。

> **Problem（`r2_ring_bsp`）：Ring + 全体同步（12 分）**
>
> **(a)（5 分）** 实现 `ar_r2`，`#wrong = 0`。
>
> **(b)（4 分）** 证明：ReduceScatter 结束时，GPU g 持有第 (g+1) mod p 块在所有卡上的总和；AllGather 结束时，每张卡都持有全部 p 块的结果。（提示：找一个关于“第 s 步之后，谁持有第 c 块的哪些卡的部分和”的不变式，对 s 做归纳。）
>
> **(c)（3 分）** 只用 host 同步时，一个 RS 步内“拷贝 → 加法”的依赖你是怎么保证的？一次 AllReduce 一共做了几次全体同步（写成 p 的函数）？用小消息的实测时间，估计每次全体同步连同它前面那批操作的代价。
>
> **Deliverable**：代码；证明；(c) 的公式与估计。

---

## 8 Problem 5：r3 Ring + event 只和邻居同步

规格：每张卡一个 stream，所有依赖都用 cudaEvent 表达，整个 AllReduce 中途不做 host 同步。至少要处理下面这些依赖：

- ① next 的加法要等 g 拷到 next 的 tmp 的那次拷贝完成；
- ② next 下一步要发出的块，要等它这一步的加法完成（同一个 stream 的顺序即可保证）；
- ③ g 往 next 的 tmp 写下一份之前，要等 next 用完上一份（防止覆盖）；
- AllGather 中“转发上一步收到的块”的依赖；
- 收尾时的 stream 语义：调用结束后，“GPU g 的主 stream 完成”应当意味着“`buf[g]` 已是最终结果”。

请在代码注释里写清每个依赖对应上面哪一条。

> **关于 cudaEvent 的几条事实**
>
> - `cudaStreamWaitEvent(stream, ev)` 等的是**调用那一刻** `ev` 最近一次 `cudaEventRecord` 所捕获的工作；之后再对 `ev` 做 record，不影响已经发出的等待。
> - 等待一个从未 record 过的 event，会立即满足。
> - `cudaEventRecord(ev, stream)` 要求 `ev` 和 `stream` 属于同一张卡；`cudaStreamWaitEvent` 则可以等另一张卡上的 event。所以 event 要在“负责 record 它的那张卡”上创建。
> - 用 `cudaEventDisableTiming` 创建 event，开销更小。
> - 如果每张卡只用一个 event 跨步复用，host 的下发顺序决定了每次 wait 等到的是哪一次 record。

> **Problem（`r3_ring_events`）：Ring + event 邻居同步（16 分）**
>
> **(a)（6 分）** 实现 `r3_setup` 和 `ar_r3`，注释写清每个依赖；在 `--check-reps 3` 下 `#wrong = 0`。
>
> **(b)（4 分）** AllGather 直接覆盖 next 的 buf，却没有“next 已经用完这一块”的 event。证明这样做是安全的：从 next 对第 c 块的最后一次使用出发，找出一条由 stream 顺序和 event 依赖组成的链，一直通到 g 在 AG 第 s 步写第 c 块。（提示：倒着追 g 在 AG 第 s 步发出的数据是从哪里来的。）
>
> **(c)（2 分）** 论证你的下发顺序不会死锁。另外，如果调用者只同步 GPU g 自己的主 stream，`buf[g]` 一定是最终结果吗？你为此做了什么？
>
> **(d)（4 分）** 对比 r2 和 r3 在各尺寸下的 time 和 issue。r3 的提升来自哪里？如果在某些尺寸上 r3 并没有更快，先别认定是自己写错了，用 issue 列（以及可选的 nsys 时间线）解释原因。
>
> **Deliverable**：代码；(b) 的证明；(c)(d) 的回答，(d) 附表。

> **提示：用 nsys 看时间线**
>
> ```bash
> mkdir -p /tmp/$USER/prof
> nsys profile -t cuda -s none -o /tmp/$USER/prof/r3_1k -f true \
>     ./ring --versions r3 --min 1K --max 1K --warmup 2 --iters 3
> nsys stats --report cuda_api_sum /tmp/$USER/prof/r3_1k.nsys-rep
> ```
>
> `nsys stats` 会顺带生成同名的 `.sqlite` 文件。表 `CUPTI_ACTIVITY_KIND_RUNTIME` 记录 API 调用，`CUPTI_ACTIVITY_KIND_MEMCPY` 和 `CUPTI_ACTIVITY_KIND_KERNEL` 记录 GPU 上的操作；按 `start` 排序，用 `correlationId` 关联，名字在 `StringIds` 表里。注意 profiler 本身会把 API 耗时放大。看完记得把 `/tmp/$USER/prof` 删掉。

---

## 9 Problem 6：r4 Ring + 流水线

规格：

- 每块切成 K 个小块（默认 K = 8）。第 c 块的第 k 个小块，是从元素 c·n/p + k·n/(pK) 开始的 n/(pK) 个元素。
- tmp[g] 做成 S 个槽位的环形 FIFO（默认 S = 8，对应 NCCL 每个 channel 的 8 格 FIFO），每个槽放一个小块。每条边 g→next 维护一个单调递增的序号 seq，跨调用保留；第 seq 个小块放进槽 seq % S。
- 每张卡两个 stream：一个负责 g 的所有发送（拷贝），主 stream 负责 g 的所有加法。目标是让“g 把下一个小块拷给 next”和“next 加上一个小块”同时进行，步与步之间不再出现整块大小的空泡。
- 依赖按小块粒度表达：槽“满”（相当于 NCCL 的 tail）、槽“空”（相当于 head）、“这个小块可以转发了”（RS 中是加法完成，AG 中是已经收到）。
- 开头要 fork：辅助 stream 要排在主 stream 上调用者已提交的工作之后。结尾要 join：主 stream 要等辅助 stream 和最后收到的数据，保证 P5 里说的 stream 语义。
- 下发顺序：外层循环步，内层循环小块。

> **注意：下面这些坑容易踩，而且不会报错**
>
> - **event 复用导致的隐性串行**：如果在同一轮下发循环里，A 先 record 了某个 event，B 随后 wait 它，那么 B 等到的是“这一轮”而不是“上一轮”的 record。结果仍然正确，只是凭空多了一个依赖，流水线被串成了一条线。解决办法有两个：把数据依赖的 event 按（步, 小块）编号；或者把发送和接收拆成两个循环，保证同一轮里同一个 event 的 record 和 wait 不会交错。
> - 槽位地址 tmp + j·n/(pK) 依赖于 n，所以换消息大小之前要先同步。框架在每个尺寸开始前都会同步，正好满足这一点。
> - K 必须整除 n/p。对于 2 的幂的尺寸，只有 K 也是 2 的幂时才不会把所有尺寸都跳过。可以试一下 `--k 3`，检查跳过规则是否生效。

> **Problem（`r4_pipeline`）：Ring + 流水线（18 分）**
>
> **(a)（6 分）** 实现 `r4_setup` 和 `ar_r4`（K、S 从 `prm.K`、`prm.S` 读）。在 (K, S) ∈ {(1,1), (8,8), (8,2), (32,4)} 下，`--check-reps 3` 都要 `#wrong = 0`。
>
> **(b)（2 分）** K = 1、S = 1 的 r4 和 r3 在语义上有什么不同？实测时间接近吗？为什么？
>
> **(c)（5 分）** K 扫描：`make ksweep` 会用 S = 8、K ∈ {1, 2, 4, 8, 16, 32} 跑 64K–1G，结果写进 `results/k_sweep.csv`；再用 `.venv/bin/python plot.py results/k_sweep.csv --k-sweep` 画出 busbw 随 K 变化的曲线（默认取 64K、1M、16M、256M、1G 五个尺寸，每个尺寸一条线），存为 `results/k_sweep.png`。用 α-β 模型解释最优 K 如何随消息大小变化：哪一项随 K 变大，哪一项随 K 变小？
>
> **(d)（2 分）** S 扫描：256 MB，K = 8，S ∈ {1, 2, 4, 8}（`./ring --versions r4 --min 256M --max 256M --slots S`）。S 什么时候重要，什么时候不重要？
>
> **(e)（3 分）** 大消息下从 r3 到 r4，流水线到底藏掉了什么？估算被藏掉的量（每步加法 kernel 的时间、步与步之间的依赖延迟等），并与实测的差值对比。
>
> **Deliverable**：代码；`results/k_sweep.png`；(b)–(e) 的回答（附数据）。

---

## 10 Problem 7：全尺寸测试与绘图

> **Problem（`full_benchmark`）：全尺寸测试与绘图（6 分）**
>
> **(a)（3 分）** 确认 4 张卡空闲后跑完整测试：`make run`，它等价于
>
> ```bash
> ./ring --versions r1,r2,r3,r4,nccl --min 1K --max 1G --k 8 \
>        --csv results/allreduce.csv | tee results/allreduce.txt
> ```
>
> 输出中有每个版本一张表，以及 busbw、time 两张汇总表。在 README 里记录测试前后 `nvidia-smi` 上有没有其他进程。
>
> **(b)（3 分）** `make plot` 生成两张图：
>
> - `results/busbw.png`：x 轴为消息大小（log2 刻度），y 轴为 busbw，每个版本一条线；如果 `~/nccl-tests/nccl_matrix/summary.csv` 存在，其中的 `Ring_Simple` 和 `default_default` 两列会用虚线画上。
> - `results/time.png`：time 随 size 变化的 log-log 图。小消息段的“平台”高度就是各版本的 α。
>
> 把两张图放进 README，并写出你从图里读到的三个现象（每个 1–2 句）。
>
> **Deliverable**：`results/allreduce.txt`、`results/allreduce.csv`、两张图和三个现象。

---

## 11 Problem 8：α-β 分析（写在 README 里）

> **Problem（`alpha_beta`）：α-β 分析（16 分）**
>
> **(a) 小消息（4 分）** 给出每个版本的 α：取 1 KB 时的耗时，或者对 1K–64K 拟合 T = a + b·S 取截距。把 α 拆成“次数 × 单价”：几步、几次 API 调用、几次全体同步，单价用 P1(d) 的数据。哪一项占大头？为什么单线程下发会让 α 随 p 增长（r4 还随 K 增长）？
>
> **(b) 大消息（4 分）** 列出各版本在 ≥ 256 MB 时的 busbw，与 NCCL 以及 P1(b)/(e) 得到的上限对比。你的实现比 NCCL 差多少？差距主要在 β（链路效率）还是残余的 α？在这台机器上，β 被什么限制住了？
>
> **(c) r2 → r3 → r4（4 分）** 每一步改动分别改变了模型里的哪一项？候选包括：全体同步的次数、host 往返、host 下发能否与 GPU 执行重叠、步与步之间的空泡、K 带来的额外 α。用数据支撑（issue 列、P1 测出的单价、实测差值）。
>
> **(d) 交叉点（2 分）** r1 在哪些尺寸上比 ring 版本快？r4 从多大的消息开始比 r3 快？先用拟合出的 α、β 做预测，再和实测对比。
>
> **(e) 和 NCCL 比 α（2 分）** 一次 `ncclAllReduce` 需要多少次 host API 调用？你的 r3、r4 各需要多少次？NCCL 把你用 API 调用完成的哪些事情搬进了 kernel 内部？
>
> **Deliverable**：README 中的分析一节，包含一张“版本 | α (µs) | β 或大消息 busbw (GB/s) | 拟合范围”的表，以及以上各题的回答。

---

## 12 Problem 9（选做）：r5 手动 pinned host 中转

什么时候做：如果 P1 显示 `cudaMemcpyPeerAsync` 在无 P2P 时表现异常（例如不是真正异步、多条边并发时被串行化，或者带宽上不去），就自己来做中转，看能不能比驱动做得更好。

规格：小块、槽位和序号都和 r4 一样，但每次“发送一个小块”拆成三段：

1. D2H：发送方在自己的 stream 上，把小块拷到这条边专用的 pinned host 槽；
2. H2D：接收方在自己的另一个 stream 上，把 host 槽拷进 tmp 槽（RS），或者直接拷进 buf（AG）；
3. 在 RS 中，接收方在主 stream 上做加法。

D2H 和 H2D 分属两张卡上的两个 stream，所以“D2H 第 k+1 个小块”和“H2D 第 k 个小块”可以重叠。host 槽的满/空需要额外的 event 来表达。host 槽的位置由命令行 `--r5-numa send|recv|off` 选择（在 `r5_setup` 里通过 `prm.r5_numa` 拿到），分别表示放在发送方的 NUMA 节点、放在接收方的节点、不绑定；分配用 `host_alloc_on_node(bytes, node)`，大小可以按 `prm.max_n`、`prm.K`、`prm.S` 算。

> **Problem（`r5_staging`）：手动中转（选做，+10 分）**
>
> **(a)（4 分）** 实现 `r5_setup` 和 `ar_r5`，在 `--check-reps 3` 下 `#wrong = 0`。
>
> **(b)（3 分）** 在 16 MB–1 GB 范围内对比 r4 和 r5（三种放置方式），列表。
>
> **(c)（3 分）** 用 P1(e) 解释结果：哪种放置最好？跨 UPI 的是 D2H 时损失大，还是 H2D 时损失大？小消息时 r5 为什么仍然很慢？
>
> **Deliverable**：代码；表；回答。

---

## 13 选做探索（不计分）

- **多线程下发**：每张卡一个 host 线程，各自下发自己的操作，跨卡依赖仍然用 event。α 能降多少？
- **CUDA Graph**：把一次 AllReduce 的全部操作 capture 成一张图再重放（多卡 capture 需要用 event 做 fork/join）。无 P2P 的 peer 拷贝能被 capture 吗？α 降了多少？
- **mini-NCCL**：每张卡一个常驻 kernel，通过映射进设备地址空间的 pinned host 内存（`cudaHostAllocMapped`）收发数据，用 host 内存里的 head/tail 标志同步，一次 AllReduce 只 launch 一次。注意 `__threadfence_system()` 和 volatile 轮询。和 NCCL 的 Simple protocol 比一比。
- **两个 channel**：把每块再分给两个拷贝 stream，或者用两个方向相反的环。有用吗？为什么？

---

## 附录 A：完成标准与报告结构

**完成标准**

- [ ] `make check` 通过，所有版本 `#wrong = 0`（`make check` 已经包含 `--check-reps 3`）。
- [ ] `results/` 下有：`allreduce.txt`（每个版本一张表 + 汇总表）、`allreduce.csv`、`busbw.png`（含 nccl-tests 的两条参考线）、`time.png`、`k_sweep.png`。
- [ ] README 写清：每个版本做了什么改动、为什么更快或更慢；α-β 分析，包括各版本的 α、大消息 busbw 与 NCCL 的差距、r2 → r3 → r4 每一步的提升来源。
- [ ] 如实记录 P1 观察到的现象，尤其是 `cudaMemcpyPeerAsync` 在无 P2P 时的异常；如果做了 r5，说明结果。
- [ ] 没有修改 `~/nccl-tests` 和 `~/pmpp/pmpp-a1-gemm`。
- [ ] 代码和结果已 `git commit`。

**报告结构**

`README.md` 已经按下面的结构给好了标题和待填项，直接往里写：

```
# L6 实验报告
## 环境（P0）
## 链路特性（P1）
## 读懂框架与 NCCL 基线（P2）
## r1 / r2 / r3 / r4（P3–P6）：每个版本的改动、为什么更快或更慢、关键数据
## 全尺寸结果（P7）：汇总表 + 图
## α-β 分析（P8）
## r5（P9，选做）
## 结论：用三段话回答 1.1 的三个问题
```

## 附录 B：常见坑

- **加载了错误的 NCCL**：系统自带一份 `libnccl.so.2`，务必按 P0 的方法确认版本。
- **event 与 stream 不在同一张卡**：`cudaEventRecord` 会直接报错。创建 event、launch kernel（包括 `launch_add`）之前，先 `cudaSetDevice` 到正确的卡。
- **隐式同步**：你自己创建的 stream 也要用 `cudaStreamNonBlocking`（框架的主 stream 已经是）。否则同步版的 `cudaMemcpy`（例如框架校验时用的）会和 legacy default stream 发生隐式同步，干扰计时。
- **冷启动**：GPU 空闲时 PCIe 链路会降速，第一次测量偏慢，一定要 warmup。
- **整数溢出**：字节数和偏移一律用 `size_t`。r1 的 tmp 在 1 GB 时有 3 GB，已经超出 32 位 int 的范围。
- **数值增长**：每做一次 AllReduce，buf 里的值就乘以 p。warmup 和计时之后的数据不能直接拿来校验，要先重新初始化（或按 `--check-reps` 的公式推算期望值）。
- **共享机器**：长时间测量前先看 `nvidia-smi`；外层循环按尺寸排列；结果明显异常时先确认有没有别人在用卡。
- **profile 产物**：nsys 的输出放在 `/tmp/$USER/` 下，不要放进仓库，用完删掉。
- **口径一致**：除非某个版本的设计本身要求不同，GPU 间拷贝都在发送方的 stream 上发起，并把当前设备设为发送方。这样版本之间的差异只来自你想比较的那个改动。

---

## 附录 C：参考数据（剧透！建议做完对应的 Problem 再看）

下面的数据由文档作者在 2026-10-20 测得，测量时 4 张卡空闲。你的数字和这里差 ±20% 很正常；如果差了一个数量级，先找 bug 或测量方法上的问题。

### C.1 P1 链路（probe）

**(a) 单条 `cudaMemcpyPeerAsync` 0→1**

| size | 调用返回 (µs) | 完成 (µs) | GB/s |
|---:|---:|---:|---:|
| 1 KB | 6.0 | 9.7 | 0.11 |
| 64 KB | 6.1 | 10.0 | 6.5 |
| 1 MB | 6.3 | 52.2 | 20.1 |
| 4 MB | 20.1 | 121.3 | 34.6 |
| 16 MB | 79.6 | 408.3 | 41.1 |
| 64 MB | 319.2 | 1551.8 | 43.3 |
| 256 MB | 1281.3 | 6130.6 | 43.8 |

调用返回时间在 1 MB 以下约为 6 µs；从 4 MB 起大约每 MB 增加 5 µs，256 MB 时 host 被阻塞 1.28 ms，约占完成时间的 21%。如果在同一个 stream 上先放一个 2 ms 的 spin kernel，或者先等一个跨卡 event，调用仍然在 6–7 µs 内返回，也就是说它不会因为前序工作没完成而阻塞。用 nsys 看，一次无 P2P 的 peer 拷贝由两部分组成：源卡 stream 上的一次 D2H，加上约 5 µs 后目的卡内部 stream 上的一次 H2D。

**(b) 并发**

- ring 的 4 条边同时拷贝时，每条边的带宽：1 MB 为 11.0，4 MB 为 13.2，16 MB 为 14.7，64 MB 为 15.1，256 MB 为 15.2 GB/s。单线程依次发出 4 个 256 MB 的调用，本身就要 5.5 ms。
- GPU0 用两个 stream 同时发 0→1、0→2：256 MB 时每条 25.4 GB/s，合计 50.8 GB/s。
- 1、2、3 → 0 扇入：256 MB 时 GPU0 收到的总带宽为 42.8 GB/s，耗时约为单条拷贝的 3 倍。

**(c) 参考上限（256 MB；host 缓冲未绑定 NUMA，从数据看落在 node0）**

| GPU | D2H | H2D | D2H+H2D 同时（每方向） |
|---|---:|---:|---:|
| GPU0 / GPU1 | 56.6 | 56.2 | 48.5 |
| GPU2 / GPU3 | 41.6 | 56.2 | 33.3 |

手动串行中转（0 → host → 1）：28.2 GB/s。

**(d) 小开销（host 端，每次）**

| 操作 | 耗时 |
|---|---:|
| kernel launch 调用 | 1.7 µs |
| tiny kernel 串行执行 | 2.1 µs / 个 |
| `cudaEventRecord` + `cudaStreamWaitEvent`（同卡） | 0.25 µs / 对 |
| 空 stream 的 `cudaStreamSynchronize` | 0.17 µs |
| tiny kernel + `cudaStreamSynchronize` 往返 | 5.2 µs |
| 1 KB `cudaMemcpyPeerAsync` 调用 | 6.1 µs |
| 32 B D2H 调用（同一张卡 / 4 张卡轮流） | 1.9 / 2.1 µs |
| 跨卡 event 链（每跳一个 tiny kernel） | 3.3 µs / 跳（下发 ≈ 执行，host-bound） |
| 1 KB peer 拷贝沿环的依赖链 | 7.5 µs / 跳（下发 ≈ 执行，host-bound） |

**(e) NUMA：4 张卡同时拷贝 64 MB，每卡每方向的 GB/s**

| 操作 | 全在 node0 | 全在 node1 | 各自本地 | 各自远端 |
|---|---:|---:|---:|---:|
| D2H | 16.9 | 16.7 | 42.8 | 18.2 |
| H2D | 32.9 | 32.9 | 55.5 | 38.6 |
| D2H+H2D | 11.0 | 11.0 | 28.5 | 13.3 |

单卡 D2H：本地 56.4、远端 41–42 GB/s。两卡同时 D2H 到各自本地节点：同一个 socket 的 GPU0+1 各 43.3，跨 socket 的 GPU0+2 各 55.8 GB/s。

### C.2 参考实现（sync 计时模式，短跑，只用于判断量级）

| 版本 | 1 KB time（issue） | 1 MB time / busbw | 4 MB time / busbw |
|---|---:|---:|---:|
| r1 | 87–118 µs（84 µs） | 503 µs / 3.1 | 1.40 ms / 4.5 |
| r2 | 361–494 µs（357 µs） | 665 µs / 2.4 | 1.09 ms / 5.8 |
| r3 | 358–497 µs（352 µs） | 565 µs / 2.8 | 0.80 ms / 7.9 |
| r4（K=8, S=8） | 2.8–3.8 ms（2.8 ms） | 4.03 ms / 0.39 | 4.07 ms / 1.6 |
| r5（K=8, S=8, send） | 2.0–2.5 ms（2.0 ms） | 2.61 ms / 0.60 | 2.61 ms / 2.4 |
| nccl（default, in-place） | 44–52 µs（29 µs） | 194 µs / 8.1 | 255 µs / 24.7 |

1 KB 一列给的是两次短跑的范围，括号里是其中较稳定那次的 issue 时间。这里没有提供参考实现在大消息下的数据；大消息的上限请参考 C.1(b)(e) 和 2.6 节的 NCCL 基线。
