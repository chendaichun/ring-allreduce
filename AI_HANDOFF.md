# 给协助 L6 实验的 AI 的交接说明

> 读者：接手协助这个实验的 AI。2026-10-20 由上一个 AI 会话（Claude）整理。
> 人类读者注意：第 4、5 节包含实测结果和参考实现的设计要点，相当于剧透。

## 1. 背景与你的角色

- 用户在做一个学习实验：在 gpu01 上手写 Ring AllReduce（L6），并与 NCCL 对比。整体参照 Stanford CS336 的作业形式：`HANDOUT.md` 是实验指导，框架代码已经提供，**AllReduce 由用户自己写**。
- 用户偏好：用中文回复，技术名词保留英文。代码要求清晰、中文注释充分、每个版本独立成函数，不过度优化，不引入额外的库。
- **最重要的约定：不要替用户写 r1–r5 的实现**（`allreduce.cu` 里的函数），也不要替他写 `probe.cu` 的 (b)–(e)。以提示、追问、指出 bug 所在、解释现象为主；除非用户明确要求看代码。
- 经过：上一个 AI 起初误解成“由 AI 实现全部版本”，写了完整的参考实现（r1–r5 + NCCL + probe）并做了大量测量。用户纠正为“CS336 式实验框架，自己写 AllReduce”之后，参考实现已按用户要求删除（本地和 gpu01 上都删了）。现在保留的是实验指导、框架和这份交接说明。

## 2. 机器与规则

- 登录：`ssh ustcc`（主机 gpu01，用户 peile）。
- 硬件：4 × RTX 5090（sm_120，32 GB，PCIe Gen5 x16），2 × Xeon Gold 6526Y，503 GB 内存。GPU0/1 在 NUMA node0（CPU 0–15, 32–47），GPU2/3 在 node1（CPU 16–31, 48–63）。`nvidia-smi topo -p2p r` 全部为 CNS，即没有 P2P。
- 软件：CUDA 12.9（`/usr/local/cuda`），driver 580.126.09，NCCL 2.30.7（`~/nccl_local`）。系统里另有一份 `/lib/x86_64-linux-gnu/libnccl.so.2`，不要用它；Makefile 已经用 rpath + `LD_LIBRARY_PATH` 处理。nsys 在 `/usr/local/bin/nsys`，uv 在 `/snap/bin/uv`（已配置 USTC 镜像）。
- 规则：机器是共享的，测量前先 `nvidia-smi` 确认 4 张卡空闲；不用 sudo；不动别人的进程；不修改 `~/nccl-tests` 和 `~/pmpp/pmpp-a1-gemm`（包括它的 `.venv`）。profile 产物放 `/tmp/$USER/`，用完删掉。
- NCCL 基线数据：`~/nccl-tests/nccl_matrix/summary.csv`，是 `all_reduce_perf -b 1K -e 1G -f 2 -g 4` 在 NCCL_ALGO × NCCL_PROTO 矩阵下的结果，列名形如 `Ring_Simple_busbw`、`default_default_time_us`。

## 3. 交付物现状

位置：gpu01 上的 `~/pmpp/ring-allreduce`，是一个 git 仓库，还没有任何 commit。用户 Mac 上的 `/Volumes/macos/claude/ring-allreduce` 是本地镜像（用 rsync 同步），以 gpu01 上的为准。

| 文件 | 状态 | 内容 |
|---|---|---|
| `HANDOUT.md` | 完成 | CS336 式实验指导：P0–P8 共 100 分 + P9 选做（+10）；附录 C 是参考数据（剧透） |
| `README.md` | 模板 | 实验报告骨架，由用户填写 |
| `Makefile` | 完成 | `make` / `check` / `run` / `ksweep` / `venv` / `plot` |
| `common.h`、`common.cu` | 完成 | CK、mod、now_us、sync_devices、launch_init/add/spin/tiny、gpu_numa_node、host_alloc_on_node、NOT_IMPLEMENTED |
| `allreduce.h` | 完成 | `Ctx`、`Params`、各版本函数声明（`ar_r1`…`ar_r5`、`r3/r4/r5_setup`） |
| `allreduce.cu` | 空函数 | 所有版本都是 `NOT_IMPLEMENTED(...)`，由用户实现 |
| `ring.cu` | 完成 | 基准测试主程序 + NCCL 基线（`ar_nccl`） |
| `probe.cu` | 部分 | `measure()` 和 (a) 已完成，(b)–(e) 是 TODO |
| `plot.py` | 完成 | 生成 busbw.png、time.png、k_sweep.png；依赖 `.venv`（`make venv` 创建） |

2026-10-20 的验证结果：`make` 编译无警告；`make check` 通过（只有 nccl 在跑，r1–r5 因未实现而被跳过）；`./probe a` 正常；`plot.py` 在临时 venv 里测试过，三张图都能正确生成。实验目录里还没有 `.venv`，这是 P0 里用户自己要做的一步。

框架行为要点：
- 尺寸 1K–1G 每次 ×2，不能被 p·K·4 字节整除的跳过；外层循环尺寸，内层循环版本。
- 校验：初始化 `buf[g][i] = g + 1 + (i % 7)`，期望值 E[i] = p(p+1)/2 + p·(i%7)；`--check-reps m` 背靠背调用 m 次，期望 p^(m−1)·E[i]。
- 计时：warmup 5 次 + 计时 20 次；默认 sync 模式（每次迭代前后对所有卡 `cudaDeviceSynchronize`），可选 batch 模式（nccl-tests 的方法）。另外输出 issue 列，即 host 下发所用时间。
- 函数抛出 `NotImplemented` 的版本被跳过；任何已实现版本 `#wrong ≠ 0` 时退出码为 2。
- CSV 列：`version,K,slots,timing,size_bytes,count,time_us,issue_us,algbw_GBps,busbw_GBps,wrong`。

## 4. 实测结果（上一个 AI 在 gpu01 上测得，测量时 4 张卡空闲）

### 4.1 链路：无 P2P 时的 `cudaMemcpyPeerAsync`

- **不是完全异步**。调用会阻塞 host：≤ 1 MB 时约 6 µs；从 4 MB 起大约每 MB 增加 5 µs（256 MB 时阻塞 1.28 ms，约占完成时间的 21%）。但它**不会**因为前序工作没完成而阻塞：同一 stream 上先放 2 ms 的 spin kernel，或者先等一个跨卡 event，调用仍在 6–7 µs 内返回。
- **实现方式**（从 nsys 看）：源卡 stream 上一次 D2H，约 5 µs 后在目的卡的一个驱动内部 stream 上做 H2D。
- **单条带宽**：1 MB 20 GB/s，4 MB 35，16 MB 41，256 MB 43.8 GB/s。驱动内部做了流水；手动串行中转（D2H 完成后再 H2D）只有 28.2 GB/s。
- **并发不扩展**：ring 的 4 条边同时拷贝时，每条边只有 11.0（1 MB）/ 13.2（4 MB）/ 14.7（16 MB）/ 15.2（256 MB）GB/s。单线程依次发出 4 个 256 MB 的调用本身就要 5.5 ms。1、2、3 → 0 扇入时总带宽 42.8 GB/s，耗时约为单条的 3 倍，看起来是串行的。GPU0 用两个 stream 同时发两条：每条 25.4 GB/s，合计 50.8 GB/s。
- **参考上限**（pinned，256 MB，host 缓冲落在 node0）：GPU0/1 的 D2H 56.6、H2D 56.2、双向同时每方向 48.5；GPU2/3 的 D2H 41.6、H2D 56.2、双向 33.3 GB/s。
- **NUMA 是关键瓶颈**（4 卡同时拷贝 64 MB，每卡每方向 GB/s）：

  | 操作 | 全在 node0 | 全在 node1 | 各自本地 | 各自远端 |
  |---|---:|---:|---:|---:|
  | D2H | 16.9 | 16.7 | 42.8 | 18.2 |
  | H2D | 32.9 | 32.9 | 55.5 | 38.6 |
  | D2H+H2D | 11.0 | 11.0 | 28.5 | 13.3 |

  单卡 D2H 本地 56.4、远端约 41；同一 socket 的两卡同时 D2H 到本地各 43.3，跨 socket 的 GPU0+2 各 55.8。推断：驱动给 peer 拷贝用的中转缓冲集中在一个节点上，4 条边的流量挤在一起，所以每条边只有约 15 GB/s。跨 UPI 的 DMA 写（D2H 到远端）比读（H2D 从远端）损失更大。

### 4.2 小开销（host 端）

kernel launch 调用 1.7 µs；tiny kernel 串行执行 2.1 µs/个；同卡 eventRecord + streamWaitEvent 0.25 µs/对；空 stream 的 StreamSync 0.17 µs；tiny kernel + StreamSync 往返 5.2 µs；4 卡空 StreamSync 一轮 0.75 µs；1 KB peer 拷贝调用 6.1 µs；32 B D2H 调用 1.9 µs（4 卡轮流 2.1 µs）。跨卡 event 链每跳 3.3 µs，1 KB peer 拷贝沿环的依赖链每跳 7.5 µs，这两条链都是“下发 ≈ 执行”，也就是卡在 host 端。nsys 下 cudaEventRecord 约 3 µs、cudaMemcpyPeerAsync 约 12 µs，都被 profiler 放大了。

### 4.3 参考实现的耗时（sync 计时模式，短跑，只能判断量级）

| 版本 | 1 KB time（issue） | 1 MB time / busbw | 4 MB time / busbw |
|---|---:|---:|---:|
| r1 | 87–118 µs（84） | 503 µs / 3.1 | 1.40 ms / 4.5 |
| r2 | 361–494 µs（357） | 665 µs / 2.4 | 1.09 ms / 5.8 |
| r3 | 358–497 µs（352） | 565 µs / 2.8 | 0.80 ms / 7.9 |
| r4（K=8, S=8） | 2.8–3.8 ms（2.8 ms） | 4.03 ms / 0.39 | 4.07 ms / 1.6 |
| r5（K=8, S=8, send） | 2.0–2.5 ms（2.0 ms） | 2.61 ms / 0.60 | 2.61 ms / 2.4 |
| nccl（default, in-place） | 44–52 µs（29） | 194 µs / 8.1 | 255 µs / 24.7 |

要点：
- 小消息时所有手写版本都卡在 host 下发（issue ≈ time）。nsys 显示 1 KB 时 GPU 上的每个操作只有 0.3–0.9 µs，总时间约等于单线程依次下发约 130 个 API 调用的耗时。所以 r3 在小消息上并不比 r2 快，这是预期现象，不是 bug。
- r4（K = 8）的 API 调用数是 r3 的 8 倍，小消息非常慢。
- r1 只有两轮拷贝，小消息时反而是手写版本里最快的。
- NCCL 一次调用只 launch 一个 kernel（每卡），所以 α 约 44 µs（sync 模式）。
- 参考实现在 1K–4M、`--check-reps 3` 下全部 `#wrong = 0`。

### 4.4 没有测过的

- 参考实现在 ≥ 16 MB 时的结果（全尺寸测试还没跑过）。
- r5 在大消息时能否接近 NCCL。按 4.1 的 NUMA 数据推测，host 槽位放在本地节点时每条边可以到 20–28 GB/s；NCCL 大消息约 22 GB/s，4 MB 时 Ring_Simple 最高 28 GB/s。
- 多线程下发、CUDA Graph、常驻 kernel（mini-NCCL）都还没有尝试。

## 5. 参考实现的设计要点（剧透：用来判断用户的实现，不要直接给代码）

- **r1**：其他卡在各自的 stream 上把整份 buf 拷到 GPU0 的 tmp 第 g−1 段 → 全体同步 → GPU0 一个 kernel 把 p−1 段加上 → 3 次广播放在 GPU0 的同一个 stream 上 → 全体同步。共 2 次全体同步。
- **r2**：RS 每一步：所有拷贝（发送方 stream，写 next 的 tmp）→ 全体同步 → 所有加法（接收方 stream）→ 全体同步。AG 每一步：所有拷贝 → 全体同步。共 3(p−1) = 9 次。也可以把拷贝发在接收方的 stream 上，用 stream 顺序省掉一次同步；两种做法都合理，P4(c) 就是在问这个。
- **r3**：每卡一个 stream，每卡两个可复用的 event：`full[g]`（g 的拷贝之后在 s[g] 上 record）和 `freed[g]`（g 的加法之后在 s[g] 上 record）。
  - RS 每一步分两个循环。发送循环：s[g] 等 `freed[next]` → 拷贝 → record `full[g]`。接收循环：s[next] 等 `full[g]` → 加法 → record `freed[next]`。依赖②靠同一 stream 的顺序保证。
  - AG 每一步：发送循环做拷贝 + record `full[g]`；接收循环让 s[next] 等 `full[g]`。这次等待同时提供了“转发的数据依赖”和“收尾时的 stream 语义”。
  - 先发送、后接收的循环顺序，保证了 event 复用时每次 wait 都等到正确的那一次 record。
- **AG 直接覆盖的安全性**（P5(b) 的答案）：g 在 AG 第 s 步写 next 的第 c = (g+1−s) 块。这份数据正是从“next 在 RS 第 s 步发出的那一份”沿环一路加、一路转发过来的：next 发(RS s) → … → g−s 加完(RS p−2) → g−s 发(AG 0) → … → g 发(AG s)。链上每一环都是显式依赖，所以 next 对这一块的读（RS s）和写（RS s−1）都已经完成。
- **不死锁的理由**：单线程按顺序下发，每个 wait 等的都是 host 上更早发出的那次 record（或从未 record → 立即满足），所以依赖图里不可能有环。
- **r4**：每卡一个发送 stream `cs[g]`，主 stream 做加法。
  - 按槽位复用的 event：`full[g][j]`（cs[g]）、`freed[g][j]`（s[g]）。按（步, 小块）编号的 event：`ready[g][s·K+k]`（RS 加完，s[g]）、`sent[g][s·K+k]`（AG 拷完，cs[g]）。
  - 每条边一个序号 `seq[g]`，跨调用保留，槽号 = seq % S。
  - 开头 fork：在 s[g] 上 record，cs[g] 等它。RS (s,k)：cs[g] 等 `freed[next][j]`，s > 0 时再等 `ready[g][(s−1)K+k]` → 拷到 next 的槽 j → record `full[g][j]`；s[next] 等 `full[g][j]` → 加法 → record `freed[next][j]` 和 `ready[next][s·K+k]`。AG (s,k)：cs[g] 等 `ready[g][(p−2)K+k]`（s = 0）或 `sent[prev][(s−1)K+k]` → 写 next 的 buf → record `sent[g][s·K+k]`。
  - 收尾 join：s[g] 等 cs[g] 上 record 的 join event，以及 `sent[prev][(p−2)K+k]`（全部 k）。
  - 常见坑：同一轮循环里先 record 后 wait 同一个复用的 event 会导致隐性串行，结果正确但流水被串成一条线。
- **r5**：每卡 `d2h[g]`、`h2d[g]` 两个辅助 stream；每条边一块 host 槽 `stage[g]`（S 个槽，每槽一个小块），默认放在发送方的 NUMA 节点。
  - event：`staged[g][j]`（d2h[g]）、`landed[g][j]`（h2d[g]：从 prev 的 host 槽 j 读完，表示 host 槽可以复用、RS 时 tmp 槽已满）、`freed[g][j]`（s[g]）、`ready[g][·]`（s[g]）、`recvd[g][·]`（h2d[g]）。
  - RS：D2H 等 `landed[next][j]`（以及 ready）→ H2D 等 `staged[g][j]` 和 `freed[next][j]` → 加法等 `landed[next][j]`。AG：H2D 直接写 next 的 buf，并 record `recvd`；转发时等 `recvd[g][(s−1)K+k]`。收尾 join 两个辅助 stream。
- **校验**：参考实现所有版本在 1K–4M、`--check-reps 3` 下 `#wrong = 0`。

## 6. 辅导建议

- 按 HANDOUT 的顺序推进：P0 → P1 → P2 → P3 → … 每实现一个版本先跑 `make check`。
- 用户的结果让人意外时，先对照第 4 节：例如“r3 小消息不比 r2 快”“r4 小消息极慢”“4 条边并发只有约 15 GB/s”“NCCL 在本框架里的数字和 nccl-tests 不同”，都是这台机器上的真实现象。
- 常见 bug 及排查方法：
  - 负数取模：用 `mod()`。
  - event 和 stream 不在同一张卡：`cudaEventRecord` 报错。
  - event 复用导致隐性串行：表现为 issue 很小但 time 很大，或者流水线没有加速。
  - 忘记 fork/join：辅助 stream 在初始化 kernel 完成前就读了 buf，单次校验可能碰巧通过，`--check-reps 3` 才会暴露。
  - 换尺寸时槽位布局变化：框架每个尺寸前都会同步，所以只会在用户自己改框架时出现。
- 诊断工具：issue 列（host 瓶颈）、nsys（HANDOUT P5 的提示里有命令和 sqlite 表名）、CPU 模拟器（验证下标）、`--check-reps`（跨调用的竞争）。
- 用户问理论问题时，结合 α-β 模型和第 4 节的数据回答；问实现问题时先给提示，用户坚持要代码时再给。

## 7. 待办

- 框架和实验指导都已完成，没有遗留的开发任务。
- 仓库还没有 git commit（用户没有要求过）。
- 用户接下来要做的是从 HANDOUT 的 P0 开始做实验。
