// allreduce.cu —— 你的 AllReduce 实现（L6 的主体，全部由你来写）
//
// 框架通过 allreduce.h 里声明的函数调用这里的代码。每个函数在实现之前都是 NOT_IMPLEMENTED(...)，
// 框架会跳过该版本并打印提示；实现某个版本时，删掉那一行即可。
//
// 可以直接用的工具（common.h）：
//   CK(...)、mod(a, p)、now_us()、sync_devices(p)、launch_add(dst, src, n, stream)、
//   gpu_numa_node(dev)、host_alloc_on_node(bytes, node)
// Ctx：c.p、c.buf[g]、c.tmp[g]（容量 c.tmp_elems[g] 个 float）、c.s[g]（每卡一个主 stream）
//
// 调度（HANDOUT 2.2）：
//   ReduceScatter 第 s 步：GPU g 把第 (g − s) mod p 块发到 next 的 tmp，next 把它加到自己 buf 的同一块
//   AllGather     第 s 步：GPU g 把第 (g + 1 − s) mod p 块直接写进 next 的 buf 同一位置
// 写 CUDA 之前，建议先用一个 CPU 模拟器验证下标（HANDOUT P4 的提示）。
// 改完代码随时跑 make check。

#include "allreduce.h"

// ================================================================ r1 朴素（HANDOUT P3）
// 1. GPU g（g ≠ 0）把整份 buf[g] 拷到 GPU0 的 tmp 第 g−1 段（cudaMemcpyPeerAsync，用发送方的 stream）
// 2. 全体同步；GPU0 把 p−1 段加到 buf[0]（自己写一个 kernel，或者多次调用 launch_add）
// 3. GPU0 把结果拷回其余各卡；全体同步
void ar_r1(Ctx& c, size_t n) {
  (void)c, (void)n;
  NOT_IMPLEMENTED("r1 朴素实现：见 HANDOUT P3");
}

// ================================================================ r2 Ring + 全体同步（HANDOUT P4）
// 只用 host 同步（对所有卡的 stream 做 cudaStreamSynchronize）表达依赖，不使用 event。
// 想一想：一个 ReduceScatter 步内，“拷贝 → 加法”的依赖怎么保证？
void ar_r2(Ctx& c, size_t n) {
  (void)c, (void)n;
  NOT_IMPLEMENTED("r2 Ring + 全体同步：见 HANDOUT P4");
}

// ================================================================ r3 Ring + event（HANDOUT P5）
// 每卡一个 stream（c.s[g]），所有依赖都用 cudaEvent 表达，中途不做 host 同步：
//   ① next 的加法，要等 g 的拷贝完成
//   ② next 下一步要发出的块，要等它这一步的加法完成（同一个 stream 的顺序即可）
//   ③ g 往 next 的 tmp 写下一份之前，要等 next 用完上一份
//   另外还有 AllGather 的转发依赖，以及收尾时的 stream 语义（见 HANDOUT P5）
// event 要在“负责 record 它的那张卡”上创建：先 cudaSetDevice，再 cudaEventCreateWithFlags。
namespace r3 {
// TODO(P5)：r3 需要的状态放在这里，例如每卡的 event
}  // namespace r3

void r3_setup(Ctx& c, const Params& prm) {
  (void)c, (void)prm;
  // TODO(P5)：创建 r3 需要的 event（建议用 cudaEventDisableTiming）
}

void ar_r3(Ctx& c, size_t n) {
  (void)c, (void)n;
  NOT_IMPLEMENTED("r3 Ring + event：见 HANDOUT P5");
}

// ================================================================ r4 Ring + 流水线（HANDOUT P6）
// 每块切成 K 个小块；tmp 做成 S 个槽位的环形 FIFO；每条边 g→next 用一个跨调用单调递增的序号 seq，
// 第 seq 个小块放进槽 seq % S。每卡两个 stream：一个负责发送（拷贝），主 stream 负责加法。
// 小心：event 复用可能导致“结果正确但被串行化”（HANDOUT P6 的注意事项）。
namespace r4 {
// TODO(P6)：K、S、每卡的发送 stream、槽位 event、按（步, 小块）编号的 event、FIFO 序号……
}  // namespace r4

void r4_setup(Ctx& c, const Params& prm) {
  (void)c, (void)prm;
  // TODO(P6)：记下 prm.K、prm.S；在每张卡上创建发送 stream 和需要的 event
}

void ar_r4(Ctx& c, size_t n) {
  (void)c, (void)n;
  NOT_IMPLEMENTED("r4 Ring + 流水线：见 HANDOUT P6");
}

// ================================================================ r5 手动 pinned host 中转（选做，HANDOUT P9）
// 每个小块：D2H（发送方的 stream，写进这条边的 host 槽）→ H2D（接收方的另一个 stream）→ RS 时在接收方加法。
// host 槽可以用 host_alloc_on_node(bytes, node) 分配；放在哪个节点由 prm.r5_numa 决定。
namespace r5 {
// TODO(P9)：D2H / H2D stream、每条边的 host 槽、host 槽满/空的 event……
}  // namespace r5

void r5_setup(Ctx& c, const Params& prm) {
  (void)c, (void)prm;
  // TODO(P9)：创建 stream 和 event，分配 host 槽（大小可按 prm.max_n、prm.K、prm.S 计算）
}

void ar_r5(Ctx& c, size_t n) {
  (void)c, (void)n;
  NOT_IMPLEMENTED("r5 手动 pinned host 中转（选做）：见 HANDOUT P9");
}
