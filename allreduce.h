// allreduce.h —— 框架（ring.cu）与你的实现（allreduce.cu）之间的接口
#pragma once

#include <string>

#include "common.h"

// 所有版本共享的上下文：由框架分配和初始化，你的函数直接使用
struct Ctx {
  int p = 0;                    // GPU 数（gpu01 上为 4）；环为 0→1→…→p−1→0
  float* buf[MAXG] = {};        // 每卡 n 个 float：输入，结束时为结果（in-place）
  float* tmp[MAXG] = {};        // 每卡的接收区（device 内存），容量 tmp_elems[g] 个 float
  size_t tmp_elems[MAXG] = {};
  cudaStream_t s[MAXG] = {};    // 每卡一个主 stream（cudaStreamNonBlocking）
};
// tmp 的容量保证（cap = 本次测试最大尺寸的元素数）：
//   每张卡 ≥ cap/p，且 ≥ S·cap/(p·K)；本次测试包含 r1 时，GPU0 ≥ (p−1)·cap。

// 命令行参数中与实现有关的部分
struct Params {
  int K = 8;                     // r4/r5：每块切成 K 个小块（--k）
  int S = 8;                     // r4/r5：FIFO 槽位数（--slots）
  size_t max_n = 0;              // 本次测试最大尺寸的元素数（setup 里可以据此分配资源）
  std::string r5_numa = "send";  // r5：host 槽位放在发送方 / 接收方的 NUMA 节点，或不绑定（send|recv|off）
};

// ---------------------------------------------------------------- 你要实现的函数（allreduce.cu）
// 语义：函数返回时，所有工作都已下发（不必等执行完）；调用者对所有卡同步之后，buf 就是结果。
//       r3/r4/r5 还要求：连续调用多次、中间不做 host 同步，结果也必须正确（框架用 --check-reps 检查）。
// setup：测试开始前调用一次，用来创建 event、辅助 stream、pinned 缓冲等。
// 还没实现的函数保留 NOT_IMPLEMENTED(...)，框架会跳过对应版本。
void ar_r1(Ctx& c, size_t n);              // HANDOUT P3
void ar_r2(Ctx& c, size_t n);              // HANDOUT P4
void r3_setup(Ctx& c, const Params& prm);  // HANDOUT P5
void ar_r3(Ctx& c, size_t n);
void r4_setup(Ctx& c, const Params& prm);  // HANDOUT P6
void ar_r4(Ctx& c, size_t n);
void r5_setup(Ctx& c, const Params& prm);  // HANDOUT P9（选做）
void ar_r5(Ctx& c, size_t n);
