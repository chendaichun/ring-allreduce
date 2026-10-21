// common.h —— L6 框架提供的公共工具（一般不需要修改）
//
//   CK(...)                              CUDA 错误检查：出错时打印位置并退出
//   MAXG                                 框架支持的最大 GPU 数
//   mod(a, p)                            非负取模：环上的下标请写 mod(g - s, p)
//   now_us()                             host 端时间（微秒）
//   sync_devices(p)                      对前 p 张卡逐一 cudaDeviceSynchronize
//   launch_init / launch_add             可验证的初始化 kernel、加法 kernel（在指定 stream 上启动）
//   launch_spin / launch_tiny            让 GPU 空转一段时间 / 几乎为空的 kernel（P1 测量用）
//   gpu_numa_node / host_alloc_on_node   NUMA 小工具（P1(e)、r5 用）
//   NOT_IMPLEMENTED(msg)                 “尚未实现”：框架捕获后跳过该版本
#pragma once

#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#define CK(x)                                                                         \
  do {                                                                                \
    cudaError_t e_ = (x);                                                             \
    if (e_ != cudaSuccess) {                                                          \
      fprintf(stderr, "CUDA 错误 %s @ %s:%d: %s\n", cudaGetErrorString(e_), __FILE__, \
              __LINE__, #x);                                                          \
      exit(1);                                                                        \
    }                                                                                 \
  } while (0)

constexpr int MAXG = 8;

// C++ 的 % 对负数返回负值，例如 (0 - 1) % 4 == -1；环上的下标请用 mod
inline int mod(int a, int p) { return ((a % p) + p) % p; }

inline double now_us() {
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// 对前 p 张卡逐一 cudaDeviceSynchronize：覆盖该卡上的所有 stream（包括驱动内部的中转操作）
inline void sync_devices(int p) {
  for (int g = 0; g < p; ++g) {
    CK(cudaSetDevice(g));
    CK(cudaDeviceSynchronize());
  }
}

// 还没实现的函数先写 NOT_IMPLEMENTED("...")；框架捕获后会跳过这个版本，并打印这条信息
struct NotImplemented : std::runtime_error {
  using std::runtime_error::runtime_error;
};
#define NOT_IMPLEMENTED(msg) throw NotImplemented(msg)

// ---------------------------------------------------------------- kernel（实现在 common.cu）
// 调用前先 cudaSetDevice 到数据所在的卡；stream 必须属于这张卡
void launch_init(float* buf, int g, size_t n, cudaStream_t st);           // buf[i] = g + 1 + (i % 7)
void launch_add(float* dst, const float* src, size_t n, cudaStream_t st);  // dst[i] += src[i]
void launch_spin(long long ns, cudaStream_t st);                           // 单线程空转 ns 纳秒
void launch_tiny(int* counter, cudaStream_t st);                           // 1 个 block，只做一次 atomicAdd

// ---------------------------------------------------------------- NUMA（实现在 common.cu）
// GPU 所在的 NUMA 节点（读 /sys/bus/pci/devices/<busid>/numa_node；读不到时返回 -1）
int gpu_numa_node(int dev);
// 在指定 NUMA 节点上分配 pinned host 内存（cudaHostAllocPortable）；node < 0 表示不绑定。
// 做法：临时把当前线程绑到该节点的 CPU 上再 cudaHostAlloc（锁页时按 first-touch 落在该节点），然后恢复。
void* host_alloc_on_node(size_t bytes, int node);
