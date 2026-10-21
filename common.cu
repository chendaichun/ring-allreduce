// common.cu —— common.h 中工具函数的实现（框架提供，一般不需要修改）
#include "common.h"

#include <sched.h>

#include <algorithm>
#include <cctype>
#include <string>

// grid 上限：SM 数 × 4（gpu01 上 4 张卡型号相同，取当前卡的值即可）
static int max_blocks() {
  static const int v = [] {
    int dev = 0, sms = 0;
    CK(cudaGetDevice(&dev));
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    return sms > 0 ? sms * 4 : 1024;
  }();
  return v;
}

static int blocks_for(size_t n) {
  return (int)std::max<size_t>(1, std::min<size_t>((n + 255) / 256, (size_t)max_blocks()));
}

// ---------------------------------------------------------------- kernel
__global__ void init_kernel(float* buf, int g, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x)
    buf[i] = (float)(g + 1 + (int)(i % 7));
}

__global__ void add_kernel(float* __restrict__ dst, const float* __restrict__ src, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x)
    dst[i] += src[i];
}

// 用 %globaltimer（纳秒）计时的空转
__global__ void spin_kernel(long long ns) {
  unsigned long long t0, t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  } while ((long long)(t - t0) < ns);
}

__global__ void tiny_kernel(int* x) {
  if (threadIdx.x == 0) atomicAdd(x, 1);
}

void launch_init(float* buf, int g, size_t n, cudaStream_t st) {
  init_kernel<<<blocks_for(n), 256, 0, st>>>(buf, g, n);
  CK(cudaGetLastError());
}

void launch_add(float* dst, const float* src, size_t n, cudaStream_t st) {
  add_kernel<<<blocks_for(n), 256, 0, st>>>(dst, src, n);
  CK(cudaGetLastError());
}

void launch_spin(long long ns, cudaStream_t st) {
  spin_kernel<<<1, 1, 0, st>>>(ns);
  CK(cudaGetLastError());
}

void launch_tiny(int* counter, cudaStream_t st) {
  tiny_kernel<<<1, 32, 0, st>>>(counter);
  CK(cudaGetLastError());
}

// ---------------------------------------------------------------- NUMA
// 读 /sys/devices/system/node/nodeN/cpulist（形如 "0-15,32-47"），把当前线程绑到这些 CPU 上
static bool bind_thread_to_node(int node, cpu_set_t* saved) {
  std::string path = "/sys/devices/system/node/node" + std::to_string(node) + "/cpulist";
  FILE* f = fopen(path.c_str(), "r");
  if (!f) return false;
  char line[512] = {0};
  if (!fgets(line, sizeof line, f)) line[0] = 0;
  fclose(f);
  cpu_set_t set;
  CPU_ZERO(&set);
  for (char* q = line; *q && *q != '\n';) {
    int a = (int)strtol(q, &q, 10), b = a;
    if (*q == '-') b = (int)strtol(q + 1, &q, 10);
    for (int cpu = a; cpu <= b; ++cpu) CPU_SET(cpu, &set);
    if (*q == ',') ++q;
    else if (*q && *q != '\n') break;
  }
  sched_getaffinity(0, sizeof *saved, saved);
  return sched_setaffinity(0, sizeof set, &set) == 0;
}

int gpu_numa_node(int dev) {
  char bus[32];
  CK(cudaDeviceGetPCIBusId(bus, sizeof bus, dev));
  for (char* q = bus; *q; ++q) *q = (char)tolower(*q);
  std::string path = std::string("/sys/bus/pci/devices/") + bus + "/numa_node";
  FILE* f = fopen(path.c_str(), "r");
  int node = -1;
  if (f) {
    if (fscanf(f, "%d", &node) != 1) node = -1;
    fclose(f);
  }
  return node;
}

void* host_alloc_on_node(size_t bytes, int node) {
  cpu_set_t saved;
  const bool bound = node >= 0 && bind_thread_to_node(node, &saved);
  void* ptr = nullptr;
  CK(cudaHostAlloc(&ptr, bytes, cudaHostAllocPortable));
  if (bound) sched_setaffinity(0, sizeof saved, &saved);
  return ptr;
}
