// probe.cu —— P1：先认识你的“链路”（框架 + 一个示例；(b)–(e) 由你来写）
//
// 用法：./probe          跑全部小节
//       ./probe a        只跑 (a)；可以组合，例如 ./probe bd
//
// 已提供：缓冲与 stream 的分配、warmup、计时工具 measure()、(a) 的完整示例。
// 你要写：section_b() … section_e()。每节打印一张表（格式自定），再把表和回答写进 README。
// 可用的工具（common.h）：launch_spin（空转，制造“前序工作未完成”）、launch_tiny（测 launch 开销）、
//   gpu_numa_node、host_alloc_on_node（(e) 用）、sync_devices、now_us。

#include <string>
#include <vector>

#include "common.h"

constexpr int P = 4;                   // gpu01 上 4 张卡
constexpr size_t MAXB = 256ull << 20;  // 最大测 256 MB

static char* src[P];  // 每卡的源缓冲（MAXB 字节）
static char* dst[P];  // 每卡的目的缓冲（3 × MAXB 字节：扇入测试时 3 个发送方各写一段）
static char* host;    // pinned host 缓冲（2 × MAXB 字节，未绑定 NUMA）
static cudaStream_t st[P], st2[P];  // 每卡两个 stream

static const std::vector<size_t> kSizes = {1ull << 10,  64ull << 10, 1ull << 20,  4ull << 20,
                                           16ull << 20, 64ull << 20, 256ull << 20};

static int reps_for(size_t bytes) {
  return bytes >= (64ull << 20) ? 5 : (bytes >= (1ull << 20) ? 20 : 200);
}

static void print_size(size_t b) {
  if (b >= (1ull << 20)) printf("%6zu MB", b >> 20);
  else printf("%6zu KB", b >> 10);
}

// 计时工具：重复 R 次 { 全体同步 → setup() → t0 → op() → t1 → 全体同步 → t2 }，返回平均值
//   call  = t1 − t0：op 里的 API 调用让 host 阻塞了多久（“调用返回时间”）
//   total = t2 − t0：从开始调用到所有卡都完成（“完成时间”）
// setup() 不计时，可以用来制造前序工作（例如 launch_spin）。
struct Timing {
  double call_us, total_us;
};

template <class Setup, class Op>
static Timing measure(int R, Setup&& setup, Op&& op) {
  double call = 0, total = 0;
  for (int r = 0; r < R; ++r) {
    sync_devices(P);
    setup();
    const double t0 = now_us();
    op();
    const double t1 = now_us();
    sync_devices(P);
    const double t2 = now_us();
    call += t1 - t0;
    total += t2 - t0;
  }
  return {call / R, total / R};
}

// ---------------------------------------------------------------- (a) 单条拷贝（示例，已完成）
static void section_a() {
  printf("\n== (a) 单条 cudaMemcpyPeerAsync 0->1（用 GPU0 的 stream）\n");
  printf("      size   call(us)  total(us)    GB/s\n");
  for (size_t b : kSizes) {
    const Timing t = measure(
        reps_for(b), [] {},
        [&] {
          CK(cudaSetDevice(0));
          CK(cudaMemcpyPeerAsync(dst[1], 1, src[0], 0, b, st[0]));
        });
    print_size(b);
    printf(" %10.1f %10.1f %7.2f\n", t.call_us, t.total_us, b / t.total_us / 1e3);
  }
}

// ---------------------------------------------------------------- (b) 并发 —— TODO(P1b)
// 三种情况，每种都列出 size | total(us) | 每条边（或合计）GB/s：
//   (i)   ring 的 4 条边 g→g+1 同时拷贝，各用发送方的 stream st[g]（写到 dst[g+1]）
//   (ii)  GPU0 同时发 0→1（st[0]）和 0→2（st2[0]）
//   (iii) 1→0、2→0、3→0 同时扇入 GPU0（写到 dst[0] 的不同偏移：dst[0] + (g−1)·b）
// 提示：如果调用本身会阻塞 host，4 条拷贝的启动时刻是错开的——算带宽时想清楚分母。
static void section_b() { printf("\n== (b) TODO(P1b)：并发\n"); }

// ---------------------------------------------------------------- (c) 参考上限 —— TODO(P1c)
// 1. 每张卡 pinned 内存的 D2H、H2D 单向带宽（cudaMemcpyAsync + host 缓冲，用 256 MB）
// 2. 同一张卡 D2H（st[g]）与 H2D（st2[g]）同时进行时，每个方向的带宽（用 host 的两半，避免读写同一块）
// 3. 手动中转 0→host→1：D2H 完成后再 H2D（用 event 串起来），与 (a) 对比
static void section_c() { printf("\n== (c) TODO(P1c)：参考上限\n"); }

// ---------------------------------------------------------------- (d) 小开销 —— TODO(P1d)
// host 端单次开销（重复 N 次取平均）：kernel launch（launch_tiny）、cudaEventRecord、
//   cudaStreamWaitEvent（同卡 / 跨卡）、空 stream 的 cudaStreamSynchronize、1 KB 的 cudaMemcpyPeerAsync。
// GPU 端延迟：1 KB 拷贝沿 0→1→2→3→0… 传 H 跳，每跳等上一跳的 event。
//   如果“下发耗时 ≈ 执行耗时”，测到的是 host 开销：可以先 launch_spin 把 GPU0 的 stream 挡住，
//   趁这段时间把整条链下发完，再看链本身的执行时间。
// 阻塞测试：同一个 stream 上先 launch_spin(2 ms)，再只测一次拷贝调用本身的耗时。
static void section_d() { printf("\n== (d) TODO(P1d)：小开销\n"); }

// ---------------------------------------------------------------- (e) NUMA —— TODO(P1e)
// 4 张卡同时 D2H / H2D / 两者同时（64 MB），host 缓冲分别放在：
//   全在 node0、全在 node1、各自本地节点（gpu_numa_node(g)）、各自远端节点。
// 用 host_alloc_on_node(bytes, node) 为每张卡各分配一块缓冲。
static void section_e() { printf("\n== (e) TODO(P1e)：NUMA\n"); }

// ---------------------------------------------------------------- main
int main(int argc, char** argv) {
  const std::string sel = argc > 1 ? argv[1] : "abcde";
  int ndev = 0;
  CK(cudaGetDeviceCount(&ndev));
  if (ndev < P) {
    fprintf(stderr, "需要 %d 张 GPU，只找到 %d 张\n", P, ndev);
    return 1;
  }
  for (int g = 0; g < P; ++g) {
    CK(cudaSetDevice(g));
    CK(cudaMalloc(&src[g], MAXB));
    CK(cudaMalloc(&dst[g], 3 * MAXB));
    CK(cudaMemset(src[g], g, MAXB));
    CK(cudaStreamCreateWithFlags(&st[g], cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&st2[g], cudaStreamNonBlocking));
  }
  CK(cudaHostAlloc(&host, 2 * MAXB, cudaHostAllocPortable));
  sync_devices(P);

  // warmup：GPU 空闲时 PCIe 链路会降速，先让 4 条边各跑几次大拷贝
  for (int i = 0; i < 5; ++i)
    for (int g = 0; g < P; ++g) {
      CK(cudaSetDevice(g));
      CK(cudaMemcpyPeerAsync(dst[(g + 1) % P], (g + 1) % P, src[g], g, MAXB, st[g]));
    }
  sync_devices(P);

  if (sel.find('a') != std::string::npos) section_a();
  if (sel.find('b') != std::string::npos) section_b();
  if (sel.find('c') != std::string::npos) section_c();
  if (sel.find('d') != std::string::npos) section_d();
  if (sel.find('e') != std::string::npos) section_e();
  return 0;
}
