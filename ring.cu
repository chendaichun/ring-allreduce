// ring.cu —— L6 基准测试框架（框架提供；一般不需要修改）
//
// 负责：命令行解析、打印环境、分配缓冲、调用各版本、正确性校验、计时、输出表格和 CSV。
// 你的实现写在 allreduce.cu；NCCL 基线（ar_nccl）在本文件里。
//
// 用法：./ring --versions r1,r2,r3,r4,nccl --min 1K --max 1G --k 8 [--slots 8]
//              [--warmup 5] [--iters 20] [--timing sync|batch] [--check-reps 1]
//              [--r5-numa send|recv|off] [--csv results/x.csv] [--ngpus 4]
//
// 对每个（尺寸, 版本）：
//   1. 校验：初始化 buf[g][i] = g + 1 + (i % 7) → 背靠背调用 check-reps 次（中间不做 host 同步）
//            → 全体同步 → 每张卡的结果与期望值逐元素比较，得到 #wrong
//   2. warmup：调用 warmup 次，每次之后全体同步
//   3. 计时（host 端）：
//        sync 模式（默认）：每次迭代 = 全体同步 → t0 → 调用一次 → t_issue → 全体同步 → t1，取 iters 次平均
//        batch 模式      ：全体同步 → t0 → 背靠背调用 iters 次 → 全体同步 → t1，再除以 iters（nccl-tests 的做法）
//      time = t1 − t0；issue = t_issue − t0 是 host 下发所用的时间（issue ≈ time 说明瓶颈在 host 端）
// 还没实现的版本（抛出 NotImplemented）会被跳过，并在输出里注明。
// 退出码：所有已实现版本的 #wrong 都为 0 时返回 0，否则返回 2（make check 依赖它）。

#include <nccl.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "allreduce.h"

#define NCK(x)                                                                        \
  do {                                                                                \
    ncclResult_t r_ = (x);                                                            \
    if (r_ != ncclSuccess) {                                                          \
      fprintf(stderr, "NCCL 错误 %s @ %s:%d: %s\n", ncclGetErrorString(r_), __FILE__, \
              __LINE__, #x);                                                          \
      exit(1);                                                                        \
    }                                                                                 \
  } while (0)

// ================================================================ NCCL 基线
// 同一进程里用 ncclCommInitAll 初始化 p 张卡；in-place，与你的版本使用同一组 stream、同一套校验和计时
static ncclComm_t g_comm[MAXG];

static void nccl_setup(const Ctx& c) {
  int devs[MAXG];
  for (int g = 0; g < c.p; ++g) devs[g] = g;
  NCK(ncclCommInitAll(g_comm, c.p, devs));
}

static void ar_nccl(Ctx& c, size_t n) {
  NCK(ncclGroupStart());
  for (int g = 0; g < c.p; ++g)
    NCK(ncclAllReduce(c.buf[g], c.buf[g], n, ncclFloat, ncclSum, g_comm[g], c.s[g]));
  NCK(ncclGroupEnd());
}

// ================================================================ 命令行
struct Options {
  std::vector<std::string> versions = {"r1", "r2", "r3", "r4", "nccl"};
  size_t min_bytes = 1024, max_bytes = 1ull << 30;
  int K = 8, S = 8;
  int warmup = 5, iters = 20;
  bool batch = false;
  int check_reps = 1;
  std::string r5_numa = "send";
  std::string csv;
  int ngpus = 0;
};

static size_t parse_size(const char* str) {
  char* end = nullptr;
  const double v = strtod(str, &end);
  size_t mult = 1;
  switch (toupper(*end)) {
    case 'K': mult = 1ull << 10; break;
    case 'M': mult = 1ull << 20; break;
    case 'G': mult = 1ull << 30; break;
    default: break;
  }
  return (size_t)(v * mult);
}

static std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  size_t a = 0;
  while (a <= s.size()) {
    size_t b = s.find(sep, a);
    if (b == std::string::npos) b = s.size();
    if (b > a) out.push_back(s.substr(a, b - a));
    a = b + 1;
  }
  return out;
}

static bool known_version(const std::string& v) {
  return v == "r1" || v == "r2" || v == "r3" || v == "r4" || v == "r5" || v == "nccl";
}

static void usage() {
  printf(
      "用法: ./ring [--versions r1,r2,r3,r4,r5,nccl] [--min 1K] [--max 1G] [--k 8] [--slots 8]\n"
      "             [--warmup 5] [--iters 20] [--timing sync|batch] [--check-reps 1]\n"
      "             [--r5-numa send|recv|off] [--csv 路径] [--ngpus 4]\n");
}

static Options parse_args(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto val = [&]() -> const char* {
      if (i + 1 >= argc) {
        usage();
        exit(1);
      }
      return argv[++i];
    };
    if (a == "--versions") o.versions = split(val(), ',');
    else if (a == "--min") o.min_bytes = parse_size(val());
    else if (a == "--max") o.max_bytes = parse_size(val());
    else if (a == "--k") o.K = atoi(val());
    else if (a == "--slots") o.S = atoi(val());
    else if (a == "--warmup") o.warmup = atoi(val());
    else if (a == "--iters") o.iters = atoi(val());
    else if (a == "--timing") o.batch = std::string(val()) == "batch";
    else if (a == "--check-reps") o.check_reps = atoi(val());
    else if (a == "--r5-numa") o.r5_numa = val();
    else if (a == "--csv") o.csv = val();
    else if (a == "--ngpus") o.ngpus = atoi(val());
    else {
      usage();
      exit(a == "-h" || a == "--help" ? 0 : 1);
    }
  }
  for (const auto& v : o.versions)
    if (!known_version(v)) {
      fprintf(stderr, "未知版本 %s（可选 r1,r2,r3,r4,r5,nccl）\n", v.c_str());
      exit(1);
    }
  if (o.K < 1 || o.S < 1 || o.iters < 1 || o.warmup < 0 || o.check_reps < 1 ||
      o.min_bytes < 4 || o.max_bytes < o.min_bytes) {
    usage();
    exit(1);
  }
  return o;
}

// ================================================================ 校验与计时
using RunFn = std::function<void(size_t)>;

struct Version {
  std::string name, desc;
  std::function<void()> setup;  // 可为空
  RunFn run;
  bool implemented = true;
};

struct Result {
  std::string ver;
  size_t bytes = 0;
  double us = 0, issue_us = 0, algbw = 0, busbw = 0;
  long long wrong = 0;
};

// 校验：初始化 → 背靠背调用 reps 次 → 逐元素比较。
// 一次 AllReduce 后每张卡应为 E[i] = Σ_g (g + 1 + i%7) = p(p+1)/2 + p·(i%7)；
// 连续 reps 次后应为 p^(reps−1)·E[i]（都是不大的整数，fp32 下精确）。返回所有卡上的错误元素总数。
static long long verify(Ctx& c, size_t n, int reps, const RunFn& run, std::vector<float>& host) {
  const int p = c.p;
  for (int g = 0; g < p; ++g) {
    CK(cudaSetDevice(g));
    launch_init(c.buf[g], g, n, c.s[g]);
  }
  sync_devices(p);
  for (int r = 0; r < reps; ++r) run(n);
  sync_devices(p);
  const double scale = std::pow((double)p, reps - 1);
  float expect[7];
  for (int m = 0; m < 7; ++m) expect[m] = (float)(scale * (p * (p + 1) / 2 + p * m));
  long long wrong = 0;
  for (int g = 0; g < p; ++g) {
    CK(cudaSetDevice(g));
    CK(cudaMemcpy(host.data(), c.buf[g], n * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; ++i) wrong += host[i] != expect[i % 7];
  }
  return wrong;
}

static double time_us(Ctx& c, size_t n, const RunFn& run, const Options& o, double* issue_us) {
  const int p = c.p;
  for (int w = 0; w < o.warmup; ++w) {
    run(n);
    sync_devices(p);
  }
  if (o.batch) {
    sync_devices(p);
    const double t0 = now_us();
    for (int it = 0; it < o.iters; ++it) run(n);
    *issue_us = (now_us() - t0) / o.iters;
    sync_devices(p);
    return (now_us() - t0) / o.iters;
  }
  double acc = 0, acc_issue = 0;
  for (int it = 0; it < o.iters; ++it) {
    sync_devices(p);
    const double t0 = now_us();
    run(n);
    acc_issue += now_us() - t0;
    sync_devices(p);
    acc += now_us() - t0;
  }
  *issue_us = acc_issue / o.iters;
  return acc / o.iters;
}

static void print_time(double us) {
  if (us < 1e4) printf("%10.2f", us);
  else if (us < 1e6) printf("%10.1f", us);
  else printf("%10.0f", us);
}

// ================================================================ main
int main(int argc, char** argv) {
  const Options o = parse_args(argc, argv);
  setvbuf(stdout, nullptr, _IOLBF, 0);

  int ndev = 0;
  CK(cudaGetDeviceCount(&ndev));
  const int p = std::min(o.ngpus > 0 ? o.ngpus : ndev, std::min(ndev, MAXG));
  if (p < 2) {
    fprintf(stderr, "至少需要 2 张 GPU\n");
    return 1;
  }
  auto want = [&](const char* v) {
    return std::find(o.versions.begin(), o.versions.end(), v) != o.versions.end();
  };

  // ---------------- 环境信息
  int drv = 0, rt = 0, nccl_ver = 0;
  CK(cudaDriverGetVersion(&drv));
  CK(cudaRuntimeGetVersion(&rt));
  NCK(ncclGetVersion(&nccl_ver));
  printf("# L6 Ring AllReduce 基准测试：float32 sum，in-place\n");
  printf("# CUDA driver %d，runtime %d，NCCL %d\n", drv, rt, nccl_ver);
  int p2p_pairs = 0;
  for (int g = 0; g < p; ++g) {
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, g));
    char bus[32];
    CK(cudaDeviceGetPCIBusId(bus, sizeof bus, g));
    printf("#  GPU%d  %s  [%s]  NUMA %d\n", g, prop.name, bus, gpu_numa_node(g));
    for (int h = 0; h < p; ++h) {
      int ok = 0;
      if (h != g) CK(cudaDeviceCanAccessPeer(&ok, g, h));
      p2p_pairs += ok;
    }
  }
  printf("# P2P：%d / %d 对可访问（0 表示 GPU 间拷贝都由驱动经 host 中转）\n", p2p_pairs,
         p * (p - 1));
  printf("# p = %d，环 0→1→…→%d→0；K = %d，slots = %d；warmup %d，iters %d，计时 %s，check-reps %d\n",
         p, p - 1, o.K, o.S, o.warmup, o.iters,
         o.batch ? "batch（背靠背）" : "sync（每次前后同步）", o.check_reps);

  // ---------------- 尺寸列表：每次 ×2，大小需能被 p × K × 4 字节整除（所有版本用同一组尺寸）
  const size_t align = (size_t)p * o.K * sizeof(float);
  std::vector<size_t> sizes;
  for (size_t b = o.min_bytes; b <= o.max_bytes; b *= 2) {
    if (b % align != 0) {
      printf("# 跳过 %zu B：不能被 p×K×4 = %zu 字节整除\n", b, align);
      continue;
    }
    sizes.push_back(b);
  }
  if (sizes.empty()) {
    fprintf(stderr, "没有可测的尺寸\n");
    return 1;
  }
  const size_t cap = sizes.back() / sizeof(float);  // 最大尺寸的元素数

  // ---------------- 分配缓冲（按最大尺寸分配一次，各尺寸复用）
  Ctx c;
  c.p = p;
  const size_t slot_elems = cap / ((size_t)p * o.K);
  for (int g = 0; g < p; ++g) {
    size_t t = std::max(cap / p, (size_t)o.S * slot_elems);
    if (g == 0 && want("r1")) t = std::max(t, (size_t)(p - 1) * cap);
    CK(cudaSetDevice(g));
    CK(cudaMalloc(&c.buf[g], cap * sizeof(float)));
    CK(cudaMalloc(&c.tmp[g], t * sizeof(float)));
    c.tmp_elems[g] = t;
    CK(cudaStreamCreateWithFlags(&c.s[g], cudaStreamNonBlocking));
  }
  Params prm;
  prm.K = o.K;
  prm.S = o.S;
  prm.max_n = cap;
  prm.r5_numa = o.r5_numa;

  // ---------------- 版本登记
  std::vector<Version> vers;
  for (const std::string& name : o.versions) {
    Version v;
    v.name = name;
    if (name == "r1") {
      v.desc = "朴素：gather 到 GPU0 → 累加 → broadcast";
      v.run = [&](size_t m) { ar_r1(c, m); };
    } else if (name == "r2") {
      v.desc = "Ring + 每阶段 host 全体同步";
      v.run = [&](size_t m) { ar_r2(c, m); };
    } else if (name == "r3") {
      v.desc = "Ring + cudaEvent 只和邻居同步";
      v.setup = [&] { r3_setup(c, prm); };
      v.run = [&](size_t m) { ar_r3(c, m); };
    } else if (name == "r4") {
      v.desc = "Ring + 流水线（K = " + std::to_string(o.K) + "，S = " + std::to_string(o.S) + "）";
      v.setup = [&] { r4_setup(c, prm); };
      v.run = [&](size_t m) { ar_r4(c, m); };
    } else if (name == "r5") {
      v.desc = "r4 的流水线 + 手动 pinned host 中转（NUMA = " + o.r5_numa + "）";
      v.setup = [&] { r5_setup(c, prm); };
      v.run = [&](size_t m) { ar_r5(c, m); };
    } else {
      v.desc = "ncclAllReduce（in-place，同一进程）";
      v.setup = [&] { nccl_setup(c); };
      v.run = [&](size_t m) { ar_nccl(c, m); };
    }
    vers.push_back(v);
  }
  for (auto& v : vers) {
    if (!v.setup) continue;
    try {
      v.setup();
    } catch (const NotImplemented& e) {
      v.implemented = false;
      printf("# 跳过 %s：%s\n", v.name.c_str(), e.what());
    }
  }
  sync_devices(p);

  // ---------------- 主循环：外层尺寸、内层版本（同一尺寸的各版本在时间上挨着测，减少环境波动的影响）
  std::vector<float> host(cap);
  std::vector<Result> results;
  for (size_t bytes : sizes) {
    const size_t n = bytes / sizeof(float);
    for (auto& v : vers) {
      if (!v.implemented) continue;
      Result r;
      r.ver = v.name;
      r.bytes = bytes;
      try {
        r.wrong = verify(c, n, o.check_reps, v.run, host);
      } catch (const NotImplemented& e) {
        v.implemented = false;
        printf("# 跳过 %s：%s\n", v.name.c_str(), e.what());
        sync_devices(p);
        continue;
      }
      r.us = time_us(c, n, v.run, o, &r.issue_us);
      r.algbw = bytes / r.us / 1e3;  // GB/s（1 GB = 1e9 B）
      r.busbw = r.algbw * 2.0 * (p - 1) / p;
      results.push_back(r);
      fprintf(stderr, "  [%-4s] %11zu B  %10.2f us（下发 %9.2f）  busbw %6.2f GB/s  #wrong %lld\n",
              v.name.c_str(), bytes, r.us, r.issue_us, r.busbw, r.wrong);
    }
  }

  auto find = [&](const std::string& v, size_t b) -> const Result* {
    for (const auto& r : results)
      if (r.ver == v && r.bytes == b) return &r;
    return nullptr;
  };
  std::vector<const Version*> done;  // 有结果的版本
  for (const auto& v : vers)
    if (v.implemented && find(v.name, sizes[0])) done.push_back(&v);

  // ---------------- 每个版本一张表（模仿 nccl-tests）
  long long total_wrong = 0;
  for (const Version* v : done) {
    printf("#\n# %s  %s\n", v->name.c_str(), v->desc.c_str());
    printf("#%11s %13s %10s %7s %7s %7s %10s\n", "size", "count", "time", "algbw", "busbw", "#wrong",
           "issue");
    printf("#%11s %13s %10s %7s %7s %7s %10s\n", "(B)", "(elements)", "(us)", "(GB/s)", "(GB/s)", "",
           "(us)");
    for (size_t b : sizes) {
      const Result* r = find(v->name, b);
      printf("%12zu %13zu ", b, b / sizeof(float));
      print_time(r->us);
      printf(" %7.2f %7.2f %7lld ", r->algbw, r->busbw, r->wrong);
      print_time(r->issue_us);
      printf("\n");
      total_wrong += r->wrong;
    }
  }

  // ---------------- 汇总表：各版本的 busbw、time 并排
  if (!done.empty()) {
    printf("#\n# 汇总 busbw (GB/s)\n#%11s", "size(B)");
    for (const Version* v : done) printf(" %9s", v->name.c_str());
    printf("\n");
    for (size_t b : sizes) {
      printf("%12zu", b);
      for (const Version* v : done) printf(" %9.2f", find(v->name, b)->busbw);
      printf("\n");
    }
    printf("#\n# 汇总 time (us)\n#%11s", "size(B)");
    for (const Version* v : done) printf(" %9s", v->name.c_str());
    printf("\n");
    for (size_t b : sizes) {
      printf("%12zu", b);
      for (const Version* v : done) printf(" %9.1f", find(v->name, b)->us);
      printf("\n");
    }
  }
  std::string todo;
  for (const auto& v : vers)
    if (!v.implemented) todo += (todo.empty() ? "" : ", ") + v.name;
  printf("#\n# 尚未实现（已跳过）：%s\n", todo.empty() ? "无" : todo.c_str());
  printf("# #wrong 总计：%lld %s\n", total_wrong, total_wrong == 0 ? "OK" : "<<<< 有错误");

  // ---------------- CSV（长表：每行一个版本 × 尺寸）
  if (!o.csv.empty()) {
    FILE* f = fopen(o.csv.c_str(), "w");
    if (!f) {
      perror(o.csv.c_str());
      return 1;
    }
    fprintf(f, "version,K,slots,timing,size_bytes,count,time_us,issue_us,algbw_GBps,busbw_GBps,"
               "wrong\n");
    for (const auto& r : results)
      fprintf(f, "%s,%d,%d,%s,%zu,%zu,%.3f,%.3f,%.4f,%.4f,%lld\n", r.ver.c_str(), o.K, o.S,
              o.batch ? "batch" : "sync", r.bytes, r.bytes / sizeof(float), r.us, r.issue_us,
              r.algbw, r.busbw, r.wrong);
    fclose(f);
    printf("# CSV 已写入 %s\n", o.csv.c_str());
  }

  if (want("nccl"))
    for (int g = 0; g < p; ++g) NCK(ncclCommDestroy(g_comm[g]));
  return total_wrong == 0 ? 0 : 2;
}
