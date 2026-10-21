# Makefile —— L6 手写 Ring AllReduce（框架提供）
#
#   make            编译 ring 和 probe
#   make check      小尺寸正确性检查（1K–4M，--check-reps 3），任何已实现版本 #wrong ≠ 0 时失败
#   make run        完整测试 1K–1G → results/allreduce.{txt,csv}
#   make ksweep     r4 的 K 扫描 → results/k_sweep.csv（KS="1 2 4 8 16 32" 可改）
#   make venv       建 Python 虚拟环境（matplotlib）
#   make plot       画图 → results/busbw.png、results/time.png（有 k_sweep.csv 时再画 k_sweep.png）
CUDA_HOME ?= /usr/local/cuda
NCCL_HOME ?= $(HOME)/nccl_local
NVCC      := $(CUDA_HOME)/bin/nvcc
ARCH      ?= sm_120
PY        := .venv/bin/python

NVFLAGS  := -O3 -std=c++17 -arch=$(ARCH) -lineinfo
NCCL_INC := -I$(NCCL_HOME)/include
# 系统里还有另一份 libnccl.so.2：rpath + LD_LIBRARY_PATH 保证加载的是 ~/nccl_local 里的 2.30.7
NCCL_LIB := -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath -Xlinker $(NCCL_HOME)/lib
export LD_LIBRARY_PATH := $(NCCL_HOME)/lib:$(LD_LIBRARY_PATH)

all: ring probe

ring: ring.cu allreduce.cu common.cu allreduce.h common.h
	$(NVCC) $(NVFLAGS) $(NCCL_INC) -o $@ ring.cu allreduce.cu common.cu $(NCCL_LIB)

probe: probe.cu common.cu common.h
	$(NVCC) $(NVFLAGS) -o $@ probe.cu common.cu

check: ring
	./ring --versions r1,r2,r3,r4,r5,nccl --min 1K --max 4M --warmup 1 --iters 3 --check-reps 3

run: ring
	mkdir -p results
	./ring --versions r1,r2,r3,r4,nccl --min 1K --max 1G --k 8 --csv results/allreduce.csv \
	  | tee results/allreduce.txt

KS ?= 1 2 4 8 16 32
ksweep: ring
	mkdir -p results
	rm -f results/k_sweep.csv
	for k in $(KS); do \
	  ./ring --versions r4 --min 64K --max 1G --k $$k --csv results/k_sweep_k$$k.csv \
	    > results/k_sweep_k$$k.txt || exit 1; \
	  if [ -f results/k_sweep.csv ]; then tail -n +2 results/k_sweep_k$$k.csv >> results/k_sweep.csv; \
	  else cp results/k_sweep_k$$k.csv results/k_sweep.csv; fi; \
	done

venv:
	uv venv .venv
	uv pip install --python $(PY) matplotlib numpy

plot:
	$(PY) plot.py results/allreduce.csv
	if [ -f results/k_sweep.csv ]; then $(PY) plot.py results/k_sweep.csv --k-sweep; fi

clean:
	rm -f ring probe

.PHONY: all check run ksweep venv plot clean
