# Implement Karpathy's llm.c in Safe Rust

Currently work in progress, some parts are very crappy.

This is an attempt to migrate Karpathy's [llm.c](https://github.com/karpathy/llm.c) to safe rust. By
saying safe, I mean no `unsafe` is introduced in the code of this repository (without considering
the dependencies).

For now, the performance of each training step of the rust version (opt-level=3) is similar to the
C version (in -O3). Note that more parallelism has been added to the rust version than the C
version, so this is not actually a fair comparison.

I am working on improving the performance by using iterators and re-slicing tricks to avoid bounds
checking.

## Quick Start

Firstly, follow the CPU quick start steps in llm.c. Note the python scripts are under `scripts/` and
are the same as in llm.c.

```bash
pip install -r requirements.txt
python scripts/prepro_tinyshakespeare.py
python scripts/train_gpt2.py
```

Then just build and run with cargo:

```bash
cargo run --release
```

The rayon is used for parallelism, currently only matmul is parallelized and the attention layer is
not parallelized yet.

To compare with the C version, change the optimization level in llm.c makefile to `-O3` and make.

```bash
make train_gpt2
OMP_NUM_THREADS=12 ./train_gpt2
```

Haven't figured out how to compile with any equivalent of `-Ofast` in rust yet.

## Benchmark

No serious benchmarking has been done yet but the performance of the rust version is similar to
the C version for each training step on a MacBook M2 Max. Each step takes ~4.4s to finish.

There are more parallelism in the rust version than the C version, so the performance should be (at
least slightly) better.

## Acknowledgments

I would like to express my sincere gratitude to ChatGPT for the assistance and support in this
project.
