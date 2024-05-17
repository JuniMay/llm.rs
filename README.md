# Implement Karpathy's llm.c in Safe Rust

Currently work in progress, some parts are very crappy.

This is an attempt to migrate Karpathy's [llm.c](https://github.com/karpathy/llm.c) to safe rust.

Most runtime costs of safe rust come from bounds checking, and for now, the performance of each
training step of the rust version (opt-level=3) is about 500ms slower than the C version (in -O3,
without openmp).

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
cargo build --release
./target/release/llm
```

THIS WILL BE REALLY SLOW, there are no parallelism and only runs on CPU.

To compare with the C version, change the optimization level in llm.c makefile to `-O3` and make
with `NO_OMP=1 make train_gpt2`:

```bash
NO_OMP=1 make train_gpt2
./train_gpt2
```

Haven't figured out how to compile with any equivalent of `-Ofast` in rust yet.

## Acknowledgments

I would like to express my sincere gratitude to ChatGPT for the assistance and support in this
project.
