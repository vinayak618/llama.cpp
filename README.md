
## llama2.cpp

Have you ever wanted to inference a baby [Llama 2](https://ai.meta.com/llama/) model in pure C++? No? Well, now you can!

<img src="assets/llama_cute.jpg" width="300" height="300">

With this code you can train the Llama 2 LLM architecture from scratch in PyTorch, then save the weights to a raw binary file, then load that into one ~simple 425-line C++ file ([run.cpp](run.cpp)) that inferences the model, simply in fp32 for now. On my cloud Linux devbox a dim 288 6-layer 6-head model (~15M params) inferences at ~100 tok/s in fp32, and about the same on my M1 MacBook Air. I was somewhat pleasantly surprised that one can run reasonably sized models (few ten million params) at highly interactive rates with an approach this simple.

Please note that this is just a weekend project: I took nanoGPT, tuned it to implement the Llama-2 architecture instead of GPT-2, and the meat of it was writing the C++ inference engine in [run.cpp](run.cpp). As such, this is not really meant to be a production-grade library right now.

Hat tip to [llama.cpp](https://github.com/ggerganov/llama.cpp) for inspiring this project. I wanted something super minimal so I chose to hard-code the llama-2 architecture, stick to fp32, and just roll one inference file of pure C++ with no dependencies.

## feel the magic

Let's just run a baby Llama 2 model in C++. You need a model checkpoint. Download this 15M parameter model I trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset (~58MB download) and place it into the default checkpoint directory `out`:

```bash
wget https://karpathy.ai/llama2c/model.bin -P out
```

(if that doesn't work try [google drive](https://drive.google.com/file/d/1aTimLdx3JktDXxcHySNrZJOOk8Vb1qBR/view?usp=share_link)). Compile and run the C++ code:

```bash
g++ -O3 -o run run.cpp -lm
./run out/model.bin
```

You'll see the text stream a sample. On my 1660 Ti Machine this runs at ~50 tokens/s, not bad for super naive fp32 single-threaded C++ code. See [performance](#performance) for compile flags that can significantly speed this up. Sample output:

*Once upon a time there was a gifted little boy called John. He was very intelligent and he had always been careful to think. 
One day, John was walking outside to play when he saw a big shadow. He asked a recommendation that he knew it, which was getting much closer and he could see a wound of light. 
John was confused, and decided to take a closer look. He was fascinated by the sight of the there he had come in across. Suddenly, he noticed a small light on the cloud that was glowing above the horizon. 
John knew he had to have courage to get closer, so he grabbed the light and followed it. The light smoymed him and his eyes made suddenly proud that he had made it back just in time. 
The moral of the story is that wisdom and using caution can be through unwilder situations. Listing to those advice out of them can always make them better, just like John had been told to be.
Once upon a time, there was a little girl named Lily. She had a sweet stuffed bear named Brownie. One day, Lily was playing with Brownie when she accidentally dropped him and
achieved tok/s: 56.255811*


## performance

*(NOTE: this guide is not great because I personally spend a lot of my time in Python land and don't have an amazing understanding of a lot of these features and flags. If someone does and is willing to help document and briefly describe some of these and their tradeoffs, I'd welcome a PR)*

There are many ways to potentially speed up this code depending on your system. Here we document a few together with a high-level guide on what they do. Here's again the default way to compile, but using -O3:

```bash
g++ -O3 -o run run.cpp -lm
```

-O3 includes optimizations that are expensive in terms of compile time and memory usage. Including vectorization, loop unrolling, and predicting branches. Here's a few more to try.

`-Ofast` Run additional optimizations which may break compliance with the C/IEEE specifications, in addition to `-O3`. See [the GCC docs](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html) for more information.

`-ffast-math` breaks IEEE compliance, e.g. allowing reordering of operations, disables a bunch of checks for e.g. NaNs (assuming they don't happen), enables reciprocal approximations, disables signed zero, etc. However, there is a good reason to be suspicious of this setting, one good writeup is here: ["Beware of fast-math"](https://simonbyrne.github.io/notes/fastmath/).

`-funsafe-math-optimizations` a more limited form of -ffast-math, that still breaks IEEE compliance but doesn't have all of the numeric/error handling changes from `-ffasth-math`. See [the GCC docs](https://gcc.gnu.org/wiki/FloatingPointMath) for more information.

`-march=native` Compile the program to use the architecture of the machine you're compiling on rather than a more generic CPU. This may enable additional optimizations and hardware-specific tuning such as improved vector instructions/width.

Putting a few of these together, the fastest throughput I saw so far on my MacBook Air (M1) is with:

```bash
g++ -Ofast -o run run.cpp -lm
```

Also, I saw someone report higher throughput replacing `gcc` with `clang`.

**OpenMP** Big improvements can also be achieved by compiling with OpenMP, which "activates" the `#pragma omp parallel for` inside the matmul. You can compile e.g. like so:

```bash
clang -Ofast -fopenmp -march=native run.c  -lm  -o run
```

(I believe you can swap clang/gcc, and may try to leave out -march=native). Then when you run inference, make sure to use OpenMP flags to set the number of threads, e.g.:

```bash
OMP_NUM_THREADS=4 ./run out/model.bin
```
