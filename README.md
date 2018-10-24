# rlopt

To run this repo of highly experimental code
```bash
julia -O3 PGTrain.jl swimmer/swimmer.jl npg_test
julia -O3 OptTrain.jl swimmer/swimmer.jl opt_test
```

To visualize results
```julia
julia> include("paths.jl"); using ExpSim
julia> simulate("/tmp/npg_test") # press ',' or '.' to change visualization modes
julia> simulate("/tmp/opt_test")
```

# Structure

The general structure of the code is organized around experiments in their own directories (ant, hopper, humanoid, etc). The experiments create functions that return model / strategies that are used by [LearningStrategies.jl](https://github.com/JuliaML/LearningStrategies.jl) to learn according to either [trajectory optimization](https://arxiv.org/pdf/1707.02342.pdf) or [Natural Policy Gradients](https://papers.nips.cc/paper/2073-a-natural-policy-gradient.pdf). Top level code is under OptTrain.jl or PGTrain.jl, which use modules under opt/ or pg/, respectively. All experiment / model specific behavior is defined as per experiment directories: a primary file that includes a functions file; the visualization looks specifically for a \*functions.jl file for interactive policy rendering.

TODO:
Fix interactive policy rendering.

# Install

MuJoCo.jl is needed to use this, and must be cloned from source.

# Setup

Make sure you use more than one thread for Julia:
```bash
export JULIA_NUM_THREADS=4
```

or whatever your number of hardware CPUs is for your machine (avoid hyperthreads).

