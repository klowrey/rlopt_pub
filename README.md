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

# Install

MuJoCo.jl is needed to use this, and must be cloned from source.

# Setup

Make sure you use more than one thread for Julia:
```bash
export JULIA_NUM_THREADS=4
```

or whatever your number of hardware CPUs is for your machine (avoid hyperthreads).

