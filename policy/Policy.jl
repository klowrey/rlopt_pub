__precompile__()

module Policy

# POLICY module; provides generic functions like gradloglikelihood, etc
# with Gaussian Linear Policy 
# and Gaussian Radial Basis Function Policy
# types

using JLD2
using ReverseDiff
using Flux
using Flux: TrackedVector
using Distributions
using Distributed
using Statistics

using LinearAlgebra
using Common: copymodel!, minibatch

abstract type AbstractPolicy end

struct GLP{T<:AbstractFloat} <: AbstractPolicy
    n::Integer
    m::Integer
    nparam::Integer

    theta::AbstractVector{T}

    model::Flux.Chain
    ls::TrackedVector

    scratch::AbstractVector{Tuple}

    function GLP{T}(obsspace::Integer, actspace::Integer) where T<:AbstractFloat
        n = obsspace
        m = actspace

        model = Chain(Dense(n, m; initW=zeros))
        ls = param(zeros(T, m))

        d = sum(length.(Tracker.data.(Flux.params(model)))) + m

        theta = zeros(T, d)
        copymodel!(theta, model)
        theta[d-m+1:end] = ls.data
        return new(n,m,d,theta,model,ls,
                   [(deepcopy(model),deepcopy(ls)) for i=1:Threads.nthreads()])
    end
end

struct NN{T<:AbstractFloat} <: AbstractPolicy
    n::Integer
    m::Integer
    nparam::Integer
    nhidden::Integer

    theta::AbstractVector{T}

    model::Flux.Chain
    ls::TrackedVector

    scratch::AbstractVector{Tuple}

    function NN{T}(obsspace::Integer, actspace::Integer,
                   nhidden::Integer; init=Flux.glorot_normal) where T<:AbstractFloat
        #@assert nlayers == 2
        n = obsspace
        m = actspace
        nh = nhidden
        #nlay = nlayers - 1 # skim off top layer

        model = Chain(Dense(n, nh, tanh; initW=init),
                      Dense(nh, nh, tanh; initW=init),
                      Dense(nh, m; initW=init)) #|> gpu
        ls = param(zeros(T, m))

        d = sum(length.(Tracker.data.(Flux.params(model)))) + m

        theta = zeros(T, d)
        copymodel!(theta, model)
        nn = new(n, m, d, nhidden, theta, model, ls,
                 [(deepcopy(model),deepcopy(ls)) for i=1:Threads.nthreads()])

        return nn
    end
end

################################################################# Create Policy

function save_pol(p::AbstractPolicy, skip::Int, modelname::String, filename::String)
    jldopen(filename, "w") do file
        file["n"] = p.n
        file["m"] = p.m
        file["theta"] = p.theta
        file["skip"] = skip
        file["model"] = modelname
        if :nhidden in fieldnames(typeof(p))
            file["nhidden"] = p.nhidden
        end
    end
end

function loadpolicy(filename::String)
    if isdir(filename)
        filename *= "/policy.jld"
    end
    d = load(filename)
    if "nhidden" in keys(d)
        p = NN{Float64}(d["n"], d["m"], d["nhidden"])
    else
        p = GLP{Float64}(d["n"], d["m"])
    end
    updatetheta(p, d["theta"])
    return p, d["skip"], d["model"]
end

############################################################## Policy Functions

function getmean!(c::AbstractVector{T}, p::AbstractPolicy,
                  features::AbstractVector{T}) where T<:AbstractFloat
    c .= p.model(features).data
end

# noise is in c vector already
function getaction!(c::AbstractVector{T}, p::AbstractPolicy,
                    features::AbstractVector{T}) where T<:AbstractFloat
    c .*= exp.(p.ls.data)
    c .+= p.model(features).data
end

function getls(p::AbstractPolicy)
    return Tracker.data.(Flux.params(p.ls))[1]
end

function updatetheta(p::AbstractPolicy, newtheta::AbstractVector{T}) where T<:AbstractFloat
    p.theta[:] = newtheta

    copymodel!(p.model, p.theta) # copy to model

    for s in p.scratch
        m, ls = s
        copymodel!(m, p.theta)
        ls.data .= p.ls.data 
    end
end

#################################################################### likelihood

function loglikelihood(p::AbstractPolicy,
                       features::AbstractArray{T},
                       actions::AbstractArray{T}) where T<:AbstractFloat
    loglikelihood(p.model, p.ls, features, actions)
end

function loglikelihood(model::Flux.Chain, ls,
                       features::AbstractMatrix{T},
                       actions::AbstractMatrix{T}) where T<:AbstractFloat
    m = length(ls.data)
    terms = -0.5*m*log(2pi) - sum(ls)

    mu = model(features)

    zs = ((actions-mu) ./ exp.(ls)).^2
    ll = sum(zs, 1)

    return ll.*-0.5 .+ terms
end

function loglikelihood(model::Flux.Chain, ls,
                       features::AbstractVector{T},
                       actions::AbstractVector{T}) where T<:AbstractFloat
    m = length(ls.data)
    terms = -0.5*m*log(2pi) - sum(ls)

    mu = model(features)

    zs = ((actions-mu) ./ exp.(ls)).^2
    ll = sum(zs)

    return ll.*-0.5 .+ terms
end

function gradLL_single!(gradll::AbstractMatrix{T},
                        p::AbstractPolicy,
                        features::AbstractMatrix{T},
                        actions::AbstractMatrix{T}) where T<:AbstractFloat
    N = size(features, 2) # N = T*K
    ff = zeros(T, p.n)
    aa = zeros(T, p.m)

    for i=1:N
        ff .= features[:,i] # implicit conversion and memory saving
        aa .= actions[:,i]

        Flux.back!(loglikelihood(p, ff, aa))

        copygrad_zero!(@view(gradll[:,i]), p.model)

        gradll[end-p.m+1:end,i] = p.ls.grad
        p.ls.grad .= 0.0
    end
end

function gradLL!(gradll::AbstractMatrix{T},
                 p::AbstractPolicy,
                 features::AbstractMatrix{T},
                 actions::AbstractMatrix{T}) where T<:AbstractFloat
    N = size(features, 2) # N = T*K

    LinearAlgebra.BLAS.set_num_threads(1)
    nthread = min(N, Threads.nthreads())
    Threads.@threads for tid=1:nthread
        #for tid=1:nthread
        thread_range = Distributed.splitrange(N, nthread)[tid]
        ff = zeros(T, p.n)
        aa = zeros(T, p.m)

        model, ls = p.scratch[tid]
        copygrad_zero!(@view(gradll[:,1]), model) # to zero out gradients
        ls.grad .= 0.0

        for i=thread_range
            ff .= features[:,i] # implicit conversion and memory saving
            aa .= actions[:,i]

            Flux.back!(loglikelihood(model, ls, ff, aa))

            copygrad_zero!(@view(gradll[:,i]), model)
            gradll[end-p.m+1:end,i] = ls.grad
            ls.grad .= 0.0
        end
    end
end

# for when data is in form Matrix[data, nsamples]
function clone(p::AbstractPolicy,
               features::AbstractMatrix{T},
               actions::AbstractMatrix{T};
               batch=16, epochs=20,
               opt=ADAM(vcat(Flux.params(p.model),
                             Flux.params(p.ls))) ) where T<:AbstractFloat

    #opt = ADAM(vcat(Flux.params(p.model), Flux.params(p.ls)))
    loss(f,a) = -mean(loglikelihood(p.model, p.ls, f, a))

    N = size(features, 2)
    nbatch = ceil(Int, N/batch)

    testidx = shuffle(1:floor(Int, N/10))
    #evalcb() = loss(features[:,testidx], actions[:,testidx])
    evalcb() = norm(loss(features[:,testidx], actions[:,testidx])) / norm(actions)

    lastscore = evalcb()
    for e=1:epochs
        rndidx = Flux.chunk(shuffle(1:N), nbatch)
        datas = [ (features[:,i], actions[:,i]) for i in rndidx ]

        Flux.train!(loss, datas, opt)

        #score = evalcb()
        #println("score: $(score)")
        #if score > lastscore
        #   println("Breaking cloning at epoch $e.")
        #   break
        #end
        #lastscore = score
    end
    println("score: $(evalcb())")

    # distribut results
    copymodel!(p.theta, p.model)
    updatetheta(p, p.theta)
end

# for when you have an Array{Array{}} type data store
function clone(p::AbstractPolicy,
               features::AbstractVector{T},
               actions::AbstractVector{T};
               batch=16, epochs=20) where T<:AbstractVector

    opt = ADAM(vcat(Flux.params(p.model), Flux.params(p.ls)))
    function loss(f,a)
        ll = 0.0
        for i=1:length(f)
            ll += loglikelihood(p.model, p.ls, f[i], a[i])
        end
        return -ll / length(f)
    end

    N = length(features)
    #nbatch = ceil(Int, N/batch)

    testidx = shuffle(1:floor(Int, N/10))
    evalcb() = loss(features[testidx], actions[testidx])
    #evalcb() = norm(loss(features[testidx], actions[testidx])) / norm(actions)

    lastscore = evalcb()
    for e=1:epochs
        rndidx = minibatch(N, batch) #Flux.chunk(shuffle(1:N), nbatch)
        datas = [ (features[i], actions[i]) for i in rndidx ]

        Flux.train!(loss, datas, opt)

        score = evalcb()
        if score > lastscore
            println("Breaking cloning at epoch $e.")
            break
        end
        lastscore = score
    end

    copymodel!(p.theta, p.model)
    updatetheta(p, p.theta)
end


######################################################################### utils

function copygrad_zero!(dst::AbstractVector, src::Flux.Chain)
    i = 1
    for l in Tracker.grad.(Flux.params(src))
        r = i:(i+length(l)-1)
        dst[r] = l
        i += length(l)
        l .= 0.0
    end
end

#function copymodel!(p::AbstractPolicy, src::AbstractVector)
#   copymodel!(p.model, src)
#   p.ls.data[:] = src[end-p.m+1:end]
#end

# API:
# each policy needs the following functions:
# getmean!     : mean policy output c = Policy(input)
# getaction!   : policy output with noise stored in c
#              : c = Policy(input) + c * exp(logstd)
# gradLL!      : gradient of loglikelihood function wrt to policy params
# split_theta! : policy is stored as a vector; splits vector into usable chunks
# updatetheta  : updates vector of parameters (if different than copy)

end # module Policy
