__precompile__()

module Baseline
using Flux

using Common: minibatch, vecvectomat!, vecvectomat, copymodel, copymodel!

#using BSON: @save
using JLD2
using LinearAlgebra
using Distributed
using Random

abstract type AbstractBaseline end

struct Linear{DT<:AbstractFloat} <: AbstractBaseline
    nfeat::Int
    T::Int
    K::Int
    reg_coeff::DT
    coeff::AbstractVector{DT} # states x N

    # workspace
    featnd::Array{DT,3}
    A::AbstractMatrix{DT}
    bline::AbstractVector{DT}
    function Linear{DT}(reg::AbstractFloat, ns::Integer, T::Integer, K::Integer) where DT<:AbstractFloat
        nfeat = 2*ns+3+1
        featnd=zeros(DT, nfeat, T, K)
        A = zeros(DT, nfeat, nfeat)
        bline = zeros(DT, T*K)
        return new(nfeat, T, K, reg, zeros(DT, nfeat),
                   featnd, A, bline)
    end
end

struct Quadratic{DT<:AbstractFloat} <: AbstractBaseline
    nfeat::Int
    T::Int
    K::Int
    reg_coeff::DT
    coeff::AbstractVector{DT} # states x N

    # workspace
    featnd::Array{DT,3}
    A::AbstractMatrix{DT}
    bline::AbstractVector{DT}
    function Quadratic{DT}(reg::AbstractFloat, ns::Integer, T::Integer, K::Integer) where DT<:AbstractFloat
        nfeat = ns+convert(Int,ns*(ns+1)/2)+ns+ns+4+1
        featnd=zeros(DT, nfeat, T, K)
        gettimefeatures!(featnd)

        A = zeros(DT, nfeat, nfeat)
        bline = zeros(DT, T*K)
        return new(nfeat, T, K, reg, zeros(DT, nfeat),
                   featnd, A, bline)
    end
end

struct NN{DT<:AbstractFloat} <: AbstractBaseline
    nhidden::Int
    nfeat::Int
    T::Int
    K::Int

    features::Function
    loss::Function
    opt::Function
    epochs::Int

    #workspace
    featnd::Array{DT,2}
    m::Flux.Chain
    bline::AbstractVector{DT}
    function NN{DT}(ns::Integer, T::Integer, K::Integer;
                    nhidden=64,
                    epochs=3,
                    step=0.001,
                    bias=0.0, # l2 regularization is broken when bias vec is 0...
                    priorreg=false,
                    featurefunc=getNNfeaturesNoTime
                   ) where DT<:AbstractFloat

        nfeat = size(featurefunc(rand(ns,2)), 1)
        featnd = zeros(DT, nfeat, T*K)
        m = Chain(Dense(nfeat, nhidden, tanh),
                  #Dense(nhidden, nhidden, tanh),
        Dense(nhidden, nhidden, tanh),
        Dense(nhidden, 1, initb=(x)->fill(bias,x)) )
        #Dense(nhidden, 1, (x)->bias+x))

        #m[end].b.data .= bias
        m0 = deepcopy(params(m))

        bline = zeros(DT, T*K)
        #opt = ADAMW(params(m), step)

        if priorreg
            println("doing Prior function")
            my_vecnorm(m) = vecnorm(m.+0.00001)
            opt = ADAM(params(m), step)
            return new(nhidden, nfeat, T, K,
                       featurefunc,
                       (a,b)->Flux.mse(m(a), b)+sum(vecnorm, params(m)-m0.+0.00001)/length(b),
                       #(a,b)->sum(abs.(m(a)-b))/length(b)+sum(my_vecnorm, params(m)-m0)/length(b),
            opt,
            epochs,
            featnd, m, bline)
        else
            println("doing MSE Loss function")
            opt = ADAM(params(m), step)
            return new(nhidden, nfeat, T, K,
                       featurefunc,
                       (a,b)->Flux.mse(m(a), b), #+0.1*(sum(vecnorm, params(m).+0.00001)/length(b)),
                       #(a,b)->sum(abs.(m(a)-b))/length(b),
            opt,
            epochs,
            featnd, m, bline)
        end
    end
end

######################################################################## common

function gettimefeatures!(feat::Array{DT,3}) where DT<:AbstractFloat
    T    = size(feat, 2)
    K = size(feat, 3)
    al = convert(Array{DT}, linspace(1, T, T)') / DT(T)
    for t=1:K
        feat[end-4,:,t] = al
        feat[end-3,:,t] = al.^2
        feat[end-2,:,t] = al.^3
        feat[end-1,:,t] = al.^4
        feat[end,:,t]   = DT(1.0)
    end
end

function getlinearfeatures(obs::AbstractMatrix{Float64})
    T = size(obs,2)
    # TODO add clipping
    o = max.(obs, -10.0)
    o = min.(o, 10.0)
    al = linspace(1, T, T)' / T
    feat = vcat(o, o.^2, al, al.^2, al.^3, ones(1, T))
end

function getquadfeatures!(feat::Array{DT,3},
                          obs::AbstractMatrix{DT},
                          t::Integer) where DT<:AbstractFloat
    n = size(obs, 1)
    T = size(feat, 2)

    feat[1:n,:,t] = obs[:,(t-1)*T+1:t*T] #implicite conversion between 32, 64
    feat[1:n,:,t] = max.(feat[1:n,:,t], DT(-10.0))
    feat[1:n,:,t] = min.(feat[1:n,:,t], DT(10.0))
    feat[1:n,:,t] /= DT(10.0)

    qfsize = convert(Int, n*(n+1)/2)
    qf = view(feat, (n+1):(n+qfsize) , :, t) #Matrix{Float64}(qfsize, T)
    k = 1
    @inbounds for i=1:n
        for j=i:n
            for x=1:T
                qf[k,x] = feat[i,x,t].*feat[j,x,t]
            end
            k += 1
        end
    end

    i0 = n+qfsize
    i1 = i0+n
    @inbounds for j = 1:n
        for i=1:T
            feat[i0+j,i,t] = sin(pi*feat[j,i,t])
            feat[i1+j,i,t] = cos(pi*feat[j,i,t])
        end
    end
end

function getNNfeatures(obs::AbstractMatrix{Float64})
    T = size(obs,2)
    o = max.(obs, -10.0)
    o = min.(o, 10.0)
    al = linspace(1, T, T)' / 1000.0
    feat = vcat(o, o.^2, al, al.^2, al.^3, ones(1, T))
    #feat = vcat(o, al, al.^2, al.^3, ones(1, T))
end

function getNNfeaturesNoTime(obs::AbstractMatrix{Float64})
    T = size(obs,2)
    o = max.(obs, -10.0)
    o = min.(o, 10.0)
    feat = vcat(o, o.^2, sin.(o), cos.(o), ones(1, T))
end


function prefit!(b::AbstractBaseline, returns::AbstractVector{DT}) where DT<:AbstractFloat
    featmat = reshape(b.featnd, b.nfeat, b.K*b.T) 
    #b.A[:] = (featmat*featmat')
    mul!(b.A, featmat, transpose(featmat))

    return featmat*returns
end

# returns: T*K
function fit!(b::AbstractBaseline, returns::AbstractVector{DT}) where DT<:AbstractFloat

    target = prefit!(b, returns) # sets A matrix, returns target

    for i=1:b.nfeat # set diag
        b.A[i,i] += b.reg_coeff
    end
    #b.coeff[:] = b.A\(target)
    if isposdef(b.A)
        A_ldiv_B!(b.coeff, cholfact!(b.A), target) # needs A to be factorized
    else
        A_ldiv_B!(b.coeff, lufact!(convert(Array{Float64}, b.A)), target) # needs A to be factorized
    end
end

function fit!(b::NN, returns::AbstractVector{DT}) where DT<:AbstractFloat
    nsamples = length(returns)
    nbatch = 50

    X = b.featnd |> gpu
    Y = returns |> gpu

    rndidx = Flux.chunk(shuffle(1:nsamples), nbatch)

    datas = [ (X[:,i], Y[i]') for i in rndidx ]

    #opt = ADAM(params(b.m))
    #evalcb() = @show( norm(b.loss(b.featnd, returns')) / norm(returns) )
    evalcb() = @show( norm(b.loss(datas[1]...)) / norm(datas[1][2]) )
    evalcb(x, y) = norm(b.loss(x, y')) / norm(y)

    for i=1:b.epochs
        Flux.train!(b.loss, datas, b.opt)#, cb = Flux.throttle(evalcb, 1))
        #if evalcb(b.featnd, returns) < 0.5
        #if evalcb() < 0.5
        #   break
        #end
    end
end

function bellmanfit!(b::NN,
                     features::AbstractMatrix{DT},
                     nstep_features::AbstractMatrix{DT},
                     returns::AbstractVector{DT},
                     discount::Float64;
                     batch=16, epochs=20,
                     nbatch=ceil(Int, N/batch),
                     rndidx=minibatch(length(returns), batch)) where DT<:AbstractFloat
    N = length(returns)

    X = b.features(features) #|> gpu
    Y = returns #|> gpu
    Z = nstep_features #|> gpu

    #evalcb() = @show( norm(b.loss(datas[1]...)) / norm(datas[1][2]) )
    #evalcb(x, y) = norm(b.loss(x, y')) / norm(y)

    idx = 1:batch
    values = zeros(DT, batch)
    #for i=1:epochs
    for n=1:nbatch
        #idx = rndidx[n]
        values[:] = discount*Baseline.predict(b, Z[:,idx])
        data = [( @view( X[:,idx] ), (Y[idx]+values)' )]

        Flux.train!(b.loss, data, b.opt)

        idx += batch
    end
    #   rndidx = minibatch(N, batch)
    #end
end


######################################################################## linear

function predict!(b::Linear, obs::AbstractMatrix{T}) where T<:AbstractFloat
    nthread = min.(b.K, Threads.nthreads()) # cant have more threads than data
    Threads.@threads for tid=1:nthread
        #for tid=1:nthread
        thread_range = Distributed.splitrange(b.K, nthread)[tid]

        for t=thread_range
            b.featnd[:,:,t] = getlinearfeatures(obs[:,(t-1)*b.T+1:t*b.T])
        end
    end
    b.bline[:] = reshape(b.featnd, b.nfeat, b.K*b.T)'*b.coeff
end

##################################################################### quadratic
# TODO this is hacky and this whole module is a disgrace
function predict(b::Quadratic, obs::AbstractMatrix{DT}) where DT<:AbstractFloat
    nsamples = size(obs, 2)
    featnd=zeros(DT, b.nfeat, 1, nsamples)
    gettimefeatures!(featnd)
    for t=1:nsamples
        getquadfeatures!(featnd, obs, t)
    end
    values = reshape(featnd, b.nfeat, nsamples)'*b.coeff
    return values
end

function predict!(b::Quadratic, obs::AbstractMatrix{DT}) where DT<:AbstractFloat
    nthread = min.(b.K, Threads.nthreads()) # cant have more threads than data
    Threads.@threads for tid=1:nthread
        #for tid=1:nthread
        thread_range = Distributed.splitrange(b.K, nthread)[tid]

        for t=thread_range
            getquadfeatures!(b.featnd, obs, t)
        end
    end
    b.bline[:] = reshape(b.featnd, b.nfeat, b.K*b.T)'*b.coeff
end

##################################################################### Neural Network


function predict(b::NN, obs::AbstractMatrix{DT}) where DT<:AbstractFloat
    return b.m( b.features(obs) ).data[1,:]
end

function predict!(b::NN, obs::AbstractMatrix{DT}) where DT<:AbstractFloat
    b.featnd[:,:] = b.features(obs)
    b.bline .= b.m(b.featnd).data[1,:]
end

end # module
