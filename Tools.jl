__precompile__()

module Tools
using LinearAlgebra
using Distributed

#################### utilities
function discount_sum(x::Vector{DT}, gamma::DT, terminal::DT) where DT<:AbstractFloat
   y = similar(x)
   runsum = terminal
   for t=length(x):-1:1
      runsum = x[t] + gamma*runsum
      y[t] = runsum
   end
   return y
end

function discount_sum!(y::Matrix{DT}, x::Matrix{DT},
                       t::Int, gamma::DT, terminal::DT) where DT<:AbstractFloat
   runsum = terminal
   for i=size(x, 1):-1:1
      runsum = x[i,t] + gamma*runsum
      y[i,t] = runsum
   end
end

function discount_sum!(y::Matrix{T}, x::Matrix{DT},
                       t::Int, gamma::DT, terminal::DT) where {T<:AbstractFloat,
                                                             DT<:AbstractFloat}
   runsum = terminal
   for i=size(x, 1):-1:1
      runsum = x[i,t] + gamma*runsum
      y[i,t] = runsum
   end
end

# reward: T x K
function compute_returns!(returns::Vector{DT},
                          rewards::Matrix{Float64},
                          gamma::Float64) where DT<:AbstractFloat
   T, K = size(rewards)
   mat_ret = reshape(returns, T, K)
   for t=1:K
      #mat_ret[:,t] = discount_sum(rewards[:,t], gamma, 0.0)
      discount_sum!(mat_ret, rewards, t, gamma, 0.0)
   end
   #return reshape(returns, T*K)
end

function compute_advantages(returns::Vector{Float64},
                            baseline::Vector{Float64},
                            obs::Matrix{Float64},
                            gamma::Float64)
   return returns - baseline
end

function computeGAEadvantages!(adv::Vector{DT},
                               baseline::Vector,
                               rewards::Matrix{Float64},
                               obs::Matrix{Float64},
                               gae::Float64,
                               gamma::Float64) where DT<:AbstractFloat
   T, K = size(rewards)
   tdsum = Array{DT}(undef, size(rewards))
   mat_adv = reshape(adv, T, K)

   nthread = min(K, Threads.nthreads()) # cant have more threads than data
   Threads.@threads for tid=1:nthread # WHY IS MT SO MUCH SLOWER??
      #nthread = 1
      #for tid=1:nthread # WHY IS MT SO MUCH SLOWER??
      thread_range = Distributed.splitrange(K, nthread)[tid]

      for t=thread_range
         base = view(baseline, (t-1)*T+1:t*T)
         for k=1:T-1
            tdsum[k,t] = rewards[k,t] + gamma * base[k+1] - base[k]
         end
         tdsum[T,t] = rewards[T,t] - base[T]
         #mat_adv[:,t] = discount_sum(tdsum[:,t], DT(gae*gamma), DT(0.0))
         discount_sum!(mat_adv, tdsum, t, DT(gae*gamma), DT(0.0))
      end
   end
end

# solver algorithms
function cpu_cg_solve(fim::Matrix{T}, # ((n+1)*m + m) X T*K
                      vpg::Vector{T}, # ((n+1)*m + m)
                      cg_iters::Integer=10,
                      reg::Float64=1e-4,
                      tol::Float64=1e-10) where T<:AbstractFloat
   # Initialize cg variables
   r = copy(vpg)
   p = copy(vpg)
   x = zeros(T, size(vpg))  # I want x to be same shape as vpg but full of zeros
   rdr = dot(vpg, vpg)  # outputs a scalar
   z = fim*p
   #z += p*reg 

   iters = 1
   for i=1:cg_iters
      v = rdr/dot(p, z)      # scalar

      x .+= v.*p
      r .-= v.*z

      rdr_new = dot(r, r)    # scalar
      ratio = rdr_new/rdr
      rdr = rdr_new

      iters = i
      if rdr < tol
         break   # this should break the for loop
      end

      p .*= ratio
      p .+= r

      #println(norm(z))
      mul!(z, fim, p)
   end
   println("    Used $iters cojugate-gradient iterations.");
   return x
end  

function hvp_cg_solve(hvpfim,
                      vpg::Vector{DT}, # ((n+1)*m + m)
                      cg_iters::Integer=10,
                      reg::Float64=1e-4,
                      tol::Float64=1e-10) where DT<:AbstractFloat
   # Initialize cg variables
   r = copy(vpg)
   p = copy(vpg)
   x = zeros(DT, size(vpg))
   rdr = dot(vpg, vpg)  # outputs a scalar
   z = zeros(DT, size(vpg))
   hvpfim(z, p)

   iters = 1
   for i=1:cg_iters
      v = rdr/dot(p, z)      # scalar

      x += v*p
      r -= v*z
      #@. x += v*p
      #@. r -= v*z

      rdr_new = dot(r, r)    # scalar
      ratio = rdr_new/rdr
      rdr = rdr_new

      iters = i
      if rdr < tol
         break   # this should break the for loop
      end

      @. p = r + ratio*p
      #p[:] .*= ratio
      #p[:] += r

      #println(norm(z))
      hvpfim(z, p)
   end
   println("    Used $iters HVP-cg iterations.");
   return x
end  


function par_cg_solver(fim::Matrix{T}, # ((n+1)*m + m) X T*K
                       vpg::Vector{DT}, # ((n+1)*m + m)
                       toworkers, #::Vector{RemoteChannel},
                       tomaster, #::RemoteChannel,
                       cg_iters::Integer=10,
                       reg::DT=1e-4) where {T<:AbstractFloat,
                                            DT<:AbstractFloat}
   put!(tomaster[myid()], vpg) # send vpg
   vpg[:] = take!(toworkers[myid()])

   z = fim*vpg
   put!(tomaster[myid()], z)
   p = similar(z)

   for i=1:cg_iters
      z[:] = take!( toworkers[myid()] )
      p[:] = take!( toworkers[myid()] )

      if isnan(z[1]) || i==cg_iters break end

      mul!(z, fim, p)
      put!(tomaster[myid()], z)
   end
   return p # p is NPG
end
function scatter(tomaster, val::Vector{T}) where T<:AbstractFloat
   @sync for w in workers()
      @async put!(tomaster[w], copy(val))
   end
end
function gather!(toworkers, val::Vector{T}) where T<:AbstractFloat
   val[:] .= T(0.0)
   for w in workers()
      val[:] += take!(toworkers[w])
   end
   val[:] /= nworkers()
end
function par_cg_manager(nparam::Integer,
                        toworkers,
                        tomaster,
                        cg_iters::Integer=10,
                        reg::T=1e-4,
                        tol::Float64=1e-10) where T<:AbstractFloat
   vpg = Array{T}(undef, nparam)
   z = Array{T}(undef, nparam)
   x = zeros(T, nparam)

   gather!(tomaster, vpg)
   scatter(toworkers, vpg)
   rdr = dot(vpg, vpg)  # outputs a scalar
   p = copy(vpg)
   r = copy(vpg)

   count = 0
   for i=1:cg_iters
      count += 1
      gather!(tomaster, z) # get Z vector
      v = rdr/dot(p, z)      # scalar

      x += v*p
      r -= v*z

      rdr_new = dot(r, r)    # scalar
      ratio = rdr_new/rdr
      rdr = rdr_new

      if rdr < tol || i == cg_iters
         z[1:5] .= NaN
         scatter(toworkers, z)
         scatter(toworkers, x) # send NPG
         break
      end

      p[:] .*= ratio
      p[:] += r

      scatter(toworkers, z)
      scatter(toworkers, p)
   end
   println("\tconverged in $count iterations.")
   return vpg, x # x == npg
end

# GPU solver
#=
function gpu_cg_solve{T<:AbstractFloat,
                      DT<:AbstractFloat}(fim::Knet.KnetArray{T,2}, # ((n+1)*m + m) X T*K
                                         vpg::Vector{DT}, # ((n+1)*m + m)
                                         cg_iters::Integer=10,
                                         reg::Float64=1e-4,
                                         tol::Float64=1e-10)
   # gradll (fat): num_params x T*K
   # vpg : vector num_params

   # Initialize cg variables
   if T == DT
      vpg32 = reshape(vpg, length(vpg), 1)
   else
      vpg32 = Array{T,2}(reshape(vpg, length(vpg), 1))
   end
   r = KnetArray{T}(vpg32)
   p = KnetArray{T}(vpg32)
   x = KnetArray{T}(zeros(size(vpg32)))
   rdr = dot(vpg, vpg) #(p' * p)[1] #dot(p, p)  # outputs a scalar
   z = fim * p

   @inbounds for i=1:cg_iters
      #v = rdr/dot(p, z)      # scalar
      v = rdr/((p'*z)[1])      # scalar

      x += v*p
      r -= v*z

      #rdr_new = dot(r, r)    # scalar
      rdr_new = (r' * r)[1] #dot(r, r)    # scalar
      ratio = rdr_new/rdr
      rdr = rdr_new

      if rdr < tol
         break   # this should break the for loop
      end

      p = r + ratio*p

      z[:] = fim*p
   end
   return x
end  
=#

end

