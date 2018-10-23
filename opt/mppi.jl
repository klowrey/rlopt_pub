
using MuJoCo
using LinearAlgebra

struct MPPIStrategy <: LearningStrategy
   costs::Vector{Float64}
   weights::Vector{Float64}

   Σ::Matrix{Float64}
   λ::Float64
   γ::Float64

   f::FunctionSet
   cfunc::Function
   rollout::Function
   c::Vector{Float64}
   o::Vector{Float64}
   function MPPIStrategy(model, sigma, lambda, gamma, funcset,
                         rollout=mjWrap.rollout2)
      dtype   = Float64
      K       = model.samples.K

      costs   = Vector{dtype}(undef, K)
      weights = Vector{dtype}(undef, K)

      new(costs, weights, sigma, lambda, gamma,
          funcset,
          (x...)->nothing,
          rollout,
          zeros(model.mjsys.nu), zeros(model.mjsys.ns))
   end
end

Base.show(io::IO, s::MPPIStrategy) = print(io, "MPPI")

function apply(f::Function, costs::Vector, states::Array{Float64,3})
   for k=1:size(states,3)
      costs[k] -= f(@view states[:,end,k])
   end
   #costs .-= f(states[:,end,:]) # block calculate
end

function setup!(s::MPPIStrategy, model::TrajOptModel)
   samples = model.samples

   # traj and samples both start from same place; send inital state to TrajOptModel
   if isnan(model.s0[1])
      @info("Setting initial state from experiment.")
      s.f.initstate!(model.mjsys, model.traj)
      model.s0 .= model.traj.state[:,1,1]
   else
      model.traj.state[:,1] .= model.s0
   end
   samples.state[:,1,:] .= model.s0

   push!(model.trace, :maxR => Vector{Float64}())
end

function update!(model::TrajOptModel, s::MPPIStrategy, iter, null)
   samples = model.samples
   T       = samples.T
   K       = samples.K
   mjsys   = model.mjsys

   #randn!(samples.ctrl)
   #rmul!(samples.ctrl, s.Σ[1]) # scale by one value
   #broadcast!(+, samples.ctrl, samples.ctrl, model.theta)
   #samples.ctrl[:,:,1] .= model.theta # have at least one thing to do a full theta rollout

   #samples.ctrl .= 0.0
   randn!( @view(samples.ctrl[:,(1+model.delay):end,:]) )
   rmul!(samples.ctrl, s.Σ[1]) # scale by one value
   broadcast!(+, samples.ctrl, samples.ctrl, model.theta)
   samples.ctrl[:,:,1] = model.theta # have at least one thing to do a full theta rollout

   mjWrap.limitcontrols(mjsys, samples)

   if iter == 1
      s.rollout(mjsys, samples,
                      s.cfunc, s.f.observe!, s.f.reward)
   else
      s.rollout(mjsys, samples,
                      s.cfunc, s.f.observe!, s.f.reward;
                      startT=1+model.delay)
   end
end

function applyweights!(theta::Matrix{Float64},
                       ctrl::Array{Float64,3}, weights::Vector{Float64})
   nu, T, K = size(ctrl)

   @inbounds for t=1:T
      #theta[:,t] = ctrl[:,t,:]*weights
      #@views A_mul_B!(theta[:,t], ctrl[:,t,:], weights)
      theta[:,t] .= 0.0
      for k=1:K
         for i=1:nu
            theta[i,t] += ctrl[i,t,k]*weights[k]
         end
      end
   end
end

function hook(s::MPPIStrategy, model::TrajOptModel, iter)

   LinearAlgebra.BLAS.set_num_threads(Threads.nthreads())
   BLAS.set_num_threads( 0 )

   mjsys   = model.mjsys
   samples = model.samples
   T       = samples.T
   K       = samples.K

   s.costs[:] = -1.0*Common.Rtau(samples.reward, s.γ)
   values = model.terminalvalue( samples.obs[:,end,:] )
   if values != nothing
      s.costs .-= values
   end

   baseline = minimum(s.costs)

   push!(model.trace[:stocR], mean(s.costs))
   push!(model.trace[:meanR], baseline)
   push!(model.trace[:maxR], maximum(s.costs))

   @. s.costs = exp((-(s.costs-baseline)/s.λ))
   η = sum(s.costs)
   @. s.weights = s.costs/η

   applyweights!(model.theta, samples.ctrl, s.weights)

   samples.ctrl[:,:,1] = model.theta # for visualization purposes

   if model.MPC && iter <= model.niter # get first controls, mj.step, store in model.traj
      traj    = model.traj

      # MPC: apply first controls to system; shift the rest around
      s.c[:] = model.theta[:,1]
      model.theta[:,1:end-1] .= @view model.theta[:,2:end]
      model.theta[:,end] .= 0.0

      # TODO 
      # TODO 
      # think about skip and how it should affect mpc mode
      # TODO 
      # TODO 
      mjWrap.reset(mjsys, model.s0, s.c, s.o)
      #for i=1:mjsys.skip
         mjWrap.step!(mjsys, s.c, model.s0, s.o) # get next state, next observation
      #end
      s.f.observe!(mjsys, model.s0, s.o) # arbitrary observation vector manipulation

      # advance our main trajectory
      traj.state[:,iter] = model.s0
      traj.ctrl[:,iter]  = s.c
      traj.obs[:,iter]   = s.o

      # update the mpc bundle of trajectories to start 'here'
      #for k=1:K
      #   samples.state[:,1,k] = model.s0
      #end
      samples.state[:,1+model.delay,:] .= model.s0
   end
end

function cleanup!(s::MPPIStrategy, model::TrajOptModel) # just rollout and store in traj
   if model.MPC == false 
      #info("Running rollout for main trajectory.")
      mjsys   = model.mjsys
      traj    = model.traj

      model.traj.state .= model.s0
      traj.ctrl[:,:,1] = model.theta
      s.rollout(mjsys, traj, s.cfunc, s.f.observe!, s.f.reward)
   end
end

