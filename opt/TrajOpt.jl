
__precompile__()

module TrajOpt

using mjWrap
using LearningStrategies
import LearningStrategies: setup!, update!, hook, cleanup!
export setup!, update!, hook, cleanup!
using ExpFunctions
using Common

export TrajOptModel
export MPPIStrategy
export ControlExp

using Random
using Statistics

struct TrajOptModel
   theta::Matrix{Float64} # parameters to improve
   mjsys::mjWrap.mjSet    # environment
   s0::Vector{Float64}    # initial / current state
   terminalvalue::Function 

   MPC::Bool

   # scratch stuff
   samples::TrajSamples
   traj::TrajSamples

   trace::Dict{Symbol,Vector{Float64}}

   delay::Int
   niter::Int

   function TrajOptModel(mjsys, H, T, K;
                         theta=zeros(mjsys.nu, H),
                         s0=zeros(mjsys.nq + mjsys.nv)*NaN, # set NaN init to signal Rollout functions
                         valuefunction=(x...)->nothing,
                         delay=0,
                         niter=T)
      ns = mjsys.ns

      if H >= T # not MPC mode
         theta=zeros(mjsys.nu, T)
         samples = mjw.allocateTrajSamples(mjsys, T, K, ns)
         traj    = mjw.allocateTrajSamples(mjsys, T, 1, ns)
         mpc     = false
         @warn("MPPI in FULL TRAJECTORY Mode")
      else
         samples = mjw.allocateTrajSamples(mjsys, H, K, ns)
         traj    = mjw.allocateTrajSamples(mjsys, T, 1, ns)
         mpc     = true
         @warn("MPPI in Model Predictive Mode")
      end

      return new(theta, mjsys, s0, valuefunction, mpc,
                 samples, traj,
                 Dict(:stocR => Vector{Float64}(),
                      :meanR => Vector{Float64}(),
                      :evalscore => Vector{Float64}() ),
                 delay,
                 niter)
   end
end

include("mppi.jl")
#include("ilqg.jl")
#include("udp.jl")

##### Control Experiment Functions

using Common: makesavedir, save

mutable struct ControlExp <: LearningStrategy
   prefix::String
   expfile::String
   expname::String
   meshfile::String
   dir::String
   statefile::String
   expresults::String

   function ControlExp(prefix::String, expfile::String,
                       expname::String, meshfile::String)
      return new(prefix, expfile, expname, meshfile, "", "", "")
   end
end

function setup!(s::ControlExp, model)
   @info("Setting up Trajectory Optimization Experiment")

   dir          = "$(s.prefix)/$(s.expname)_opt"
   expdir       = makesavedir(dir, s.expfile, model.mjsys.name, s.meshfile)
   s.dir        = "$(expdir)"
   s.statefile  = "$(expdir)/data.jld2"
   s.expresults = "$(expdir)/expmt.jld2"
end

#function hook(s::ControlExp, model, i)
#   if i==1 || model.stocR[i] >= maximum(model.stocR[1:i-1])
#      @info("\tSaving traj for experiment ", s.dir)
#      save_traj(model.mjsys, model.traj, s.statefile)
#      save(s.expresults, Dict(:stocR => model.stocR[1:i],
#                              :meanR => model.meanR[1:i],
#                              :evals => model.evalscore[1:i]))
#   end
#end

function cleanup!(s::ControlExp, model)
   mjWrap.save_traj(model.mjsys, model.traj, s.statefile) # save result of MPC anyway
   save(s.expresults, model.trace)
   @info("Saved Experiment to ", s.dir)
end

end

