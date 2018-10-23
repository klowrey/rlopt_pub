

__precompile__()

module PolicyGradient

using mjWrap
using LearningStrategies
import LearningStrategies: setup!, update!, hook, cleanup!, finished
export setup!, update!, hook, cleanup!, finished

using Policy
using Common
using ExpFunctions: FunctionSet

using LinearAlgebra

export PolicyGradModel
export NPGStrategy, PolicyRollout

struct PolicyGradModel
   theta::Vector     # policy parameters
   mjsys::mjWrap.mjSet
   policy::Policy.AbstractPolicy

   # scratch stuff
   samples::mjWrap.TrajSamples
   meansamples::mjWrap.TrajSamples

   trace::Dict{Symbol,Vector{Float64}}

   niter::Int

   function PolicyGradModel(theta,
                            mjsys, policy,
                            T, K, meanT, meanK, niter)
      ns = mjsys.ns
      samples     = mjw.allocateTrajSamples(mjsys, T, K, ns)
      meansamples = mjw.allocateTrajSamples(mjsys, meanT, meanK, ns)
      return new(theta, mjsys, policy,
                 samples, meansamples,
                 Dict(:stocR => Vector{Float64}(),
                      :meanR => Vector{Float64}(),
                      :evalscore => Vector{Float64}() ), niter)
   end
end

include("npg.jl")

################## utilities

using Common: makesavedir, save
export PolicyExp

# Policy Gradient experiment specific things
mutable struct PolicyExp <: LearningStrategy
   prefix::String
   expfile::String
   expname::String
   meshfile::String
   dir::String
   statefile::String
   meanfile::String
   policyfile::String
   expresults::String

   function PolicyExp(prefix::String, expfile::String,
                      expname::String, meshfile::String)
      return new(prefix, expfile, expname, meshfile, "", "", "", "")
   end
end

function setup!(s::PolicyExp, model)
   @info("SETTING UP NPG EXPERIMENT")

   poltype = Base.datatype_name(typeof(model.policy))
   dir = "$(s.prefix)/$(s.expname)_$poltype"
   expdir = makesavedir(dir, s.expfile, model.mjsys.name, s.meshfile)
   s.dir        = "$(expdir)"
   s.statefile  = "$(expdir)/data.jld2"
   s.meanfile   = "$(expdir)/mean.jld2"
   s.policyfile = "$(expdir)/policy.jld2"
   s.expresults = "$(expdir)/expmt.jld2"
end

function savepolicygrad(s::PolicyExp, model)
   @info("\tSaving traj for experiment ", s.dir)
   mjWrap.save_traj(model.mjsys, model.samples, s.statefile)
   mjWrap.save_traj(model.mjsys, model.meansamples, s.meanfile)
   Policy.save_pol(model.policy,
                   model.mjsys.skip,
                   basename(model.mjsys.name),
                   s.policyfile) # needs skip and model for policy playback (FIX LATER)
   save(s.expresults, model.trace)
end

function hook(s::PolicyExp, model, i)
   #if i==1 || model.trace[:stocR][i] >= maximum(model.trace[:stocR][1:i-1])
   N = 100
   if mod1(i, N) == N
      savepolicygrad(s, model) 
   end
end

function cleanup!(s::PolicyExp, model)
   savepolicygrad(s, model)  # TODO hack may not want to do this...
   @info("Saved Experiment to ", s.dir)
   # email myself when done
   #run(`bash ./plot2html.sh $(scores[end]) $(expfile)`)
   #if cluster == 1
   #   println(workers())
   #   for w in workers()
   #      rmprocs(w)
   #   end
   #end
end

# TODO for parallel code
# setup! starts up remote do-work functions
# update! does the baseline sync and distributed CG, same as old parallelepoch

end
