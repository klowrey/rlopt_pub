
# DOCOPT faster than argparse, but a little sensitive to spacing
doc = """We use this to train Policy Gradients. UW CSE Motion Control Lab.

Usage:
   PGTrain.jl [(<experiment> <name>)]
   PGTrain.jl -h | --help

Options:
   -h --help   Show this screen.

"""

using DocOpt
opt = docopt(doc, ARGS)

push!(LOAD_PATH, "./")

using Common: plotlines
using Random
using Statistics

prefix="/tmp/"
modeldir="$(pwd())/models"
meshfile = "$(modeldir)/meshes/" # hack for now...

expfile=opt["<experiment>"]
expname=opt["<name>"]
#polfile=opt["load"]
#cluster=opt["par"]
#myseed =opt["seed"]; if typeof(myseed) != Int myseed = parse(Int, myseed) end
const SEED = 12345

Random.seed!(SEED) # random seed, set globally
push!(LOAD_PATH, "$(pwd())")

println("Experiment called $(expname)")
println("Loading experiment parameters from $(expfile)")

################################################### start work by loading files
include("paths.jl")

using Policy
using PolicyGradient
using Baseline
using ExpFunctions
using LearningStrategies

const METHOD = :NPG
include("$(pwd())/$(expfile)") # LOADING TASK FILE with PARAMETERS
#@assert isdefined(:pgmodel)
#@assert isdefined(:pg_specs)
#@assert isdefined(:myfuncs)

pgmodel, pg_specs = getNPG()

# plot stoc & mean every other iter
# TODO General strategies; put somewhere else?
evaluatepolicy = IterFunction((model,i)->push!(model.trace[:evalscore],
                                               myfuncs.evaluate(model.mjsys, model.meansamples)))

const NITER = 4
plotR = IterFunction(NITER, (model,i)->plotlines(i,"Reward",
                                                 (model.trace[:stocR],"Stoc"),
                                                 (model.trace[:meanR],"Mean")))


plotEval = IterFunction(NITER, (model,i)->plotlines(i,"Evaluation",
                                                    (model.trace[:evalscore],"")))

#plotCTRL = IterFunction(NITER, (model,i)->
#                        begin
#                           _, idx = findmax(sum(model.meansamples.reward, dims=1))
#                           plotlines(i,"Ctrl", model.meansamples.ctrl[:,:,idx])
#                        end )
#plotCTRL2 = IterFunction(NITER, (model,i)->plotlines(i,"Ctrl Stoc", model.samples.ctrl[:,:,1]))

# meta strategies: order matters!!
pg_strat = strategy(pg_specs,              # main loops for NPG

                    PolicyExp(prefix, expfile,
                              expname, meshfile), # setup some things, hook for save

                    evaluatepolicy,          # hooks
                    plotR,            
                    plotEval,
                    #plotCTRL,
                    #plotCTRL2,
                    Verbose(MaxIter(pgmodel.niter)))

@time learn!(pgmodel, pg_strat)


########################## done


# Get a policy from file or make a new one
#=
if repr(polfile) != "nothing"
println("Loading policy from $(polfile).")
pol = Policy.loadpolicy(polfile)
params.poltype = repr(typeof(pol))[8:end]
println("Policy type: $(params.poltype)")
else
println("Making new Policy.")
pol = Environment.makepolicy(params, mjsys.ns, mjsys.nu)
end
=#

# RUN IT
#if nprocs() > 1 # if there's an additional worker process
#   print_with_color(:yellow, "CLUSTER-PARALLEL PG.\n")
#   println("EACH PROCESS NUMT: ", params.K)
#   scores, means, evals, grads, meantraj = PolicyGradient.parallelepoch!(params, mjsys, pol, myfuncs)
#else
#   print_with_color(:yellow, "SINGLE MACHINE OPERATION.\n")
#   scores, means, evals, grads, meantraj = PolicyGradient.epoch!(params, mjsys, pol, myfuncs)
#end



