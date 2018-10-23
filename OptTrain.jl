
# DOCOPT faster than argparse, but a little sensitive to spacing
doc = """We use this to find an MPPI trajectory. UW CSE Motion Control Lab.

Usage:
   OptTrain.jl [(<experiment> <name>)]
   OptTrain.jl -h | --help

Options:
   -h --help   Show this screen.

"""

using DocOpt
opt = docopt(doc, ARGS)


using Distributed
using LinearAlgebra
using Random
using Statistics

push!(LOAD_PATH, "./")

using Common: plotlines

prefix="/tmp/"
modeldir="$(pwd())/models"
meshfile = "$(modeldir)/meshes/" # hack for now...

expfile=opt["<experiment>"]
expname=opt["<name>"]
#cluster=opt["par"]
#myseed =opt["seed"]; if typeof(myseed) != Int myseed = parse(Int, myseed) end
myseed = 12345

const SEED = myseed
Random.seed!(SEED)
push!(LOAD_PATH, "$(pwd())")

println("Experiment called $(expname)")
println("Loading experiment parameters from $(expfile)")

################################################### start work by loading files
#using mjWrap
include("paths.jl")
using TrajOpt
using LearningStrategies

const METHOD = :OPT
include("$(pwd())/$(expfile)") # LOADING TASK FILE with PARAMETERS
#@assert isdefined(Base, :opt)
#@assert isdefined(Base, :opt_specs)
#@assert isdefined(Base, :myfuncs)

opt, opt_specs = getOpt()

# plot stoc & mean every other iter
evaluatepolicy = IterFunction((model,i)->push!(model.trace[:evalscore], myfuncs.evaluate(model.mjsys, model.samples)))

const NITER = 20
plotR = IterFunction(NITER, (model,i)->plotlines(i,"Costs, H=$(model.samples.T)",
                                                 (model.trace[:stocR],"Avg"),
                                                 (model.trace[:meanR],"Min"),
                                                 (model.trace[:maxR],"Max")))

plotEval = IterFunction(NITER, (model,i)->plotlines(i,"Evaluation",
                                                    (model.trace[:evalscore],"")))

plotCTRL = IterFunction(NITER, (model,i)->plotlines(i,"Ctrl",
                                                    #model.samples.ctrl[:,:,1]))
                                                    @view( model.traj.ctrl[1:2,:,1] )))

plotState = IterFunction(NITER, (model,i)->plotlines(i,"State",
                                                     model.traj.state[1:3,1:i,1])) # X axis plot for now

# meta strategies: order matters!!
opt_strat = strategy(opt_specs,                  # main loop

                     ControlExp(prefix, expfile,
                                expname, meshfile), # setup some things, hook for save

                     #evaluatepolicy,             # hooks

                     plotR,            

                     #plotEval,
                     plotCTRL,

                     #plotState,
                     #MaxIter(opt.niter)))
                     Verbose(MaxIter(opt.niter)) )

@time learn!(opt, opt_strat)


########################## done


