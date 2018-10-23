
include("../mjWrap.jl")
using ..mjWrap

T = 1#0
K = 1 #50

modelname = "models/humanoid.xml"

myskip = 1
mjsys = mjw.load_model(modelname, myskip, "normal")
mjsys2 = mjw.load_model(modelname, myskip, "normal")
samples = mjw.allocateTrajSamples(mjsys, T, K)
samples2 = mjw.allocateTrajSamples(mjsys, T, K)

samples.ctrl[:] .= 0.5
samples2.ctrl[:] .= 0.5

#@time mjw.rollout(mjsys, samples)
@time mjw.rollout2(mjsys2, samples2)
#Profile.clear_malloc_data()
println()
#J@time mjw.rollout(mjsys, samples)
@time mjw.rollout2(mjsys2, samples2)
println()
#@time mjw.rollout(mjsys, samples)
@time mjw.rollout2(mjsys2, samples2)

# actual test??
