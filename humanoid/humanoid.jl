push!(LOAD_PATH, "../")
push!(LOAD_PATH, "./")

include("humanoidfunctions.jl")

####################################### Experiment Parameters / HyperParameters 

model_file = "$(modeldir)/humanoid.xml"

myskip = 2
dtype = Float64 # ALWAYS DO FLOAT64

my_mjsys = mjw.load_model(model_file, myskip, "normal")

const LAYING_QPOS=[-0.164158, 0.0265899, 0.101116, 0.684044, -0.160277,
                   -0.70823, -0.0693176, -0.1321, 0.0203937, 0.298099,
                   0.0873523, 0.00634907, 0.117343, -0.0320319, -0.619764,
                   0.0204114, -0.157038, 0.0512385, 0.115817, -0.0320437, 
                   -0.617078, -0.00153819, 0.13926, -1.01785, -1.57189,
                   -0.0914509, 0.708539, -1.57187] # dim 28

function getOpt()
    gamma = 0.99
    opt_K = 120 #240 #256 #512
    H = div(128, myskip)
    T = 300
    opt   = TrajOptModel(my_mjsys, H, T, opt_K;
                         theta = zeros(my_mjsys.nu, H)#,
                         #s0 = vcat(LAYING_QPOS, zeros(my_mjsys.nv))
                         #s0 = [my_mjsys.d.qpos; my_mjsys.d.qvel]
                        )

    opt_specs = MPPIStrategy(opt,
                             Matrix(0.2*I, my_mjsys.nu, my_mjsys.nu), # sigma
                             0.8,
                             gamma,
                             myfuncs)

    return opt, opt_specs
end

