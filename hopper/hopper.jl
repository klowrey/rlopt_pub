push!(LOAD_PATH, "../")
push!(LOAD_PATH, "./")

include("hopperfunctions.jl")

####################################### Experiment Parameters / HyperParameters 

model_file = "$(modeldir)/challenge_hopper.xml"

myskip = 5
dtype = Float64

my_mjsys       = mjw.load_model(model_file, myskip, "normal")

function getNPG()
    T = 800
    K = 32 #64
    niter = 80
    gamma = 0.995
    gae = 0.98

    fullFIM = true #false

    norm_step_size = 0.05

    cg_iter= 18
    cg_reg = 1e-6
    cg_tol = 1e-10

    my_policy      = Policy.GLP{dtype}(my_mjsys.ns, my_mjsys.nu) # inputs: n, m
    #my_policy      = Policy.NN{dtype}(my_mjsys.ns, my_mjsys.nu, 32) # inputs: n, m

    pgmodel        = PolicyGradModel(copy(my_policy.theta),
                                     my_mjsys,
                                     my_policy,
                                     T, K, T, 10, niter)

    # NPG baseline aka value function approximation
    baseline    = Baseline.Quadratic{Float64}(1e-5, my_mjsys.ns, T, K)
    #baseline    = Baseline.NN{Float64}(my_mjsys.ns, T, K, 128, 0.001, Baseline.getNNfeatures)

    pg_specs   = NPGStrategy{dtype}(pgmodel,
                                    baseline,
                                    fullFIM,
                                    norm_step_size,
                                    gamma, gae,
                                    cg_iter, cg_reg, cg_tol, myfuncs)

    return pgmodel, pg_specs
end

function getOpt()
    opt_K = 120 #240 #256 #512
    H = div(128, myskip)
    T = div(1000, myskip)
    mpcT = 100

    opt   = TrajOptModel(my_mjsys, H, T, opt_K;
                         theta = zeros(my_mjsys.nu, H)#,
                         #s0 = [my_mjsys.d.qpos; my_mjsys.d.qvel]
                        )

    opt_specs = MPPIStrategy(opt,
                             Matrix(0.2*I, my_mjsys.nu, my_mjsys.nu), # sigma
                             0.8,
                             gamma,
                             myfuncs)


    return opt, opt_specs
end

