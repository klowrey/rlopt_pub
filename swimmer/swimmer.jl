push!(LOAD_PATH, "../")
push!(LOAD_PATH, "./")

include("swimmerfunctions.jl")

####################################### Experiment Parameters / HyperParameters 

model_file = "$(modeldir)/swimmer.xml"

# TODO NOTES ON SWIMMER:
# with K == 1, it can learn in about 150 iterations
# with K = 16, 50 iterations (16 x data, 1/3 steps) doesn't learn as well.
# wtf.
# low-pass on FIM helps in K == 16 case.
# Big NOTE: this is done on LESS-DIVERSE swimmer: even more like trajopt

# simulator parameters
T = 500
K = 32 #16*4 #36*4
myskip = 5
niter = 50 #160 #0
#K = 16*4
#niter = 25

# solver parameters
gamma = 0.995
gae = 0.98

fullFIM = true

norm_step_size = 0.05

cg_iter= 12
cg_reg = 1e-6
cg_tol = 1e-10

dtype = Float64

my_mjsys       = mjw.load_model(model_file, myskip, "normal")


function getNPG()
   T = 1000
   K = 60
   myskip = 4
   niter = 50 #160 #0

   # solver parameters
   gamma = 0.995
   gae = 0.98

   fullFIM = true

   norm_step_size = 0.05

   cg_iter= 12
   cg_reg = 1e-6
   cg_tol = 1e-10


   dtype = Float64

   my_policy      = Policy.GLP{dtype}(my_mjsys.ns, my_mjsys.nu) # inputs: n, m
   #my_policy      = Policy.NN{dtype}(my_mjsys.ns, my_mjsys.nu, 32) # inputs: n, m

   pgmodel        = PolicyGradModel(copy(my_policy.theta),
                                    my_mjsys,
                                    my_policy,
                                    T, K, T, 10, niter)

   # NPG baseline aka value function approximation
   #baseline    = Baseline.Quadratic{Float64}(1e-5, my_mjsys.ns, T, K)
   baseline    = Baseline.NN{Float64}(my_mjsys.ns, T, K; nhidden=64, step=0.0001)

   pg_specs   = NPGStrategy{dtype}(pgmodel,
                                   baseline,
                                   fullFIM,
                                   norm_step_size,
                                   gamma, gae,
                                   cg_iter, cg_reg, cg_tol, myfuncs)
   return pgmodel, pg_specs
end

function  getOpt()
   mpcT = 100
   opt = TrajOptModel(my_mjsys,
                      mpcT, # horizon for mpc; set to T for non-MPC mode
                      T,
                      16; #K;
                      theta = randn(my_mjsys.nu, mpcT),
                      s0 = [my_mjsys.d.qpos; my_mjsys.d.qvel], # set initial state here
                      #valuefunction = (x)->100#randn()*10
                     )

   opt_specs = MPPIStrategy(opt,
                            Matrix(0.1I, my_mjsys.nu, my_mjsys.nu), # sigma
                            0.3,
                            gamma,
                            myfuncs)
   return opt, opt_specs
end

