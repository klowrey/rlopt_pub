push!(LOAD_PATH, "../")
push!(LOAD_PATH, "./")

include("antfunctions.jl")

myskip = 6
model_file = "$(modeldir)/mass_ant.xml"
my_mjsys       = mjw.load_model(model_file, myskip, "normal")

function getOpt()
   T = 250
   H = 15
   opt_K = 64 #(my_mjsys.nq+my_mjsys.nv)*my_mjsys.nu
   println("Opt_K: $opt_K")
   opt = TrajOptModel(my_mjsys, H, T, opt_K;
                      #theta = randn(my_mjsys.nu, H),
                      theta = zeros(my_mjsys.nu, H),
                      #niter = 5
                     )

   opt_specs = MPPIStrategy(opt,
                            Matrix(0.1I, my_mjsys.nu, my_mjsys.nu), # sigma
                            0.8, # lambda
                            gamma,
                            myfuncs)

   return opt, opt_specs
end

function getNPG()
   dtype = Float64
   T = 500
   K = 240
   niter = 400

   my_policy      = Policy.NN{dtype}(my_mjsys.ns, my_mjsys.nu, 32) # inputs: n, m

   ls = Policy.getls(my_policy)
   ls .= -1.0

   meanK = 10
   pgmodel        = PolicyGradModel(copy(my_policy.theta),
                                    my_mjsys,
                                    my_policy,
                                    T, K, T, meanK, niter)

   # NPG baseline aka value function approximation
   #baseline    = Baseline.Quadratic{Float64}(1e-5, my_mjsys.ns, T, K)
   baseline    = Baseline.NN{Float64}(my_mjsys.ns, T, K)

   cg_iter= 20 #30
   cg_reg = 1e-6
   cg_tol = 1e-12

   gamma = 0.995
   gae = 0.98

   norm_step_size = 0.2
   fullFIM = true
   pg_specs   = NPGStrategy{dtype}(pgmodel,
                                   baseline,
                                   fullFIM,
                                   norm_step_size,
                                   gamma, gae,
                                   cg_iter, cg_reg, cg_tol, myfuncs)

   return pgmodel, pg_specs
end

