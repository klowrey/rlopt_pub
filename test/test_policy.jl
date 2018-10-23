
include("../paths.jl")
using Policy

function testgradll(p, n, m, N)
   dt     = eltype(p.theta)
   feat   = rand(dt, n, N)
   act    = rand(dt, m, N)
   gradll = zeros(dt, p.nparam, N)

   @time Policy.gradLL!(gradll, p, feat, act)
   return gradll
end

#function testll(p, n, m, N)
#   dt   = eltype(p.theta)
#   feat = rand(dt, n, N)
#   act  = rand(dt, m, N)
#
#   ll   = zeros(dt, 1, N)
#
#   @time Policy.loglikelihood!(ll, p.theta, p, feat, act)
#end

function testclone(p, n, m, N)
   dt   = eltype(p.theta)
   feat = randn(dt, n, N)
   A = randn(m, n)
   b = rand(m)
   act  = A*feat .+ b

   @time Policy.clone(p, feat, act; epochs=30)
   act2 = p.model(feat).data

   println("BC:")
   println(mean(mean(act2-act, 2)))

   println(A)
   println(b)
end

n = 30
m = 20
N = 10000
hidden = 64

dt = Float64
#for dt in [Float32, Float64]
#   info(dt)
#   #glp = Policy.FLP{dt}(n, m)
#   nn  = Policy.FNN{dt}(n, m, 64, 2)
#
#   info("Gradll")
#   #info("GLP")
#   #testgradll(glp, n, m, 12)
#   #testgradll(glp, n, m, N)
#   info("NN")
#   testgradll(nn, n, m, 12)
#   testgradll(nn, n, m, N)
#
#   info("LL")
#   #info("GLP")
#   #testll(glp, n, m, 12)
#   #testll(glp, n, m, N)
#   info("NN")
#   testll(nn, n, m, 12)
#   testll(nn, n, m, N)
#end

feat = rand(dt, n, N)
act  = rand(dt, m, N)
ll   = zeros(dt, 1, N)
#glp = Policy.GLP{dt}(n, m)
fnn  = Policy.NN{dt}(n, m, hidden)
nn   = Policy.GLP{dt}(n, m)
#nn  = Policy.NN{dt}(n, m, hidden, 2)

#Policy.updatetheta(nn, fnn.theta)

#g1 = testgradll(fnn, n, m, N);
#
#printlin("GradLL diff: $(mean(g2-g1))")

testclone(fnn, n, m, N)

