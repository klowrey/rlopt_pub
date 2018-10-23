using Pkg
pkgs = ["JLD2",
        "ReverseDiff",
        "Distributions",
        "LearningStrategies",
        "IterativeSolvers",
        "Distances",
        "GLFW",
        "DocOpt",
        "UnicodePlots",
        "Flux",
        "UnsafeArrays"]

installed = Pkg.installed()

for p in pkgs
   if haskey(installed, p) == false
      Pkg.add(p)
   else
      @info "$p installed"
   end
end

if haskey(installed, "MuJoCo") == false
   Pkg.clone("git://www.github.com/klowrey/MuJoCo.jl.git")
   Pkg.build("MuJoCo")
end

#Pkg.checkout("LearningStrategies")

using MuJoCo
for p in pkgs
   @info "Preloading $p"
   eval(parse("using $p"))
end
