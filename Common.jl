
__precompile__()

module Common

using UnicodePlots
using JLD2, FileIO
using Flux

const WIDTH=50
const HEIGHT=8

function addmachine(name, count, tunnel::Bool=false)
   #topo=:all_to_all
   topo=:master_slave
   println("ADDING $name to cluster")
   if tunnel
      addprocs([(name, count)]; max_parallel=1, topology=topo, tunnel=true)
   else
      addprocs([(name, count)]; max_parallel=1, topology=topo)
   end
   println("WORKERS: ", nworkers())
end

#samples.ctrl[:,1,:]   .= 0.0 # x0 for ou process
#samples.ctrl[:,end,:] .= 0.0 # μ for ou process
#for k=1:K
#   Common.ou_process!(@view(samples.ctrl[:,:,k]),s.Σ[1],2.0,1/T)
#end

# ornstein-uhlenbeck process for ctrl noise for stochastic search
# μ  = target value of x[end]; get some random target?
# σ  = noise. sure.
# θ  = how quickly to go to μ; fights against σ
# dt = dt of model * skip; dt of environment
# x0 = initial value of x[1]
function ou_process!(x::AbstractMatrix{Float64},
                     σ::Float64=1.0, θ::Float64=2.0,
                     dt::Float64=1/size(x,2))
   nu = size(x,1)
   T  = size(x,2)-1
   W  = zeros(nu)
   x0 = copy(x[:,1])
   μ  = copy(x[:,end])
   for (i,t) in enumerate(0.0:dt:T*dt)
      ex = exp(-θ*t)
      for j=1:nu
         x[j,i] = x0[j]*ex + μ[j]*(1.0-ex) + σ*ex*W[j]
         W[j] += exp(θ*t)*sqrt(dt)*randn()
      end
   end
end

function ou_process!(x::AbstractVector{Float64}, x0::Float64,
                     μ::Float64=(2rand()-1),
                     σ::Float64=1.0, θ::Float64=2.0,
                     dt::Float64=1/length(x))
   T = length(x)-1
   W = 0.0
   for (i,t) in enumerate(0.0:dt:T*dt)
      ex = exp(-θ*t)
      x[i] = x0*ex + μ*(1.0-ex) + σ*ex*W
      W += exp(θ*t)*sqrt(dt)*randn()
   end
end

function Rtau(x::AbstractVector{Float64}, gamma::Float64)::Float64
   T = length(x)
   R = 0.0
   for t=1:T
      R += gamma^(t-1) * x[t]
   end
   return R
end

# X: TxK
function Rtau(x::AbstractMatrix{Float64}, gamma::Float64)::Vector{Float64}
   T, K = size(x)
   R = zeros(K) 
   for t=1:T
      g = gamma^(t-1)
      for k=1:K
         R[k] += g * x[t,k]
      end
   end
   return R
end

function plotlines(iter::Integer,title::String,lines::AbstractMatrix)
   yin = minimum(lines)
   yax = maximum(lines)
   p = lineplot(lines[1,:],
                xlim=[0, size(lines,2)], 
                ylim=[yin, yax], 
                title=title,
                width=WIDTH, height=HEIGHT)
   for l=2:size(lines, 1)
      lineplot!(p, lines[l,:])
   end

   display(p)
end

function plotlines(iter::Integer,title::String,lines::AbstractVector)
   yin = minimum(lines)
   yax = maximum(lines)
   p = lineplot(lines,
                xlim=[0, size(lines,2)], 
                ylim=[yin, yax], 
                title=title,
                width=WIDTH, height=HEIGHT)

   display(p)
end

function plotlines(iter::Integer,title::String,lines...)
   if length(lines) < 1
      error("plotlines needs data and title")
   end

   n,m = extrema(lines[1][1][:])
   numP = length(lines[1][1][:])
   #r = collect(Int, round.(linspace(1, numP, min(100, numP))))
   r = (numP-min(200, numP)+1):numP
   for l in lines[2:end]
      n2, m2 = extrema(l[1][r])
      if n2 < n n = n2 end
      if m2 > m m = m2 end
   end

   p = lineplot(lines[1][1][r],
                title=title,
                ylim=[n, m],
                width=WIDTH, height=HEIGHT, name=lines[1][2])
   for l in lines[2:end]
      lineplot!(p, l[1][r], name=l[2])
   end

   display(p)
end

function plotexpmt(file)
   d = load(file)
   plotlines(length(d["stocR"]),"Reward",
             (d["stocR"],"Stoc"),
             (d["meanR"],"Mean"))
   return d
end

function makesavedir(dir_name::String, exp_file::String,
                     m_file::String, mesh_file::String="",
                     overwrite::Bool=false)
   if isdir(dir_name) == false
      mkdir(dir_name)
   else
      if overwrite == false
         val = 1
         while isdir("$(dir_name)_$(val)") == true
            val += 1
         end
         dir_name = "$(dir_name)_$(val)"
         mkdir(dir_name)
      end
   end
   # copy file to save directory
   filename = basename(exp_file)
   cp(exp_file, "$(dir_name)/$(filename)", force=overwrite)
   @info("Saving experiment $exp_file")
   open(exp_file) do f
      for l in eachline(f)
         if startswith(l, "include")
            includefile = basename(split(l, "\"")[2])
            @info("Including $includefile")
            cp(dirname(exp_file)*"/"*includefile, "$(dir_name)/$(includefile)", force=overwrite)
         end
      end
   end
   filename = basename(m_file)
   cp(m_file, "$(dir_name)/$(filename)", force=overwrite)
   @info("Model file $m_file")
   if (mesh_file != "")
      if isdir(mesh_file)
         symlink(mesh_file, dir_name*"/meshes")
      else
         symlink(dirname(mesh_file), dir_name*"/meshes")
      end
   end
   return dir_name
end

function save(expresults::String, data::Dict{Symbol, Vector{Float64}})
    jldopen(expresults, "w") do file
        for d in data
           write(file, "$(d[1])", d[2])
        end
    end
end

function copymodel(src::Flux.Chain)
   i = 1
   DT = eltype(params(m)[1].data)
   dst = Vector{DT}( sum(length.(params(src))) )
   copymodel!(dst, src)
   return dst
end


function copymodel!(dst::AbstractVector, src::Flux.Chain)
   i = 1
   for l in Tracker.data.(Flux.params(src))
      r = i:(i+length(l)-1)
      dst[r] = l
      i += length(l)
   end
end

function copymodel!(dst::Flux.Chain, src::AbstractVector)
   i = 1
   for l in Tracker.data.(Flux.params(dst))
      r = i:(i+length(l)-1)
      l[:] = src[r]
      i += length(l)
   end
end

function minibatch(nsamples::Int, szbatch::Int,
                   nbatch = div(nsamples, szbatch) )
   return [ randperm(nsamples)[1:szbatch] for i=1:nbatch ]
end


function vecvectomat(vv)
   if length(vv) > 0
      return reshape(vcat(vv[:]...), length(vv[1]), length(vv))
   else
      return Matrix{eltype(eltype(vv))}(0,0)
   end
end

function vecvectomat!(m, vv)
   r, c = size(m)
   @assert c == length(vv)
   @assert r == length(vv[1]) # assumes all elements of vv are same size
   for i=1:length(vv)
      m[:,i] = vv[i]
   end
end

end
