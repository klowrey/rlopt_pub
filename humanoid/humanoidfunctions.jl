
using Distances

using mjWrap
using MuJoCo

################################################## Experiment Centric Functions
## accepts model and trajectory data
## returns sum of trajectory rewards for each traj; saves per timestep reward
function rewardfunction(x::mjWrap.mjSet,
                        s0::AbstractVector{Float64}, # pass in states
                        s1::AbstractVector{Float64},
                        o0::AbstractVector{Float64}, # observations
                        o1::AbstractVector{Float64},
                        ctrl::AbstractVector{Float64})::Float64
   tid = Threads.threadid()
   d = x.datas[tid]

   height = s1[3]
   t_height = 1.25
   t_speed = 0.8

   reward = 0.0

   #reward -= 1.0*abs(s1[1]) #(s1[1]^2)         #(10*s1[1])^2
   #reward -= 1.0*abs(s1[2]) #(s1[2]^2)         #(10*s1[2])^2
   if height < t_height
      reward -= 2.0*abs(t_height - s1[3])
   #else
      #reward -= abs( s1[x.nq+1] )
      #reward -= abs( s1[x.nq+2] )
      #reward -= 1.0*abs(s1[1]) #(s1[1]^2)         #(10*s1[1])^2
      #reward -= 1.0*abs(s1[2]) #(s1[2]^2)         #(10*s1[2])^2
   #else
   #   reward += 1.0 - abs(t_speed-s1[x.nq+1])
   end

   return reward
end

function obsfunction(x::mjWrap.mjSet,
                     s::AbstractVector{Float64},
                     o::AbstractVector{Float64})
end

## accepts model,
## default state,
## preallocated matrix of states (nqnv x K)
function initfunction!(x::mjWrap.mjSet,
                       s::mjWrap.TrajSamples)
   mag = 0.2
   init_state = view( s.state, :, 1, : )
   for t=1:s.K
      #@. init_state[:,t] = s.s0 #+ rand() * mag - mag/2.0
      #init_state[3,t] -= 0.1
      #init_state .= 0.0
      #init_state[1:x.nq,t] = LAYING_QPOS
      init_state[:,t] = s.s0 #+ rand() * mag - mag/2.0
   end
end

function ctrlfunction!(x::mjWrap.mjSet, ctrl::Array{Float64})
   randn!(ctrl)
end

# modelfunc should takes in time index (iteration index) to know change curric
function modelfunction!(x::mjWrap.mjSet, iter::Int=0)
end

## evaluation function: reward function shapes exploration, evaluation
## function defines success or failure
function evalfunction(x::mjWrap.mjSet, d::mjWrap.TrajSamples)
   #return mean(d.state[1,end,:]) # x position at time T, for all K
   return mean(@view( d.state[3,end,:] )) # z position at time T, for all K
end

using ExpFunctions
myfuncs = FunctionSet(modelfunction!,
                      initfunction!,
                      ctrlfunction!,
                      obsfunction,
                      rewardfunction,
                      evalfunction)


