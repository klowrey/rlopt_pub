################################################## Experiment Centric Functions

using mjWrap

## accepts model,
## current and next states,
## current and next observations,
## applied controls that took us from current to next state
function rewardfunction(x::mjWrap.mjSet,
                        s0::AbstractVector{Float64}, # pass in states
                        s1::AbstractVector{Float64},
                        o0::AbstractVector{Float64}, # observations
                        o1::AbstractVector{Float64},
                        ctrl::AbstractVector{Float64})::Float64

   pos0 = s0[1]
   pos1 = s1[1]
   height = s1[3]

   reward = (pos1 - pos0) / (x.dt*x.skip) # forward x direction
   #upright_bonus = 1.0
   #if height > 1.0
   #    reward += upright_bonus
   #end
   #reward -= 3.0 * (height - 0.65)^2 # mode == normal
   #reward = max(reward, -1.0)
   #reward -= norm(x.d.ctrl)^2

   return max(reward, -1.0)
end

function obsfunction(x::mjWrap.mjSet,
                     s::AbstractVector{Float64},
                     o::AbstractVector{Float64})
   o[1] = s[3] # copy z height and quat
   o[2] = s[4]
   o[3] = s[5]
   o[4] = s[6]
   o[5] = s[7]
end

## accepts model,
## default state,
## preallocated matrix of states (nqnv x K)
function initfunction!(x::mjWrap.mjSet,
                       s::mjWrap.TrajSamples)
   q0 = [0.0, 0.0, 0.65, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0]
   snew = view( s.state, :, 1, : )
   K = size(snew, 2)
   nq, nv, nu, ns = mjWrap.modelparams(x)
   s0 = s.s0
   angle = 2pi # diverse
   #angle = pi/20 # nondiverse
   conts = 0
   meanre = mean(sum(s.reward, 1))
   #stdre  = stdev(sum(s.reward, 1))
   for t=1:K
      re = sum(s.reward[:,t])
      val, idx = findmax(s.reward[:,t])
      #if re < meanre || s.state[3,end,t] < 0.8 # if not continuing
      if true #re < meanre # if not continuing
         snew[1:3,t]    = q0[1:3]    + rand() * 0.8 - 0.2  # init_qpos
         #snew[3] = 0.7
         snew[4:7,t]    = q0[4:7]

         zangle         = ( rand() * angle ) - angle/2.0
         snew[4,t]      = cos( zangle/2.0 ) # set W correctly
         snew[7,t]      = sin( zangle/2.0 ) # z rotation 
         snew[8:x.nq,t] = q0[8:x.nq] + rand() * 1.0 - 0.5  # init_qpos

         snew[(1+x.nq):end,t] = rand(x.nv) * 0.1 - 0.05
      else
         conts += 1
         snew[:,t] = s.state[:,idx,t]
      end
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
   return mean(@view(d.state[1,d.T,:])) # x position at time T, for all K
end


function modelfunction!(x::mjWrap.mjSet)
end

using ExpFunctions
myfuncs = FunctionSet(modelfunction!,
                      initfunction!,
                      ctrlfunction!,
                      obsfunction,
                      rewardfunction,
                      evalfunction)


