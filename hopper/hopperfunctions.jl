
using mjWrap

################################################## Experiment Centric Functions

## accepts model,
## current and next states,
## current and next observations,
## applied controls that took us from current to next state

function rewardfunction(x::mjWrap.mjSet,
                        s0::AbstractVector{Float64}, # pass in states
                        s1::AbstractVector{Float64},
                        o0::AbstractVector{Float64}, # observations
                        o1::AbstractVector{Float64},
                        ctrl::AbstractVector{Float64}) # and controls
    pos0 = s0[1]
    pos1 = s1[1]
    height = s1[2]
    angle = s1[3]

    reward = (pos1 - pos0) / (x.dt*x.skip) # forward x direction

    upright_bonus = 3.0
    t_height = 0.8
    if height > t_height
        reward += upright_bonus
    end
    #reward -= upright_bonus * (height - t_height)^2
    reward -= 1e-2 * sum(ctrl.^2)

    return reward
end

function obsfunction(x::mjWrap.mjSet,
                     s::AbstractVector{Float64}, o::AbstractVector{Float64})
    o[2] = mjWrap.wraptopi(o[2]) # global angle
end   

## accepts model,
## default state,
## preallocated matrix of states (nqnv x K)
function initfunction!(x::mjWrap.mjSet,
                       s::mjWrap.TrajSamples)
    K = size(s.state, 3)

    nq, nv, nu, ns = mjWrap.modelparams(x)
    s0 = s.s0

    sTemp = zeros(nq+nv)
    c = zeros(nu)
    o = zeros(ns)
    tid = 1

    diverse = true #false

    init_state = view( s.state, :, 1, : )
    for t=1:K
        init_state[:,t] .= s0
        if isodd(t) && diverse
            # print("D ")
            angle = pi
            init_state[2,t]    = s0[2]   +rand()  * 0.1 # - 0.05
            init_state[3,t]    = s0[3]   +rand()  * angle - angle/2.0

            mjWrap.reset(x, tid, init_state[:,t], c, o)
            for i=1:50
                mjWrap.step!(x, tid, c, sTemp, o)
            end

            if (init_state[2,t] < 1)
                println("Hopper height: ", init_state[2,t])
            end
            # s.state[:,t] = sTemp
        else
            # print("N ")
            s.state[2,t]            = s0[2]           +rand()  * 0.1 - 0.05
            init_state[3,t]          = s0[3]           +rand()  * 1.0 - 0.5
            init_state[(1+nq):end,t] = s0[(1+nq):end]+rand(nv) * 0.01 - 0.005
            if (init_state[2,t] < 1)
                println("Hopper height: ", init_state[2,t])
            end
        end
    end
end

function modelfunction!(x::mjWrap.mjSet, iter::Int=0)
end

## accepts model and ndarray of control noise to be applied at sample time
## Array is: nu x T x K , same size as ctrl
function ctrlfunction!(x::mjWrap.mjSet, ctrl::Array{Float64})
    randn!(ctrl)
end

## evaluation function: reward function shapes exploration, evaluation
## function defines success or failure
function evalfunction(x::mjWrap.mjSet, d::mjWrap.TrajSamples)
    # swimmer success: max distance along x axis covered at end of T
    xaxis_index = 1
    return mean(d.state[xaxis_index,end,:]) # x position at time T, for all K
end

using ExpFunctions
myfuncs = FunctionSet(modelfunction!,
                      initfunction!,
                      ctrlfunction!,
                      obsfunction,
                      rewardfunction,
                      evalfunction)

