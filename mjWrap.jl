# primary features:
# rollout sequence of controls
# df/dx, df/du derivatives

# features to add:
# address mjdata, mjmodel fields by string or symbol
#   replicate the get_sensors stuff
__precompile__()

module mjWrap

using MuJoCo
using JLD2

using UnsafeArrays
using LinearAlgebra.BLAS
using Distributed

struct mjSet
    m::mj.jlModel
    d::mj.jlData
    datas::Vector{mj.jlData}

    # helpers
    sensor_names::Dict{Symbol, AbstractRange}
    name::String
    skip::Integer
    mode::Integer

    nq::Integer
    nv::Integer
    nu::Integer
    ns::Integer
    dt::Float64

    #function mjSet(model_name::String) # TODO?
    #end
end

struct TrajSamples
    T::Integer
    K::Integer
    s0::Vector{Float64}      # nqnv
    ctrl::Array{Float64,3}   # nu x T x K
    state::Array{Float64,3}  # nqnv x T x K
    obs::Array{Float64,3}    # nu x T x K
    reward::Array{Float64,2} # T x K
end

struct Trajectory
    ctrl::Matrix{Float64}   # nu x T x K
    state::Matrix{Float64}  # nqnv x T x K
    obs::Matrix{Float64}    # nu x T x K
    reward::Vector{Float64} # T x K
end

#=
type TrajDerivatives
    T::Integer
    K::Integer
    d_x::Array{Float64}
    d_u::Array{Float64}
    s_x::Array{Float64}
    s_u::Array{Float64}

    #c::Array{Float64}
    #c_x::Array{Float64}
    #c_u::Array{Float64}
    #c_xx::Array{Float64}
    #c_uu::Array{Float64}
    #c_ux::Array{Float64}
end
=#

const mjw = mjWrap
export TrajSamples, mjSet, mjw
const GDD_POS = 1
const GDD_ACC = 2
const GDD_Q = 3
const GDD_COST = 4

function finish()
    mj.deactivate()
    return 0;
end

function modelparams(x::mjSet)
    return x.nq, x.nv, x.nu, x.ns
end

function clear_model(x::mjSet)
    mj.deleteModel(x.m.m)
    mj.deleteData(x.d.d)
    for i=1:length(x.datas)
        mj.deleteData(x.datas[i])
    end
end

function load_model(model_name::String, skip::Integer, mode::String="normal", ns::Integer=0)
    pm = mj_loadXML(model_name, C_NULL)
    if pm == nothing
        return
    end
    pd = mj_makeData(pm)

    m, d = mj.mapmujoco(pm, pd)
    ndata = Threads.nthreads()
    datas = Vector{jlData}(undef, ndata)
    print("Making mjDatas: 1 ")
    datas[1] = d
    for i=2:ndata
        print("$i ")
        datas[i] = mj.mapdata(pm, mj_makeData(pm))
    end
    println()

    if m.m[].nsensor > 0
        sensors = mj.name2range(m, m.m[].nsensor,
                                m.name_sensoradr, m.sensor_adr, m.sensor_dim)
    else
        sensors = Dict(:none=>0:0)
    end

    nq = Int64(m.m[].nq)
    nv = Int64(m.m[].nv)
    nu = Int64(m.m[].nu)
    if ns==0 ns = Int64(m.m[].nsensordata) end
    dt = m.m[].opt.timestep #mj.get(m, :opt, :timestep)

    return mjSet(m, d, datas, sensors, model_name, skip, 0,
                 nq, nv, nu, ns, dt)
end

############ UTILS ############
function wraptopi(ang::Float64)
    return mod(ang+pi, 2pi) - pi
end

function unitfit(ang::Float64, val::Float64)
    clamp.(ang, -val, val)./val
end

function inertiaweight(m::mj.jlModel, d::mj.jlData,
                       v::AbstractVector{Float64})
    nv = length(d.qvel)
    mark = mj.MARKSTACK(d)

    My = @view d.stack[(mark+1):(mark+nv)]
    mj_solveM(m.m, d.d, My, v, 1)
    force = v'*My

    mj.FREESTACK(d, Int32(mark))
    return force/mj_getTotalmass(m.m)
end

#function stackAlloc(m::mj.jlModel, d::mj.jlData)
#   mark = mj.MARKSTACK(d)
#end

function limitcontrols(x::mjSet, s::TrajSamples)
    for u=1:x.nu
        clamp!(uview(s.ctrl,u,:,:),
               x.m.actuator_ctrlrange[1,u],
               x.m.actuator_ctrlrange[2,u])
    end
end

# modify ctrl vector with the noise stored in ctrl
function inertiascale!(m::mj.jlModel, d::mj.jlData,
                       input::AbstractVector{Float64},
                       output::AbstractVector{Float64})
    mj.solveM2(m.m, d.d, output, input, 1)
end

function state2data(d::mj.jlData, s::AbstractVector{Float64})
    nq = length(d.qpos)
    nv = length(d.qvel)
    copyto!(d.qpos, 1, s, 1, nq)
    copyto!(d.qvel, 1, s, nq+1, nv)
end

function data2state(s::AbstractVector{Float64}, d::mj.jlData)
    nq = length(d.qpos)
    nv = length(d.qvel)
    copyto!(s, d.qpos)
    copyto!(s, nq+1, d.qvel, 1, nv)
end

############################

function reset(m::mj.jlModel, d::mj.jlData,
               s0::AbstractVector{Float64},
               ctrl0::AbstractVector{Float64},
               obs::AbstractVector{Float64})
    # reset
    d.qacc .= 0.0
    state2data(d, s0)
    copyto!(d.ctrl, ctrl0)
    d.d[].time = 0.0 #mj.set(d, :time, 0.0)

    # out
    if length(d.sensordata) > 0
        mj_forward(m, d)
        copyto!(obs, d.sensordata)
    else
        mj_forwardSkip(m.m, d.d, Integer(mj.STAGE_NONE), 1)
    end
end

# accepts threading; tid is 1 indexed in Julia
function reset(x::mjSet,
               s0::AbstractVector{Float64},
               ctrl0::AbstractVector{Float64},
               obs::AbstractVector{Float64})
    tid = Threads.threadid()
    if x.mode>0
        reset(x.m, x.datas[tid], s0, zeros(x.datas[tid].ctrl), obs)
    else
        reset(x.m, x.datas[tid], s0, ctrl0, obs)
    end
end

########################################################################## step

function quat2euler(quat::AbstractVector{Float64})
    w, x, y, z = quat
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = atan2(sinr, cosr)

    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1
        pitch = copysign(pi / 2, sinp) # use 90 degrees if out of range
    else
        pitch = asin(sinp)
    end

    # yaw (z-axis rotation)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)  
    yaw = atan2(siny, cosy)

    return roll, pitch, yaw
end

function euler2quat(roll::Float64, pitch::Float64, yaw::Float64)
    # blame wikipedia if this doesn't work
    cy = cos(yaw * 0.5)
    sy = sin(yaw * 0.5)
    cr = cos(roll * 0.5)
    sr = sin(roll * 0.5)
    local cp = cos(pitch * 0.5)
    sp = sin(pitch * 0.5)

    w = cy * cr * cp + sy * sr * sp
    x = cy * sr * cp - sy * cr * sp
    y = cy * cr * sp + sy * sr * cp
    z = sy * cr * cp - cy * sr * sp
    return [w, x, y, z]
end

function gdd_step!(m::mj.jlModel, d::mj.jlData,
                   x::mjSet,
                   skip::Integer,
                   ctrl::AbstractVector{Float64},
                   s1::AbstractVector{Float64},
                   obs::AbstractVector{Float64})
    tid = Threads.threadid()
    nq = length(d.qpos)
    nv = length(d.qvel)
    nu = length(ctrl)
    njnt = m.m[].njnt
    nefc = d.d[].nefc
    #copyto!(d.ctrl, ctrl)

    alpha = 50.0
    local beta = 5.0
    weight = 300.0
    gain = 20.0
    p_weight = 300.0
    p_gain = 20.0

    GDopt = x.GDopt[tid]
    #println(tid, ": ", size(x.GD_Q))
    Q = view( x.GD_Q, :, tid)
    q = view( x.GD_q, :, tid)
    P = view( x.GD_P, :, tid)
    p = view( x.GD_p, :, tid)
    GDopt.Q = pointer(Q)
    GDopt.q = pointer(q)
    GDopt.P = pointer(P) # each time this could be different
    GDopt.p = pointer(p)

    mydiff = zeros(nu)
    for i=1:skip
        mj_step1(m.m, d.d)

        #setGDcost()
        mj.fullM(m.m,       GDopt.Q, d.qM)
        mj.mju_scl(GDopt.Q, GDopt.Q, alpha, nv*nv)
        mj.mulM(m.m, d.d,   GDopt.q, d.qvel)
        mj.mju_scl(GDopt.q, GDopt.q, alpha*beta, nv)

        # find all free / quat joints
        # find position indexes for them; convert quat to euler
        # take difference of euler angles
        if x.mode == GDD_POS
            pj = 1
            qi = 1 # offset for number of free / ball joints encountered
            for j=1:njnt
                if m.jnt_type[j] == Int(mj.mjJNT_FREE)
                    #println("Free")
                    mydiff[pj] = d.qpos[qi] - ctrl[pj]; qi+=1; pj+=1 # x
                    mydiff[pj] = d.qpos[qi] - ctrl[pj]; qi+=1; pj+=1 # y
                    mydiff[pj] = d.qpos[qi] - ctrl[pj]; qi+=1; pj+=1 # z
                    roll,pitch,yaw = quat2euler(d.qpos[qi:(qi+3)]); qi+=4
                    mydiff[pj:(pj+2)] = [roll,pitch,yaw] - ctrl[pj:(pj+2)]; pj+=3 # quat
                elseif m.jnt_type[i] == Int(mj.mjJNT_BALL)
                    #println("ball")
                    roll,pitch,yaw = quat2euler(d.qpos[qi:(qi+3)]); qi+=4
                    mydiff[pj:(pj+2)] = [roll,pitch,yaw] - ctrl[pj:(pj+2)]; pj+=3 # quat
                else # slide or hinge
                    #println("1dof")
                    mydiff[pj] = d.qpos[qi] - ctrl[pj]; qi+=1; pj+=1
                end
            end
            if qi!=(nq+1)
                error("bad lineup for qpos")
            end
            if pj!=(nu+1)
                error("bad lineup for dofs")
            end
            for j=1:nu # nu == nv; for each dof...
                Q[(j-1)*(nv+1) + 1] += p_weight # diagonal weights
                Madr = m.dof_Madr[j] + 1 # julia indexing
                q[j] += p_weight*
                (p_gain*mydiff[j] + 7.0*sqrt(p_gain*d.qM[Madr])*d.qvel[j])
            end
        elseif x.mode == GDD_ACC
            #mydiff[:] = ctrl #.*10.0
            for j=1:nu # nu == nv; for each dof...
                Q[(j-1)*(nv+1) + 1] += p_weight # diagonal weights
                #q[j] += p_weight*ctrl[j]
                Madr = m.dof_Madr[j] + 1 # julia indexing
                q[j] += p_weight*
                (p_gain*ctrl[j] + 7.0*sqrt(p_gain*d.qM[Madr])*d.qvel[j])
            end
        elseif x.mode == GDD_Q
            #Q[:] += At_mul_B(ctrl[1:(nv*nv)], ctrl[1:(nv*nv)])
            #q[:] += ctrl[(nv*nv+1):end]
            for i=1:nv
                Q[(i-1)*(nv+1)+1] += abs(ctrl[nv+i])*p_weight

                Madr = m.dof_Madr[i] + 1 # julia indexing
                q[i] += p_weight*
                (p_gain*ctrl[i]+7.0*sqrt(p_gain*d.qM[Madr])*d.qvel[i])
            end
            #q[:] += ctrl[(nv+1):end]*p_weight
        end
        #setGDcostDone

        mj.op_GDD_solve(m.m, d.d,
                        Ptr{mj.opGDDOption}(pointer_from_objref(GDopt)))
        mj_step2(m.m, d.d)
    end

    mj_forward(m, d)
    data2state(s1, d)
    copyto!(obs, d.sensordata)
end

function step!(m::mj.jlModel, d::mj.jlData, skip::Integer,
               ctrl::AbstractVector{Float64},
               s1::AbstractVector{Float64},
               obs::AbstractVector{Float64})
    copyto!(d.ctrl, ctrl)

    mj_step2(m.m, d.d) # qpos qvel set from before, just use new ctrl and euler step
    for i=2:skip
        #mj_step(m, d) # TODO MAKE FASTER with fwd and euler
        mj_forwardSkip(m.m, d.d, Integer(mj.STAGE_NONE), 1)
        mj_Euler(m.m, d.d)
    end

    if length(d.sensordata) > 0
        mj_forward(m, d)
        copyto!(obs, d.sensordata)
    else
        mj_forwardSkip(m.m, d.d, Integer(mj.STAGE_NONE), 1)
    end

    data2state(s1, d)
end

# accepts threading; tid is 1 indexed in Julia
function step!(x::mjSet,
               ctrl::AbstractVector{Float64},
               s1::AbstractVector{Float64},
               obs::AbstractVector{Float64})
    tid = Threads.threadid()
    if x.mode>0
        gdd_step!(x.m, x.datas[tid], x, tid,
                  x.skip, ctrl, s1, obs)
    else
        step!(x.m, x.datas[tid], x.skip, ctrl, s1, obs)
    end
end

# resets position then takes step
function step!(x::mjSet,
               s0::AbstractVector{Float64},
               ctrl::AbstractVector{Float64},
               s1::AbstractVector{Float64},
               obs::AbstractVector{Float64})
    error("init_step s0 -> s1 not implemented")
end

function alloc_fields(nqnv::Integer, nu::Integer, ns::Integer,
                      T::Integer, K::Integer)
    ctrl  = zeros(nu, T, K)
    state = zeros(nqnv, T, K) # just allocate space
    obs   = zeros(ns,   T, K)
    return ctrl, state, obs
end

# optionally we can specify what we want our observations to be elsewhere
# instead of in mujoco's sensors format
function allocateTrajSamples(x::mjSet, T::Integer, K::Integer,
                             ns::Integer=x.ns)
    nq, nv, nu, _ = mjw.modelparams(x)
    ctrl, state, obs = alloc_fields(nq+nv, nu, ns, T, K)

    s0 = zeros(nq+nv)
    s0[1:nq]     = x.d.qpos
    s0[nq+1:end] = x.d.qvel

    return TrajSamples(T, K, s0,
                       ctrl, state, obs, zeros(T,K))
end

function save_traj(x::mjSet,
                   s::TrajSamples,
                   filename::String)
    mode = x.mode>=1 ? "GDD_$(x.mode)" : "normal"
    jldopen(filename, "w") do file
        file["state"] = s.state
        file["ctrl"] = s.ctrl
        file["obs"] = s.obs
        file["model"] = basename(x.name)
        file["skip"] = x.skip
        file["mode"] = mode
    end
end

function roll(x::mjw.mjSet, startT::Int,
              state::AbstractMatrix{Float64},
              obs::AbstractMatrix{Float64},
              ctrl::AbstractMatrix{Float64},
              reward::AbstractVector{Float64},
              ctrlfunc!::Function,
              obsfunc!::Function,
              rewardfunc::Function)
    T = size(state,2)
    s = @view state[:,startT]
    c = @view ctrl[:,startT]
    o = @view obs[:,startT]
    reset(x, s, c, o)
    obsfunc!(x, s, o)    # arbitrary observation vector manipulation
    for t=startT:(T-1)

        c = @view ctrl[:,t]
        o = @view obs[:,t]

        # then we will execute the stored control
        ctrlfunc!(x, c, o)        # we can also set noise to be in ctrl

        s = @view state[:,t+1]
        o = @view obs[:,t+1]
        step!(x, c, s, o) # get next state, next observation
        obsfunc!(x, s, o) # arbitrary observation vector manipulation

        #reward[t] = rewardfunc(x, s0, s, o0, o, c)
        reward[t] = rewardfunc(x, s, s, o, o, c)
    end
    s = @view state[:,T]
    c = @view ctrl[:,T]
    o = @view obs[:,T]
    reward[T] = rewardfunc(x, s, s, o, o, c)
end

function roll2(x::mjw.mjSet, startT::Int,
               state::AbstractMatrix{Float64},
               obs::AbstractMatrix{Float64},
               ctrl::AbstractMatrix{Float64},
               reward::AbstractVector{Float64},
               ctrlfunc!::Function,
               obsfunc!::Function,
               rewardfunc::Function)
    T = size(state,2)
    s = uview( state, :, startT )
    c = uview( ctrl, :, startT )
    size_o = size(obs, 1) > 0 ? true : false
    if size_o
        o = uview( obs, :, startT ) 
    else
        o = Vector{Float64}(undef, 0)
    end
    reset(x, s, c, o)
    obsfunc!(x, s, o)    # arbitrary observation vector manipulation
    for t=startT:(T-1)

        c = uview( ctrl,:,t)
        if size_o o = uview( obs,:,t) end

        # then we will execute the stored control
        ctrlfunc!(x, c, o)        # we can also set noise to be in ctrl

        s = uview( state,:,t+1)
        if size_o o = uview( obs,:,t+1) end
        step!(x, c, s, o) # get next state, next observation
        obsfunc!(x, s, o) # arbitrary observation vector manipulation

        #reward[t] = rewardfunc(x, s0, s, o0, o, c)
        reward[t] = rewardfunc(x, s, s, o, o, c)
    end
    s = uview( state,:,T)
    c = uview( ctrl,:,T)
    if size_o o = uview( obs,:,T) end
    reward[T] = rewardfunc(x, s, s, o, o, c)
end

function rollout2(x::mjw.mjSet, 
                  samp::mjw.TrajSamples,
                  ctrlfunc!::Function=(y...)->nothing,
                  obsfunc!::Function=(y...)->nothing,
                  rewardfunc::Function=(y...)->0.0;
                  startT::Int=1)
    nthread = Int(min(samp.K, Int(Threads.nthreads()))) # cant have more threads than data
    BLAS.set_num_threads(1)
    Threads.@threads for i=1:nthread
    #for i=1:nthread
        BLAS.set_num_threads(1)
        thread_range = Distributed.splitrange(samp.K, nthread)[i]

        for k=thread_range
            if size(samp.obs, 1) > 0
                roll2(x, startT,
                      uview(samp.state,:,:,k), uview(samp.obs,:,:,k),
                uview(samp.ctrl,:,:,k), uview(samp.reward,:,k),
                ctrlfunc!, obsfunc!, rewardfunc)
            else
                roll2(x, startT,
                      uview(samp.state,:,:,k), Matrix{Float64}(undef, 0, 0),
                uview(samp.ctrl,:,:,k), uview(samp.reward,:,k),
                ctrlfunc!, obsfunc!, rewardfunc)
            end
        end
    end
    BLAS.set_num_threads(Threads.nthreads())
end

function rollout_st(x::mjw.mjSet, 
                    samp::mjw.TrajSamples,
                    ctrlfunc!::Function=(y...)->nothing,
                    obsfunc!::Function=(y...)->nothing,
                    rewardfunc::Function=(y...)->0.0;
                    startT::Int=1)
    for k=1:samp.K
        roll2(x, startT,
              uview(samp.state,:,:,k), uview(samp.obs,:,:,k),
              uview(samp.ctrl,:,:,k), uview(samp.reward,:,k),
              ctrlfunc!, obsfunc!, rewardfunc)
    end
end

function rollout(x::mjw.mjSet, traj::mjw.Trajectory,
                 ctrlfunc!::Function=(y...)->nothing,
                 obsfunc!::Function=(y...)->nothing,
                 rewardfunc::Function=(y...)->0.0)
    nthread = Int(min(samp.K, Int(Threads.nthreads()))) # cant have more threads than data
    BLAS.set_num_threads(1)
    nq, nv, ns, nu = x.nq, x.nv, x.ns, x.nu
    Threads.@threads for i=1:nthread
        #for i=1:nthread
        BLAS.set_num_threads(1)
        s0, s = Array{Float64}(nq+nv), Array{Float64}(nq+nv)
        o0, o = Array{Float64}(ns), Array{Float64}(ns)
        c     = Array{Float64}(nu)
        thread_range = Distributed.splitrange(samp.K, nthread)[i]
        #println("i: $i, tid $tid : $thread_range")

        @inbounds for k=Range(thread_range)
            s[:] = samp.state[:,1,k]
            c[:] = samp.ctrl[:,1,k]
            reset(x, s, c, o)
            obsfunc!(x, s, o)    # arbitrary observation vector manipulation
            samp.state[:,1,k] = s
            s0[:]             = s
            samp.obs[:,1,k]   = o
            o0[:]             = o
            for t=Range(1:samp.T)

                c .= samp.ctrl[:,t,k]  # if ctrlfunc does nothing (no controller)
                # then we will execute the stored control
                ctrlfunc!(x, c, o)        # we can also set noise to be in ctrl

                step!(x, c, s, o) # get next state, next observation
                obsfunc!(x, s, o) # arbitrary observation vector manipulation


                #reward[t] = rewardfunc(x, s0, s, o0, o, c)
                reward[t] = rewardfunc(x, s, s, o, o, c)
            end
            s = uview( state,:,T)
            c = uview( ctrl,:,T)
            o = view( obs,:,T)
            reward[T] = rewardfunc(x, s, s, o, o, c)
        end
    end
end
end # mj module


