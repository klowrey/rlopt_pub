
__precompile__()

module ExpSim

using Sim
using MuJoCo, Common, GLFW
using Policy: getmean!, AbstractPolicy, loadpolicy
using Flux
using mjWrap #: load_model, mjSet
using UnsafeArrays

using JLD2, FileIO
using ExpFunctions
using Printf

export simulate 

@enum RUN_MODE runmodel=1 rundata=2 runpolicy=3
const runmodes = [runmodel, rundata, runpolicy]    # HAAAAAAAAAAAACK
mutable struct ExpDisplay
   T::Int
   t::Int
   K::Int
   k::Int
   nqnv::Int
   states::Array{Float64, 3}

   pol #::AbstractPolicy
   using_policy::Bool
   skip::Int

   mode::RUN_MODE

   mjsys::mjWrap.mjSet
   myfuncs #::ExpFunctions.FunctionSet

   function ExpDisplay(mjsys, states, pol, skip)
      nqnv, T, K = size(states)
      return new(T, 1, K, 1, nqnv, states,
                 pol, true, skip, runmodel, mjsys, nothing)
   end
end

function traj2state(exd::ExpDisplay)
   t, k = exd.t, exd.k
   m, d = exd.mjsys.m, exd.mjsys.d
   nq = m.m[].nq
   d.qpos .= exd.states[1:nq, t, k]
   d.qvel .= exd.states[(nq+1):end, t, k]
   mj_forward(m, d)
end

function file2state(s::Simulation, exd::ExpDisplay) 
   #copy data from traj struct, keeping track of indicies
   if mod1(s.framenum, exd.skip) == exd.skip && s.lastframenum != s.framenum
      if (exd.t >= exd.T) 
         exd.t = 1  #start of new trajectory
         exd.k += 1
         if exd.k > exd.K exd.k = 1 end
         s.d.d[].time = 0.0
         @printf("Rendering Trajectory %d\n", exd.k)
      end
      traj2state(exd)
      exd.t += 1 # advance our saved trajectory pointer
      s.lastframenum = s.framenum
   end
end

function applypolicy(s::Simulation, exd::ExpDisplay)
   d = exd.mjsys.d
   if mod1(s.framenum, exd.skip) == exd.skip && s.lastframenum != s.framenum
      # Needs to eval the function loaded at run time
      obs = zeros(exd.mjsys.ns)
      eval( :($(exd.myfuncs.observe!)($(exd.mjsys), [$(d.qpos); $(d.qvel)], $(obs))) )
      c = uview(d.ctrl)
      o = uview(obs)
      getmean!(c, exd.pol, o)
      # getmean!(d.ctrl, exd.pol, obs)
      s.lastframenum = s.framenum
   end
end

function my_step(s::Simulation, exd::Union{ExpDisplay,Nothing})
   m = s.m
   d = s.d
   if exd == nothing
      mj_step(m, d)
   else
      if exd.mode == rundata
         file2state(s, exd)
         d.d[].time += m.m[].opt.timestep*exd.skip
      elseif exd.mode == runpolicy && exd.using_policy && exd.pol != nothing
         applypolicy(s, exd)
         mj_step(m, d)
      else # model mode / passive policy mode
         d.ctrl .= 0.0
         mj_step(m, d)
      end
   end
   s.framenum += 1
end

function simulation(s::Simulation, exd::Union{ExpDisplay,Nothing})
   # println("simulation")
   d = s.d
   m = s.m
   if s.paused
      if s.pert[].active > 0
         mjv_applyPerturbPose(m, d, s.pert, 1)  # move mocap and dynamic bodies
         mj_forward(m, d)
      end
   else
      #slow motion factor: 10x
      factor = (s.slowmotion ? 10 : 1)

      # advance effective simulation time by 1/refreshrate
      startsimtm = d.d[].time
      while ((d.d[].time - startsimtm) * factor < (1.0 / s.refreshrate))
         # clear old perturbations, apply new
         #mju_zero(d->xfrc_applied, 6 * m->nbody);
         d.xfrc_applied .= 0.0
         if s.pert[].select > 0
            mjv_applyPerturbPose(m, d, s.pert, 0) # move mocap bodies only
            mjv_applyPerturbForce(m, d, s.pert)
         end

         my_step(s, exd)

         # break on reset
         if (d.d[].time < startsimtm) break end
      end
   end
end

function render(s::Simulation, exd::Union{ExpDisplay,Nothing}, w::GLFW.Window)
   wi, hi = GLFW.GetFramebufferSize(w)
   rect = mjrRect(Cint(0), Cint(0), Cint(wi), Cint(hi))
   smallrect = mjrRect(Cint(0), Cint(0), Cint(wi), Cint(hi))

   simulation(s, exd)

   # update scene
   mjv_updateScene(s.m, s.d,
                   s.vopt, s.pert, s.cam, Int(mj.CAT_ALL), s.scn)
   # render
   mjr_render(rect, s.scn, s.con)

   if s.showsensor
      if (!s.paused) Sim.sensorupdate(s) end
      Sim.sensorshow(s, smallrect)
   end

   if s.showinfo
      str_slow = s.slowmotion ? "(10x slowdown)" : ""
      str_paused = (s.paused ? "\nPaused" : "\nRunning")*"\nTime:"
      time = round(s.d.d[].time, digits=3)
      if exd != nothing
         if exd.mode == runmodel
            status = "\nPassive Model\n$(time)"
         elseif exd.mode == rundata
            str_paused *= "\nTraj\nT\n"
            status = "\nData Mode\n$(time)\n$(exd.k)\n$(exd.t)"
         elseif exd.mode == runpolicy
            status = "\nPolicy Interaction\n$(time)"
         end
      else
         status = "\nPassive Model\n$(time)"
      end
      status = str_slow * status

      mjr_overlay(Int(mj.FONT_NORMAL), Int(mj.GRID_BOTTOMLEFT), rect,
                  str_paused,
                  status, s.con)
   end

   if s.record != nothing
      mjr_readPixels(s.vidbuff, C_NULL, rect, s.con);
      write(s.record[1], s.vidbuff[1:3*rect.width*rect.height]);
   end

   # Swap front and back buffers
   GLFW.SwapBuffers(w)
end

function mycustomkeyboard(s::Simulation, exd::Union{ExpDisplay,Nothing}, window::GLFW.Window,
                          key::GLFW.Key, scancode::Int32, act::GLFW.Action, mods::Int32)
   if act == GLFW.RELEASE return end

   if exd != nothing # more than just model d
      mode = exd.mode
      if key == GLFW.KEY_COMMA
         if mode==runmodel mode=runpolicy else mode=runmodes[Int(mode) - 1] end
      elseif key == GLFW.KEY_PERIOD
         if mode==runpolicy mode=runmodel else mode=runmodes[Int(mode) + 1] end
      elseif mode == runpolicy && key == GLFW.KEY_P
         exd.using_policy = !exd.using_policy
         if exd.using_policy println("Using Policy") end
      end
      exd.mode = mode

      if exd.mode == rundata
         # If holding shift, then cycle through trajectories
         if mods & GLFW.MOD_SHIFT > 0
            if key == GLFW.KEY_RIGHT
               # exd.t += 20
               # if exd.t > exd.T
               exd.k += 1; if exd.k > exd.K exd.k = 1 end
               exd.t = 1
               # end
               traj2state(exd)
            elseif key == GLFW.KEY_LEFT
               # exd.t -= 20
               # if exd.t < 1
               exd.k -= 1; if exd.k < 1 exd.k = exd.K end
               exd.t = 1
               # end
               traj2state(exd)
            end
         else  # Else step forward in time
            if s.paused
               if key == GLFW.KEY_RIGHT
                  exd.t += 1; if exd.t > exd.T exd.t = 1 end
                  traj2state(exd)
               elseif key == GLFW.KEY_LEFT
                  exd.t -= 1; if exd.t < 1 exd.t = exd.T end
                  traj2state(exd)
               end
            end
         end
      end
   end

   Sim.mykeyboard(s, window, key, scancode, act, mods)
end

function findinfile(name::String, s::String)
   open(name) do f
      while !eof(f)
         x = readline(f)
         if startswith(x,s) 
            return x
         end
      end
   end
end

macro preloadexp(dir)
   return quote
      files        = readdir($dir)
      modelfile    = $dir*"/"*files[findall((x)->endswith(x, ".xml"), files)][1]
      expfile      = $dir*"/"*files[findall((x)->endswith(x, ".jl")&&occursin(x, "functions")==false, files)][1]
      functionfile = $dir*"/"*files[findall((x)->endswith(x, "functions.jl"), files)][1]
      polfile      = $dir*"/policy.jld2"
      if isfile($dir*"/mean.jld2")
         datafile  = $dir*"/mean.jld2" #TODO HACK loading mean samples for XXX
      else
         datafile  = $dir*"/data.jld2"
      end

      #@info("Loading Experiment $dir")
      #@info("Model file: $modelfile")
      #@info("Function file: $functionfile")
      #@info("Loading policy file $polfile")
      include(functionfile)

      expmt = Common.plotexpmt($dir*"/expmt.jld2") # why not
      maxiter = argmax(expmt["stocR"])

      if isfile(polfile)
         mypolicy, frameskip, _ = loadpolicy(polfile)
      else
         mypolicy = nothing 
         println("Loading skip from: ", expfile)
         frameskip = eval( Meta.parse( findinfile(expfile, "myskip")))
         println(Meta.parse( findinfile(expfile, "myskip")))
         @warn("Can't load policy, or no policy to load")
      end
      @info("Frameskip: $frameskip")
      if isdefined(:ns) == false
         ns = size(load(datafile, "obs"), 1)
      end
      my_mjsys = mjw.load_model(modelfile, frameskip, "normal", ns)

      states = load(datafile, "state")
      exd = ExpDisplay( my_mjsys, load(datafile, "state"), mypolicy, frameskip)

      #exd.myfuncs = Main.myfuncs
      exd.myfuncs = myfuncs
      @info("Affecting model accoring to iteration $maxiter")
      for i=1:maxiter
         exd.myfuncs.setmodel!(exd.mjsys, i) # applies curriculum
      end
      return exd
   end
end

function loadexp(dir, width, height)
   @info(dir)
   exd = eval( :(@preloadexp($dir)) )

   s = Sim.start(exd.mjsys.m, exd.mjsys.d,  width, height)

   # Simulation struct and ExpDisplay struct share mjmodel and mjdata
   GLFW.SetKeyCallback(s.window, (w,k,sc,a,mo)->mycustomkeyboard(s,exd,w,k,sc,a,mo))
   GLFW.SetWindowRefreshCallback(s.window, (w)->render(s,exd,w))

   return s, exd
end

function loadmodel(modelfile, width, height)
   ptr_m = mj_loadXML(modelfile, C_NULL)
   ptr_d = mj_makeData(ptr_m)
   m, d = mj.mapmujoco(ptr_m, ptr_d)
   mj_forward(m, d)
   s = Sim.start(m, d, width, height)
   @info("Model file: $modelfile")

   # Simulation struct and ExpDisplay struct share mjmodel and mjdata
   GLFW.SetKeyCallback(s.window, (w,k,sc,a,mo)->mycustomkeyboard(s,nothing,w,k,sc,a,mo))
   GLFW.SetWindowRefreshCallback(s.window, (w)->render(s,nothing,w))

   return s, nothing
end

function start(args::Vector{String}, width=800, height=480)
   opt = settings(args)

   if isdir(opt["arg1"]) 
      s, exd = loadexp(opt["arg1"], width, height)
   elseif endswith(opt["arg1"], ".xml")
      s, exd = loadmodel(opt["arg1"], width, height)
   else
      error("Unrecognized file input.")
   end

   return s, exd
end

function simulate(s, exd::Union{ExpDisplay,Nothing})
   # Loop until the user closes the window
   Sim.alignscale(s)
   while !GLFW.WindowShouldClose(s.window)
      render(s, exd, s.window)
      GLFW.PollEvents()
   end
   GLFW.DestroyWindow(s.window)
end

function simulate(f::String, width=1280, height=800)
   if isdir(f) 
      s, exd = loadexp(f, width, height)
   elseif endswith(f, ".xml")
      s, exd = loadmodel(f, width, height)
   else
      error("Unrecognized file input.")
   end
   simulate(s, exd)
end

end
