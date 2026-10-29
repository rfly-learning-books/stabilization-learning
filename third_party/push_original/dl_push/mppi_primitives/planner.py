"""Hydrax MPPI using 50 ms controls, with verified model D/Q sequence labels."""
from dataclasses import dataclass
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from flax.struct import dataclass as jax_dataclass
from hydrax.task_base import Task
from hydrax.algs.mppi import MPPI
from hydrax.alg_base import Trajectory
from .model import model_xml,contact_targets


@dataclass
class PlanConfig:
    samples: int = 256
    horizon: int = 40
    knots: int = 9
    iterations: int = 3
    noise: float = .20
    temperature: float = 1.
    contraction: float = .25
    prep_switch: float = .012
    progress_weight: float = 5.
    violation_weight: float = 200.
    side_tolerance: float = .005
    angle_tolerance: float = .035
    goal_tolerance: float = .005
    backward_tolerance: float = .0001
    workspace_limit: float = .23
    solver_iterations: int = 20

    def __post_init__(self):
        if self.samples<2 or self.horizon<1 or self.knots<2 or self.iterations<1 or self.solver_iterations<1:
            raise ValueError('Invalid planner sizes')
        if not 0<self.contraction<1:raise ValueError('contraction must lie in (0,1)')
        for name in ('noise','temperature','prep_switch','progress_weight','violation_weight',
                     'side_tolerance','angle_tolerance','goal_tolerance','backward_tolerance','workspace_limit'):
            if not np.isfinite(getattr(self,name)) or getattr(self,name)<=0:raise ValueError(name)


@jax_dataclass
class DQTrajectory(Trajectory):
    # Per whole sequence: delta Vprep, delta Vgoal, Q, path violation, progress integral.
    labels: jax.Array
    feasible: jax.Array
    states: jax.Array


class PushXTask(Task):
    def __init__(self,spec,cfg):
        self.cfg=cfg
        self.xml=model_xml(spec,iterations=cfg.solver_iterations)
        model=mujoco.MjModel.from_xml_string(self.xml)
        super().__init__(model,impl='jax')
        self.dt=.05  # control period; model.opt.timestep remains .005
        self.targets=jnp.asarray(contact_targets(spec))
        self.com=jnp.asarray(model.body('object').ipos[:2])

    def prep(self,qpos):
        c,s=jnp.cos(qpos[3]),jnp.sin(qpos[3])
        relative=qpos[4:6]-qpos[:2]
        local=jnp.array([c*relative[0]+s*relative[1],-s*relative[0]+c*relative[1]])
        return jnp.min(jnp.linalg.norm(self.targets-local,axis=-1))

    def energies(self,state):
        return jnp.array([.5*(self.prep(state.qpos)/.1)**2,
                          .5*((state.userdata[0]-state.qpos[0])/.1)**2])

    def running_cost(self,state,control):
        q,v=state.qpos,state.qvel
        # Convert origin velocity to COM velocity for the exported PhysX convention.
        c,s=jnp.cos(q[3]),jnp.sin(q[3])
        offset=jnp.array([c*self.com[0]-s*self.com[1],s*self.com[0]+c*self.com[1]])
        com_v=v[:2]+v[3]*jnp.array([-offset[1],offset[0]])
        return ((q[1]-state.userdata[2])/.01)**2+((q[3]-state.userdata[3])/.1)**2 \
            +.02*jnp.sum((com_v/.05)**2)+.02*(v[3]/.5)**2+.002*jnp.sum((control/.3)**2)

    def terminal_cost(self,state):
        return self.running_cost(state,jnp.zeros(2))


class DQMPPI(MPPI):
    """Reuse hydrax sampling/warm-start/softmax; override physical rollout and score."""
    def __init__(self,task,cfg):
        self.cfg=cfg
        super().__init__(task,num_samples=cfg.samples,noise_level=cfg.noise,
                         temperature=cfg.temperature,plan_horizon=cfg.horizon*.05,
                         num_knots=cfg.knots,spline_type='zero',iterations=cfg.iterations)

    def sample_knots(self,params):
        knots,params=super().sample_knots(params)
        # Explicit no-action and current mean candidates; no historical policy.
        knots=knots.at[0].set(jnp.zeros_like(params.mean)).at[1].set(params.mean)
        return knots,params

    def rollout_with_randomizations(self,state,tk,knots,rng):
        del rng
        times=state.time+jnp.arange(self.ctrl_steps)*self.dt
        controls=self.interp_func(times,tk,knots)

        def one(sequence):
            e0=self.task.energies(state)
            preparing=(self.task.prep(state.qpos)>self.cfg.prep_switch)&(jnp.abs(state.userdata[0]-state.qpos[0])>self.cfg.goal_tolerance)
            def step(data,u):
                data=data.replace(ctrl=u)
                data=jax.lax.fori_loop(0,10,lambda _,x:mjx.step(self.model,x),data)
                energies=self.task.energies(data)
                active=jnp.where(preparing,energies[0],energies[1])
                return data,(jnp.concatenate((data.qpos,data.qvel)),self.dt*self.task.running_cost(data,u),self.dt*active)
            final,(states,qsteps,progress)=jax.lax.scan(step,state,sequence)
            states=jnp.concatenate((jnp.concatenate((state.qpos,state.qvel))[None],states),axis=0)
            en=self.task.energies(final)
            delta=en-e0
            threshold=jnp.where(preparing,.5*(self.cfg.prep_switch/.1)**2,.5*(self.cfg.goal_tolerance/.1)**2)
            e_start=jnp.where(preparing,e0[0],e0[1])
            contraction=jnp.maximum(jnp.where(preparing,delta[0],delta[1])+self.cfg.contraction*jnp.maximum(e_start-threshold,0),0)
            # Values below zero are satisfied; path includes the initial state.
            path_terms=jnp.array([
                (jnp.max(jnp.abs(states[:,:2]))-self.cfg.workspace_limit)/.1,
                (jnp.max(jnp.abs(states[:,1]-state.userdata[2]))-self.cfg.side_tolerance)/.01,
                (jnp.max(jnp.abs(states[:,3]-state.userdata[3]))-self.cfg.angle_tolerance)/.1,
                (jnp.max(-jnp.diff(states[:,0]))-self.cfg.backward_tolerance)/.1,
                (jnp.max(states[:,0]-state.userdata[0])-self.cfg.goal_tolerance)/.1,
            ])
            path_violation=jnp.maximum(path_terms,0).sum()
            terminal=self.task.terminal_cost(final)
            q=qsteps.sum()+terminal
            costs=jnp.concatenate((qsteps+self.cfg.progress_weight*progress,
                    jnp.array([terminal+self.cfg.violation_weight*(contraction+path_violation)])))
            labels=jnp.array([delta[0],delta[1],q,path_violation,progress.sum()])
            feasible=(contraction<=1e-6)&(path_violation<=1e-6)&jnp.all(jnp.isfinite(states))&jnp.all(jnp.isfinite(labels))
            return costs,labels,feasible,states
        costs,labels,feasible,states=jax.vmap(one)(controls)
        return DQTrajectory(controls=controls,knots=knots,costs=costs,
                            trace_sites=jnp.zeros((len(knots),self.ctrl_steps+1,0,3)),
                            labels=labels,feasible=feasible,states=states)


def select_candidate(costs,feasible):
    costs=np.asarray(costs)
    finite=np.isfinite(costs)
    valid=np.asarray(feasible,dtype=bool)&finite
    if valid.any():return int(np.argmin(np.where(valid,costs,np.inf))),True
    if not finite.any():raise FloatingPointError('All MPPI candidates have non-finite cost')
    return int(np.argmin(np.where(finite,costs,np.inf))),False


class Planner:
    def __init__(self,spec,cfg):
        self.cfg=cfg;self.task=PushXTask(spec,cfg);self.controller=DQMPPI(self.task,cfg)
        self.base=self.task.make_data()
        self.optimize=jax.jit(jax.vmap(self.controller.optimize,in_axes=(0,0)))
        self.evaluate=jax.jit(jax.vmap(self.controller.rollout_with_randomizations,in_axes=(0,0,0,0)))

    def initialize(self,count,seed):
        return jax.vmap(lambda k:self.controller.init_params(seed=k))(jnp.arange(seed,seed+count))

    def inject(self,states,goals,initial_xy,initial_yaw,times):
        states=jnp.asarray(states)
        def one(s,g,p,yaw,t):
            return self.base.replace(qpos=jnp.array([s[0],s[1],s[2]-.0205,s[3],s[8],s[9]]),
                                     qvel=s[jnp.array([4,5,6,7,10,11])],time=t,
                                     userdata=jnp.array([g,p[0],p[1],yaw]))
        return jax.vmap(one)(states,jnp.asarray(goals),jnp.asarray(initial_xy),jnp.asarray(initial_yaw),jnp.asarray(times))

    def plan(self,data,params,obs=None,refiner=None):
        params,trajectory=self.optimize(data,params)
        # MPPI's weighted mean can violate constraints even when some samples pass.
        mean=self.evaluate(data,params.tk,params.mean[:,None],params.rng)
        trajectory=jax.tree.map(lambda a,b:jnp.concatenate((a,b),axis=1),trajectory,mean)
        if refiner is not None:
            costs=np.asarray(trajectory.costs.sum(-1))
            indices=costs.argmin(-1)
            knots=np.asarray(trajectory.knots)[np.arange(len(indices)),indices]
            proposal=refiner(obs,knots,np.asarray(data.qpos),np.asarray(data.userdata),self.task,self.cfg)
            refined=self.evaluate(data,params.tk,jnp.asarray(proposal)[:,None],params.rng)
            trajectory=jax.tree.map(lambda a,b:jnp.concatenate((a,b),axis=1),trajectory,refined)
        costs=np.asarray(trajectory.costs.sum(-1));feasible=np.asarray(trajectory.feasible)
        chosen=[select_candidate(c,f) for c,f in zip(costs,feasible)]
        index=np.array([i for i,ok in chosen]);valid=np.array([ok for i,ok in chosen])
        batch=np.arange(len(index))
        params=params.replace(mean=trajectory.knots[batch,index])
        return params,trajectory,index,valid
