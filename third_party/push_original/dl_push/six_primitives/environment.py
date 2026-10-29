"""GPU-vectorized SAPIEN environment; direct normalized pusher XY actions."""
from dataclasses import dataclass, asdict
from pathlib import Path
import importlib
import math
import sys
import numpy as np
import torch
from gymnasium import spaces

ROOT = Path(__file__).resolve().parents[2]
EXPORT = ROOT/'gcsro/overnight/exports/pushmixed_prior_mjx_20260909'
FRAMEWORK = ROOT/'reference'
TASKS = ('push_x_pos','push_x_neg','push_y_pos','push_y_neg','rotate_cw','rotate_ccw')
SPLITS = {'train':('T','Square','RectWide','Plus','H'),
          'validation':('C_wide','C_narrow'), 'test':('L','Z')}
RESET_MODES = ('fixed_origin','full','contact_pose','contact_translation','primitive_contact')


def activate_export():
    """Use the requested export's task/agent/assets and bundled ManiSkill only."""
    for name,root in [('ca_glpr_unseen_shape',EXPORT),('starter',EXPORT),('mani_skill',FRAMEWORK)]:
        for key,module in list(sys.modules.items()):
            origin=getattr(module,'__file__',None)
            if (key==name or key.startswith(name+'.')) and origin and not Path(origin).resolve().is_relative_to(root):
                raise RuntimeError(f'{key} already imported from {origin}; start a fresh process')
    for path in (FRAMEWORK,EXPORT):
        if str(path) in sys.path:sys.path.remove(str(path))
        sys.path.insert(0,str(path))
    module=importlib.import_module('ca_glpr_unseen_shape.env.mixed_shapes')
    import mani_skill
    if not Path(module.__file__).resolve().is_relative_to(EXPORT):raise RuntimeError('Wrong task source')
    if not Path(mani_skill.__file__).resolve().is_relative_to(FRAMEWORK):raise RuntimeError('Wrong ManiSkill source')
    return module


@dataclass
class SixConfig:
    num_envs: int = 12
    split: str = 'train'
    horizon: int = 240
    image_size: int = 64
    reset_mode: str = 'fixed_origin'
    fixed_shape: str | None = None
    fixed_task: int | None = None
    distance_min: float = .02
    distance_max: float = .08
    angle_min: float = .15
    angle_max: float = 1.57
    position_tolerance: float = .005
    angle_tolerance: float = .035
    speed_tolerance: float = .01
    omega_tolerance: float = .08
    hold_steps: int = 10
    contraction: float = .02
    workspace_limit: float = .23
    reach_shaping: float = 0.
    contact_state: bool = False
    rotation_objective: str = 'cumulative_translation_norm_v2'
    task_ids: tuple[int, ...] = (0,1,2,3,4,5)

    def __post_init__(self):
        self.task_ids=tuple(self.task_ids)
        if not self.task_ids or len(set(self.task_ids))!=len(self.task_ids) or any(k not in range(6) for k in self.task_ids):raise ValueError('Invalid task set')
        if self.split not in SPLITS or self.reset_mode not in RESET_MODES:
            raise ValueError('Unknown split or reset mode')
        if self.fixed_shape is not None and self.fixed_shape not in SPLITS[self.split]:raise ValueError('Shape outside split')
        if self.fixed_task is not None and self.fixed_task not in range(6):raise ValueError('Unknown task')
        if self.fixed_task is not None and self.fixed_task not in self.task_ids:raise ValueError('Fixed task outside task set')
        if min(self.num_envs,self.horizon,self.hold_steps)<1 or self.image_size<16:raise ValueError('Invalid sizes')
        for name in ('distance_min','distance_max','angle_min','angle_max','position_tolerance',
                     'angle_tolerance','speed_tolerance','omega_tolerance','workspace_limit'):
            if not np.isfinite(getattr(self,name)) or getattr(self,name)<=0:raise ValueError(name)
        if self.distance_min>self.distance_max or self.angle_min>self.angle_max:raise ValueError('Invalid goal range')
        if not np.isfinite(self.reach_shaping) or self.reach_shaping<0:raise ValueError('Invalid shaping weight')
        if not np.isfinite(self.contraction) or self.contraction<0:raise ValueError('Invalid contraction')
        if self.rotation_objective != 'cumulative_translation_norm_v2':raise ValueError('Unsupported rotation objective')


def errors(task,initial_xy,initial_yaw,xy,lifted_yaw,amount):
    """Pure vectorized goal/side-error contract; rotation uses lifted angle."""
    axes=torch.tensor([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.],[0.,0.],[0.,0.]],device=xy.device)
    axis=axes[task];normal=torch.stack((-axis[:,1],axis[:,0]),-1)
    delta=xy-initial_xy;angle=lifted_yaw-initial_yaw
    turning=task>=4;sign=torch.where(task==4,-1.,1.)
    progress=torch.where(turning,sign*angle,(delta*axis).sum(-1))
    g=amount-progress
    transverse=(delta*normal).sum(-1)
    return g,transverse,angle,delta,progress


class SixPrimitiveEnv:
    """Torch batched API, no autoreset. reset(env_idx=...) resets selected lanes.

    s: 18 physical/reference features; g: one signed remaining m or rad;
    shape: canonical binary body-frame silhouette (1,H,W); task: six-way onehot.
    """
    def __init__(self,config=None,render_mode=None):
        self.config=config or SixConfig();c=self.config
        activate_export()
        from .scene import PartialResetScene
        cls=type('SixPrimitivePhysicalScene',(PartialResetScene,),{
            '_SHAPE_NAMES':SPLITS[c.split], '_FIXED_ORIGIN_RESET':c.reset_mode=='fixed_origin'})
        self.raw=cls(num_envs=c.num_envs,obs_mode='state',sim_backend='gpu',control_mode='velocity_xy',
                     fixed_shape=c.fixed_shape,reset_mode='full' if c.reset_mode in ('primitive_contact','fixed_origin') else c.reset_mode,contact_diagnostics=True,
                     observation_width=c.image_size,observation_height=c.image_size,render_mode=render_mode)
        self.device=self.raw.device;self.num_envs=c.num_envs;self.dt=.05
        if self.raw._sim_steps_per_control!=10 or self.raw.pusher_max_speed!=.3:
            self.close();raise RuntimeError('Export timing/action contract changed')
        self.single_action_space=spaces.Box(-1.,1.,(2,),np.float32)
        self.single_observation_space=spaces.Dict({
            's':spaces.Box(-np.inf,np.inf,(18,),np.float32),
            'g':spaces.Box(-np.inf,np.inf,(1,),np.float32),
            'shape':spaces.Box(0,1,(1,c.image_size,c.image_size),np.float32),
            'task':spaces.Box(0,1,(6,),np.float32)})
        n=c.num_envs
        self.task=torch.zeros(n,dtype=torch.long,device=self.device)
        self.amount=torch.zeros(n,device=self.device)
        self.initial_xy=torch.zeros(n,2,device=self.device)
        self.initial_yaw=torch.zeros(n,device=self.device)
        self.lifted_yaw=torch.zeros(n,device=self.device)
        self.last_yaw=torch.zeros(n,device=self.device)
        self.steps=torch.zeros(n,dtype=torch.long,device=self.device)
        self.hold=torch.zeros_like(self.steps)
        self.done=torch.ones(n,dtype=torch.bool,device=self.device)
        self.max_side=torch.zeros(n,device=self.device)
        self.max_angle=torch.zeros(n,device=self.device)
        self.backward_steps=torch.zeros(n,dtype=torch.long,device=self.device)
        self.translation_error_sum=torch.zeros(n,device=self.device)
        self.rng=torch.Generator(device=self.device)
        self.rng.manual_seed(20260910)
        # x increases right, y increases up; fixed .4 m field, no shape ID feature.
        coords=(torch.arange(c.image_size,device=self.device)+.5-c.image_size/2)*(.4/c.image_size)
        yy,xx=torch.meshgrid(-coords,coords,indexing='ij')
        points=torch.stack((xx,yy),-1)
        self.masks=torch.stack([spec.inside(points).float()[None] for spec in self.raw._shape_specs])
        # Body-frame contour support function per shape, used for task-compatible
        # contact-preparation error (reward/error design only; no action reference).
        n_ang=720
        self.support_angles=torch.linspace(-math.pi,math.pi,n_ang+1,device=self.device)[:-1]
        dirs=torch.stack((torch.cos(self.support_angles),torch.sin(self.support_angles)),-1)
        flat=points.reshape(-1,2)
        self.support=torch.stack([(flat[spec.inside(points).reshape(-1)].to(self.device)@dirs.T).amax(0)
                                  for spec in self.raw._shape_specs])

    def _physical(self):
        xy=self.raw._gather_tee_xy()
        yaw=self.raw._gather_tee_yaw()
        velocity,omega=self.raw._gather_tee_vel()
        return xy,yaw,velocity,omega,self.raw.pusher.pose.p[:,:2],self.raw.pusher.get_linear_velocity()[:,:2]

    def reset(self,seed=None,env_idx=None,task=None,amount=None):
        ids=torch.arange(self.num_envs,device=self.device) if env_idx is None else torch.as_tensor(env_idx,device=self.device,dtype=torch.long)
        if ids.ndim!=1 or ids.numel()==0 or ids.unique().numel()!=ids.numel() or (ids<0).any() or (ids>=self.num_envs).any():raise ValueError('Invalid lane indices')
        if seed is not None:self.rng.manual_seed(seed)
        self.raw._FIXED_ORIGIN_RESET = self.config.reset_mode == 'fixed_origin'
        self.raw.reset(seed=seed,options={'env_idx':ids})
        xy,yaw,*_=self._physical();n=len(ids);c=self.config
        selected=task if task is not None else c.fixed_task
        choices=torch.tensor(c.task_ids,device=self.device,dtype=torch.long)
        tasks=choices[torch.randint(len(choices),(n,),device=self.device,generator=self.rng)] if selected is None else torch.as_tensor(selected,device=self.device,dtype=torch.long).expand(n)
        if ((tasks<0)|(tasks>5)).any():raise ValueError('Task must be 0..5')
        if not torch.isin(tasks,choices).all():raise ValueError('Reset task outside configured task set')
        self.task[ids]=tasks
        low=torch.where(tasks>=4,c.angle_min,c.distance_min);high=torch.where(tasks>=4,c.angle_max,c.distance_max)
        amounts=low+(high-low)*torch.rand(n,device=self.device,generator=self.rng) if amount is None else torch.as_tensor(amount,device=self.device,dtype=torch.float32).expand(n)
        if not torch.isfinite(amounts).all() or (amounts<=0).any():raise ValueError('Goal must be positive and finite')
        self.amount[ids]=amounts
        self.initial_xy[ids]=xy[ids];self.initial_yaw[ids]=yaw[ids]
        self.lifted_yaw[ids]=yaw[ids];self.last_yaw[ids]=yaw[ids]
        self.raw._last_control_pusher_tee_force[ids]=0
        self.steps[ids]=0;self.hold[ids]=0;self.done[ids]=False
        self.max_side[ids]=0;self.max_angle[ids]=0;self.backward_steps[ids]=0
        self.translation_error_sum[ids]=0
        # Move legacy goal visuals to the actual primitive target.
        from mani_skill.utils.structs import Pose
        axes=torch.tensor([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.],[0.,0.],[0.,0.]],device=self.device)
        goalxy=xy[ids]+axes[tasks]*amounts[:,None]
        goalyaw=yaw[ids]+torch.where(tasks==4,-amounts,torch.where(tasks==5,amounts,0.))
        for i,goal in enumerate(self.raw.goals):
            positions=goal.pose.p.clone();quats=goal.pose.q.clone()
            active=ids[self.raw._active_shape_id[ids]==i]
            local=self.raw._active_shape_id[ids]==i
            positions[active,:2]=goalxy[local]
            quats[active]=self.raw._yaw_quaternion(goalyaw[local])
            goal.set_pose(Pose.create_from_pq(positions,quats))
        if c.reset_mode=='primitive_contact':
            self._reset_primitive_contact(ids)
        else:
            self.raw.scene._gpu_apply_all()
            self.raw.scene.px.gpu_update_articulation_kinematics()
            self.raw.scene._gpu_fetch_all()
        return self.observe(),self.metrics()

    def goal_description(self, lane):
        """Public task semantics: rotation has no translation target."""
        if not 0 <= lane < self.num_envs:
            raise IndexError('Invalid environment lane')
        task = int(self.task[lane])
        amount = float(self.amount[lane])
        yaw = float(self.initial_yaw[lane])
        position = None
        if task < 4:
            axes = ((1.,0.),(-1.,0.),(0.,1.),(0.,-1.))
            position = (self.initial_xy[lane].cpu().numpy() + amount*np.array(axes[task])).tolist()
        else:
            yaw += -amount if task == 4 else amount
        return {'task':TASKS[task], 'amount':amount, 'position':position, 'yaw':yaw}

    def _reset_primitive_contact(self,ids):
        # Reset-only curriculum: broad yaw, small center offsets, suitable exterior
        # contact side for the selected primitive. No teleports during step().
        from mani_skill.utils.structs import Pose
        mask=self.raw.scene._reset_mask.clone()
        self.raw.scene._reset_mask[:]=False;self.raw.scene._reset_mask[ids]=True
        try:
            center=(torch.rand(len(ids),2,device=self.device,generator=self.rng)-.5)*.04
            self.initial_xy[ids]=center
            pusher=torch.zeros_like(center)
            axes=torch.tensor([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]],device=self.device)
            for row,lane in enumerate(ids.tolist()):
                shape=int(self.raw._active_shape_id[lane]);spec=self.raw._shape_specs[shape]
                yaw=self.initial_yaw[lane];co,si=torch.cos(yaw),torch.sin(yaw)
                rot=torch.stack((torch.stack((co,-si)),torch.stack((si,co))))
                vertices=torch.tensor([[cx+sx*hx,cy+sy*hy] for (cx,cy),(hx,hy) in spec.boxes
                                       for sx in (-1,1) for sy in (-1,1)],device=self.device)
                vertices=vertices@rot.T
                task=int(self.task[lane])
                if task<4:
                    outward=-axes[task]
                else:
                    az=2*np.pi*torch.rand((),device=self.device,generator=self.rng)
                    outward=torch.stack((torch.cos(az),torch.sin(az)))
                projection=vertices@outward
                support=vertices[(projection-projection.max()).abs()<1e-6]
                point=support.mean(0)
                if task<4:
                    # Ray through the actor origin reaches actual compound boundary,
                    # unlike an extreme support vertex which introduces large torque.
                    from shapely.geometry import box,LineString
                    from shapely.ops import unary_union
                    from shapely import get_coordinates
                    geometry=unary_union([box(cx-hx,cy-hy,cx+hx,cy+hy)
                                          for (cx,cy),(hx,hy) in spec.boxes]).buffer(self.raw.agent.pusher_radius)
                    local=(rot.T@outward).detach().cpu().numpy()
                    hit=geometry.intersection(LineString([(0.,0.),tuple(local*.5)]))
                    coords=get_coordinates(hit)
                    if len(coords):
                        distance=float((coords@local).max())
                        point=outward*(distance-self.raw.agent.pusher_radius)
                if task>=4:
                    # Offset toward a support corner to expose a torque lever arm.
                    tangent=torch.stack((-outward[1],outward[0]))
                    sign=-1 if task==4 else 1
                    point=support[(support@tangent*sign).argmax()]
                pusher[row]=center[row]+point+outward*(self.raw.agent.pusher_radius+.002)
            # Use selected-lane pose tensors; preserve all untouched lanes.
            for shape,actor in enumerate(self.raw.tees):
                positions=actor.pose.p[ids].clone();quats=actor.pose.q[ids].clone()
                selected=self.raw._active_shape_id[ids]==shape
                positions[selected,:2]=center[selected]
                actor.set_pose(Pose.create_from_pq(positions,quats))
            self.raw.agent.reset(pusher.clamp(-self.raw._PUSHER_JOINT_LIMIT,self.raw._PUSHER_JOINT_LIMIT))
            # Retarget visible markers after the reset-only center displacement.
            direction=torch.cat((axes,torch.zeros(2,2,device=self.device)))[self.task[ids]]
            for shape,goal in enumerate(self.raw.goals):
                positions=goal.pose.p[ids].clone();quats=goal.pose.q[ids].clone()
                selected=self.raw._active_shape_id[ids]==shape
                positions[selected,:2]=(center+direction*self.amount[ids,None])[selected]
                goal.set_pose(Pose.create_from_pq(positions,quats))
            self.raw.scene._gpu_apply_all()
            self.raw.scene.px.gpu_update_articulation_kinematics()
            self.raw.scene._gpu_fetch_all()
        finally:self.raw.scene._reset_mask=mask

    def prep_error(self):
        """Task-compatible contact-preparation error from contour geometry.

        Translation: pusher should sit just outside the support point of the shape
        along the approach direction (opposite the push axis), on the axis line.
        Rotation: contact side is +x in world, tangential push learned by policy.
        """
        xy,yaw,*_=self._physical()
        axes=torch.tensor([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.],[0.,0.],[0.,0.]],device=self.device)
        turn=self.task>=4
        contact=torch.where(turn[:,None],torch.tensor([1.,0.],device=self.device).expand(self.num_envs,2),-axes[self.task])
        co=torch.cos(yaw);si=torch.sin(yaw)
        dx=contact[:,0]*co+contact[:,1]*si
        dy=-contact[:,0]*si+contact[:,1]*co
        alpha=torch.atan2(dy,dx)
        n_ang=self.support.shape[1]
        idx=((alpha+math.pi)/(2*math.pi)*n_ang).long()%n_ang
        radius=self.support[self.raw._active_shape_id,idx]+self.raw.agent.pusher_radius+.002
        target=xy+contact*radius[:,None]
        pusher=self.raw.pusher.pose.p[:,:2]
        return (pusher-target).norm(dim=-1)

    def metrics(self):
        xy,yaw,v,omega,px,pv=self._physical();c=self.config
        g,transverse,angle,delta,progress=errors(self.task,self.initial_xy,self.initial_yaw,xy,self.lifted_yaw,self.amount)
        turn=self.task>=4
        scale=torch.where(turn,torch.ones_like(g),torch.full_like(g,.1))
        energy=.5*(g/scale).square()
        position_drift=delta.norm(dim=-1)
        # Q is the undesired motion cost, independent of primary progress energy.
        q_cost=torch.where(turn,position_drift/.02,(transverse/.01).square()+(angle/.1).square())
        q_cost=q_cost+.02*(v.norm(dim=-1)/.05).square()+.02*(omega/.5).square()
        tolerance=torch.where(turn,c.angle_tolerance,c.position_tolerance)
        side_ok=torch.where(turn,torch.ones_like(turn),
                            (transverse.abs()<=c.position_tolerance)&(angle.abs()<=c.angle_tolerance))
        settled=(v.norm(dim=-1)<=c.speed_tolerance)&(omega.abs()<=c.omega_tolerance)
        eligible=(g.abs()<=tolerance)&side_ok&settled
        return dict(g=g,energy=energy,q_cost=q_cost,transverse_error=transverse,angle_change=angle,
                    position_drift=position_drift,progress=progress,speed=v.norm(dim=-1),omega=omega,
                    translation_error_sum=self.translation_error_sum.clone(),
                    translation_error_integral=self.translation_error_sum.clone()*self.dt,
                    q_discount=torch.where(turn,1.,.99),
                    eligible=eligible,task=self.task.clone(),shape_id=self.raw._active_shape_id.clone(),
                    prep=self.prep_error())

    def observe(self):
        xy,yaw,v,omega,px,pv=self._physical()
        delta=xy-self.initial_xy;angle=self.lifted_yaw-self.initial_yaw
        # Same origin/velocity convention as exported PushMixed; no obsolete goal fields.
        auxiliary=self.raw._last_control_pusher_tee_force/30 if self.config.contact_state else 1-self.steps/self.config.horizon
        s=torch.cat((px/.25,pv/.3,xy/.25,v/.3,torch.sin(yaw)[:,None],torch.cos(yaw)[:,None],
                     omega[:,None],(px-xy)/.25,delta/.1,angle[:,None],
                     self.amount[:,None],auxiliary[:,None]),-1)
        obs={'s':s,'g':self.metrics()['g'][:,None],
             'shape':self.masks[self.raw._active_shape_id].clone(),
             'task':torch.nn.functional.one_hot(self.task,6).float()}
        return {key:value.clone() for key,value in obs.items()}

    def contact_gap(self):
        xy,yaw,_,_,px,_=self._physical()
        relative=self.raw._rotate(px-xy,-yaw)
        distance=torch.zeros(self.num_envs,device=self.device)
        for i,spec in enumerate(self.raw._shape_specs):
            selected=self.raw._active_shape_id==i
            centers=torch.tensor([c for c,h in spec.boxes],device=self.device)
            halves=torch.tensor([h for c,h in spec.boxes],device=self.device)
            q=(relative[:,None,:]-centers).abs()-halves
            sdf=q.clamp_min(0).norm(dim=-1)+q.amax(-1).clamp_max(0)
            distance[selected]=torch.relu(sdf.amin(-1)[selected]-self.raw.agent.pusher_radius)
        return distance/.1

    def step(self,action):
        if self.done.any():raise RuntimeError('Reset done lanes before stepping the batch')
        action=torch.as_tensor(action,dtype=torch.float32,device=self.device)
        if action.shape!=(self.num_envs,2) or not torch.isfinite(action).all() or (action.abs()>1.000001).any():
            raise ValueError('Expected finite normalized actions (N,2) in [-1,1]')
        before=self.metrics()
        reach_before=self.contact_gap() if self.config.reach_shaping else torch.zeros(self.num_envs,device=self.device)
        prep_before=before['prep']
        self.raw.step(action.clamp(-1,1))  # exactly one real 50 ms transition, no candidate search
        yaw=self.raw._gather_tee_yaw()
        self.lifted_yaw+=torch.atan2(torch.sin(yaw-self.last_yaw),torch.cos(yaw-self.last_yaw))
        self.last_yaw=yaw.clone();self.steps+=1
        info=self.metrics()
        self.translation_error_sum+=info['position_drift']
        info['translation_error_sum']=self.translation_error_sum.clone()
        info['translation_error_integral']=self.translation_error_sum.clone()*self.dt
        delta_v=info['energy']-before['energy']
        scale=torch.where(self.task>=4,1.,.1)
        signed_progress=(before['g']-info['g'])/scale
        backward=torch.relu(-signed_progress)
        overshoot=torch.relu(-info['g']/scale)
        d_violation=torch.relu(delta_v+self.config.contraction*self.dt*before['energy'])+backward+overshoot
        self.max_side=torch.maximum(self.max_side,torch.where(self.task>=4,info['position_drift'],info['transverse_error'].abs()))
        self.max_angle=torch.maximum(self.max_angle,torch.where(self.task>=4,torch.zeros_like(info['g']),info['angle_change'].abs()))
        self.backward_steps+=(backward>.001).long()
        self.hold=torch.where(info['eligible'],self.hold+1,torch.zeros_like(self.hold))
        success=self.hold>=self.config.hold_steps
        xy,*_=self._physical()
        fail=xy.abs().amax(-1)>self.config.workspace_limit
        terminated=success|fail
        truncated=(self.steps>=self.config.horizon)&~terminated
        self.done=terminated|truncated
        q_cost=info['q_cost']+.002*action.square().sum(-1)
        reward=-q_cost  # Q rewards only secondary cost; D is a separate training signal.
        side_path_ok=(self.max_side<=self.config.position_tolerance)&(self.max_angle<=self.config.angle_tolerance)
        path_ok=torch.where(self.task>=4,torch.ones_like(side_path_ok),side_path_ok)&(self.backward_steps==0)
        info.update(success=success,strict_success=success&path_ok,max_side_error=self.max_side.clone(),
                    max_push_angle_error=self.max_angle.clone(),backward_steps=self.backward_steps.clone(),fail=fail,hold=self.hold.clone(),d_target=delta_v,
                    progress_target=signed_progress,d_violation=d_violation,q_cost=q_cost,
                    energy_before=before['energy'],backward=backward,overshoot=overshoot,
                    timeout=truncated,q_terminal=terminated|((self.task>=4)&truncated),
                    reach_before=reach_before,reach_after=self.contact_gap() if self.config.reach_shaping else torch.zeros_like(reach_before),
                    prep_before=prep_before,prep_after=info['prep'],
                    action=action.clone(),pusher_distance=(self.raw.pusher.pose.p[:,:2]-xy).norm(dim=-1),
                    contact_force=self.raw._last_control_pusher_tee_force.clone())
        obs=self.observe()
        if not all(torch.isfinite(x).all() for x in obs.values()):raise FloatingPointError('Nonfinite PhysX feedback')
        return obs,reward,terminated,truncated,info

    def render(self):return self.raw.render()
    def close(self):self.raw.close()

    def provenance(self):
        import hashlib
        files=[EXPORT/'ca_glpr_unseen_shape/env'/f for f in ('mixed_shapes.py','shapes.py','slider_agent.py')]
        files+=sorted(Path(__file__).parent.glob('*.py'))
        return {'backend':'SAPIEN GPU PhysX','task_source':str(EXPORT),'framework_source':str(FRAMEWORK),
                'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                'config':asdict(self.config),'tasks':TASKS,'splits':SPLITS,
                'shape_input':'synthetic binary body-frame mask from simulator footprint; not camera segmentation',
                'action':'direct normalized pusher velocity XY, u=.30*a m/s','control_dt':self.dt}
