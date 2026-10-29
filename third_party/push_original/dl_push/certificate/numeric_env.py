"""Pure numerical Gymnasium integrator: q_dot = diag(R(theta),1) c_i u.

Every step integrates immediately. No pusher, contact solver, transfer path,
engagement, actuator lag, or task-phase machine. Shapes serve only as C maps
and polygons for observation/rendering/overlap. Metres and radians throughout.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector.utils import batch_space
from shapely.geometry import shape, Polygon, box

DEFAULT_LIBRARY=Path(__file__).resolve().parents[2]/'runs/mode_library_curved32_20260912'


def exact_step(q,c,u,dt):
    """Broadcastable exact constant-mode flow, including zero angular speed."""
    q=np.asarray(q);c=np.asarray(c);s=np.asarray(u)*dt;a=c[...,2]*s
    amp=s*np.sinc(a/(2*np.pi));heading=q[...,2]+a/2
    return q+np.stack([amp*(np.cos(heading)*c[...,0]-np.sin(heading)*c[...,1]),
        amp*(np.sin(heading)*c[...,0]+np.cos(heading)*c[...,1]),a],axis=-1)


@dataclass(frozen=True)
class NumericConfig:
    workspace_m:float=.5
    dt:float=.2
    u_max:float=.6
    max_steps:int=100
    length_scale:float=.1
    success_coverage:float=.95
    position_tolerance_m:float=.005
    yaw_tolerance_rad:float=np.deg2rad(2.)
    success_mode:str='pose'  # pose, coverage, or none
    reward_mode:str='coverage'  # coverage or progress
    switch_penalty:float=0.
    terminate_on_outside:bool=False
    render_size:int=256

    def __post_init__(self):
        values=[self.workspace_m,self.dt,self.u_max,self.max_steps,self.length_scale,
            self.position_tolerance_m,self.yaw_tolerance_rad,self.render_size]
        if not np.isfinite(values).all() or min(values)<=0:raise ValueError('Invalid configuration')
        if self.success_mode not in ('pose','coverage','none'):raise ValueError('Unknown success mode')
        if self.reward_mode not in ('coverage','progress'):raise ValueError('Unknown reward mode')
        if not 0<self.success_coverage<=1:raise ValueError('Invalid coverage threshold')


class NumericPushVectorEnv(gym.vector.VectorEnv):
    """NumPy-batched integrator, 32 lanes / one calibrated shape per lane.

    action={'mode': int[N], 'u': float[N,1]}. Each action takes effect NOW,
    including a changed mode. current_mode=-1 before first positive input;
    mode observation encodes it as 0 (otherwise actual mode index + 1).
    Completed lanes freeze; reset(options={'reset_mask':bool[N]}) independently
    resets them. shape_names at construction selects/repeats any available C.
    Boundaries are informational by default, never clip/bounce the dynamics.
    """
    metadata={'render_modes':['rgb_array'],'autoreset_mode':gym.vector.AutoresetMode.DISABLED}

    def __init__(self,num_envs=32,shape_names=None,library_root=DEFAULT_LIBRARY,config=None):
        self.config=config or NumericConfig();self.num_envs=int(num_envs);self.render_mode='rgb_array'
        if self.num_envs<1:raise ValueError('num_envs must be positive')
        root=Path(library_root)
        self.available_shapes=sorted(p.parent.name for p in root.glob('*/report.json'))
        if not self.available_shapes:raise FileNotFoundError(root)
        self.shape_names=list(shape_names) if shape_names is not None else [self.available_shapes[i%len(self.available_shapes)] for i in range(self.num_envs)]
        if len(self.shape_names)!=self.num_envs or not set(self.shape_names)<=set(self.available_shapes):raise ValueError('Invalid shape selection')
        # Load numeric arrays only; never instantiate MuJoCo/MJX or an executor.
        from .library import ModeLibrary
        libs=[ModeLibrary.load(root/n/'library') for n in self.shape_names]
        if any(l.count!=12 for l in libs):raise ValueError('Expected 12 modes per shape')
        self.C=np.stack([l.c for l in libs]);self.valid=np.stack([l.valid for l in libs])
        self.cal_bounds=np.stack([np.stack([l.u_min,l.u_max],-1) for l in libs])
        self.fingerprints=[l.fingerprint for l in libs]
        self.polys=[shape(json.loads((root/n/'polygon.json').read_text())) for n in self.shape_names]
        self.ids=np.array([self.available_shapes.index(n) for n in self.shape_names],dtype=np.int64)
        B=lambda s:spaces.Box(-np.inf,np.inf,s,np.float64)
        self.single_action_space=spaces.Dict({'mode':spaces.Discrete(12),'u':spaces.Box(0,self.config.u_max,(1,),np.float64)})
        self.single_observation_space=spaces.Dict({'pose':B((3,)),'target':B((3,)),
            'error':B((3,)),'twist_world':B((3,)),'C':B((3,12)),
            'valid_modes':spaces.MultiBinary(12),'calibrated_bounds':B((12,2)),
            'shape_id':spaces.Discrete(len(self.available_shapes)), 'current_mode':spaces.Discrete(13),
            'remaining_steps':spaces.Discrete(self.config.max_steps+1)})
        self.action_space=batch_space(self.single_action_space,self.num_envs)
        self.observation_space=batch_space(self.single_observation_space,self.num_envs)
        self.q=np.zeros((self.num_envs,3));self.target=self.q.copy();self.twist=self.q.copy()
        self.mode=np.full(self.num_envs,-1,dtype=np.int64);self.steps=np.zeros(self.num_envs,dtype=np.int64)
        self.switches=self.steps.copy();self.done=np.zeros(self.num_envs,bool);self.term=self.done.copy();self.trunc=self.done.copy()
        self.reason=np.full(self.num_envs,'',dtype=object);self.coverage=np.zeros(self.num_envs)
        self.outside=self.done.copy();self.target_polys=[None]*self.num_envs;self._ready=False;self.closed=False
        self.metadata={**self.metadata,'render_fps':round(1/self.config.dt)}

    def _world(self,q,points):
        c,s=np.cos(q[2]),np.sin(q[2]);return np.asarray(points)@np.array([[c,s],[-s,c]])+q[:2]

    def _polygon(self,i,q):
        p=self.polys[i];return Polygon(self._world(q,p.exterior.coords),[self._world(q,r.coords) for r in p.interiors])

    def _errors(self):
        p=self.q[:,:2]-self.target[:,:2];a=self.target[:,2];delta=self.q[:,2]-a
        # Raw continuous local yaw: do not silently wrap non-diagonal CLFs.
        return np.c_[(np.cos(a)*p[:,0]+np.sin(a)*p[:,1])/self.config.length_scale,
                     (-np.sin(a)*p[:,0]+np.cos(a)*p[:,1])/self.config.length_scale,delta]

    def _obs(self):
        return dict(pose=self.q.copy(),target=self.target.copy(),error=self._errors(),twist_world=self.twist.copy(),
            C=self.C.copy(),valid_modes=self.valid.astype(np.int8),calibrated_bounds=self.cal_bounds.copy(),
            shape_id=self.ids.copy(),current_mode=self.mode.copy()+1,
            remaining_steps=np.maximum(0,self.config.max_steps-self.steps))

    def _info(self,extra=None):
        return dict(shape_name=np.array(self.shape_names),coverage=self.coverage.copy(),
            is_success=self.reason=='SUCCESS',reason=self.reason.astype(str),switch_count=self.switches.copy(),
            simulated_time=self.steps*self.config.dt,current_mode=self.mode.copy(),outside_workspace=self.outside.copy(),
            extrapolated=np.zeros(self.num_envs,bool) if extra is None else extra)

    def _metrics(self,i):
        poly=self._polygon(i,self.q[i]);self.coverage[i]=float(np.clip(poly.intersection(self.target_polys[i]).area/self.polys[i].area,0,1))
        w=self.config.workspace_m;self.outside[i]=not box(0,0,w,w).covers(poly)

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        if seed is not None or not hasattr(self,'rngs'):
            self.rngs=[np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(self.num_envs)]
        opt=options or {};mask=np.asarray(opt.get('reset_mask',np.ones(self.num_envs,bool)))
        if mask.shape!=(self.num_envs,) or mask.dtype!=bool:raise ValueError('reset_mask must be bool[N]')
        if not self._ready and not mask.all():raise ValueError('First reset must initialize every lane')
        for key in ('pose','target'):
            if key in opt and (np.asarray(opt[key]).shape!=(self.num_envs,3) or not np.isfinite(opt[key]).all()):raise ValueError(f'{key} requires finite [N,3]')
        w=self.config.workspace_m;world=box(0,0,w,w)
        for i in np.flatnonzero(mask):
            rng=self.rngs[i];t=np.array(opt['target'][i] if 'target' in opt else [w/2,w/2,np.pi/4],float)
            if not world.covers(self._polygon(i,t)):raise ValueError('Target outline outside workspace')
            if 'pose' in opt:q=np.array(opt['pose'][i],float)
            else:
                for _ in range(1000):
                    q=t+np.r_[rng.uniform(-.08,.08,2),rng.uniform(-np.pi/3,np.pi/3)]
                    if world.covers(self._polygon(i,q)):break
                else:raise RuntimeError('Cannot sample valid reset')
            if not world.covers(self._polygon(i,q)):raise ValueError('Object outline outside workspace')
            self.q[i]=q;self.target[i]=t;self.target_polys[i]=self._polygon(i,t)
            self.mode[i]=-1;self.steps[i]=self.switches[i]=0;self.done[i]=self.term[i]=self.trunc[i]=False
            self.reason[i]='';self.twist[i]=0;self._metrics(i)
        self._ready=True
        return self._obs(),self._info()

    def step(self,action):
        if not self._ready:raise gym.error.ResetNeeded('Call reset first')
        modes=np.asarray(action['mode']);u=np.asarray(action['u'],float)
        if modes.shape!=(self.num_envs,) or u.shape!=(self.num_envs,1):raise ValueError('mode[N], u[N,1] required')
        if modes.dtype.kind not in 'iu' or np.any((modes<0)|(modes>=12)):raise ValueError('Invalid mode index')
        if not np.isfinite(u).all() or np.any((u<0)|(u>self.config.u_max)):raise ValueError('u outside [0,u_max]; no clipping')
        u=u[:,0];active=~self.done;idx=np.flatnonzero(active)
        if np.any(active & ~self.valid[np.arange(self.num_envs),modes]):raise ValueError('Invalid mode')
        old=self.coverage.copy();c=self.C[idx,:,modes[idx]]
        switched=active & (u>0) & (self.mode>=0) & (modes!=self.mode)
        self.switches+=switched.astype(np.int64)
        self.mode[active & (u>0)]=modes[active & (u>0)]
        self.q[idx]=exact_step(self.q[idx],c,u[idx],self.config.dt)
        a=self.q[idx,2]
        self.twist[idx]=np.c_[(np.cos(a)*c[:,0]-np.sin(a)*c[:,1])*u[idx],
            (np.sin(a)*c[:,0]+np.cos(a)*c[:,1])*u[idx],c[:,2]*u[idx]]
        self.steps[active]+=1;err=self._errors()
        for i in idx:
            self._metrics(i)
            success=self.coverage[i]>=self.config.success_coverage
            if self.config.success_mode=='pose':success=success and np.linalg.norm(err[i,:2])*self.config.length_scale<=self.config.position_tolerance_m and abs(err[i,2])<=self.config.yaw_tolerance_rad
            if self.config.success_mode=='none':success=False
            if self.outside[i] and self.config.terminate_on_outside:self.term[i]=True;self.reason[i]='WORKSPACE_LIMIT'
            elif success:self.term[i]=True;self.reason[i]='SUCCESS'
            elif self.steps[i]>=self.config.max_steps:self.trunc[i]=True;self.reason[i]='BUDGET_EXCEEDED'
        self.done=self.term|self.trunc
        reward=np.clip(self.coverage/self.config.success_coverage,0,1) if self.config.reward_mode=='coverage' else self.coverage-old
        reward=(reward-self.config.switch_penalty*switched)*active
        bounds=self.cal_bounds[np.arange(self.num_envs),modes]
        extra=active & (u>0) & ((u<bounds[:,0])|(u>bounds[:,1]))
        return self._obs(),reward,self.term.copy(),self.trunc.copy(),self._info(extra)

    def render(self):
        if not self._ready:raise gym.error.ResetNeeded('Call reset first')
        from PIL import Image,ImageDraw
        size=self.config.render_size;imgs=[]
        for i,name in enumerate(self.shape_names):
            img=Image.new('RGB',(size,size),'white');draw=ImageDraw.Draw(img)
            def px(a):return [(float(x/self.config.workspace_m*size),float((1-y/self.config.workspace_m)*size)) for x,y in a]
            draw.polygon(px(self.target_polys[i].exterior.coords),fill='#d7efdc',outline='#21866a',width=2)
            draw.polygon(px(self._polygon(i,self.q[i]).exterior.coords),fill='#bed8ed',outline='#2369b0',width=2)
            draw.rectangle((0,0,size-1,size-1),outline='#bbbbbb')
            draw.text((5,5),f'{name} | {self.steps[i]} | cov {self.coverage[i]:.2f}',fill='black')
            imgs.append(np.asarray(img))
        return np.stack(imgs)

    def close(self):self.closed=True


class NumericPushEnv(gym.Env):
    metadata={'render_modes':['rgb_array'],'render_fps':5}
    def __init__(self,shape_name='T',library_root=DEFAULT_LIBRARY,config=None,render_mode='rgb_array'):
        self.vector=NumericPushVectorEnv(1,[shape_name],library_root,config)
        self.action_space=self.vector.single_action_space;self.observation_space=self.vector.single_observation_space
        self.render_mode=render_mode
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);opt=dict(options or {})
        for k in ('pose','target'):
            if k in opt:opt[k]=np.asarray(opt[k])[None]
        obs,info=self.vector.reset(seed=seed,options=opt)
        return {k:v[0] for k,v in obs.items()},{k:v[0] for k,v in info.items()}
    def step(self,action):
        obs,r,t,tr,info=self.vector.step({'mode':np.array([action['mode']]),'u':np.asarray(action['u'])[None]})
        return {k:v[0] for k,v in obs.items()},float(r[0]),bool(t[0]),bool(tr[0]),{k:v[0] for k,v in info.items()}
    def render(self):return self.vector.render()[0]
    def close(self):self.vector.close()


if 'DL-Push/Numeric-v0' not in gym.registry:
    gym.register('DL-Push/Numeric-v0',entry_point='dl_push.certificate.numeric_env:NumericPushEnv')
