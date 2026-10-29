"""Reduced response plant with an explicit execution-time clock (not MuJoCo).

The object remains stationary during repositioning; only the push changes pose.
Every stage consumes the deadline and contributes to integrated task error.
"""
import numpy as np
import operator
from shapely.affinity import rotate, translate
from .timing import HOME
from .. import so101_multishape as m

def energy(q,cfg):return float(np.dot(q[:2],q[:2])/cfg.length_scale**2+2*np.sin(q[2]/2)**2)
def overlap(spec,q):
    moved=translate(rotate(spec['poly'],q[2],origin=(0,0),use_radians=True),*q[:2])
    return moved.intersection(spec['poly']).area/spec['poly'].area
def reached(spec,q):return np.linalg.norm(q[:2])<.002 and abs(q[2])<np.deg2rad(4) and overlap(spec,q)>.9

def action_index(index):
    """Do not silently round fractional actions to a different push."""
    if isinstance(index,(bool,np.bool_)):
        raise ValueError('Action index must be an integer')
    try:return operator.index(index)
    except TypeError as error:raise ValueError('Action index must be an integer') from error

def finite_vector(value,size,label):
    vector=np.asarray(value,float)
    if vector.shape!=(size,) or not np.isfinite(vector).all():
        raise ValueError(f'{label} must be a finite vector of length {size}')
    return vector.copy()

class TimedResponseEnv:
    def __init__(self,bank,shapes=None,seed=0):
        self.bank=bank;self.cfg=bank.config;self.names=list(bank.names if shapes is None else shapes)
        if (not self.names or len(set(self.names))!=len(self.names) or
            any(n not in bank.names for n in self.names)):raise ValueError('Unknown/empty or repeated shape split')
        self.rng=np.random.default_rng(seed);self.reset()
    def reset(self,shape=None,pose=None):
        if shape is not None and shape not in self.names:raise ValueError('Shape outside declared split')
        q=finite_vector(pose,3,'Pose') if pose is not None else None
        self.name=shape if shape is not None else self.names[int(self.rng.integers(len(self.names)))];self.spec=self.bank.specs[self.name]
        self.q=q if q is not None else np.r_[self.rng.uniform(-.012,.012,2),self.rng.uniform(-.35,.35)]
        self.mode=-1;self.contact=False;self.pusher=HOME.copy();self.elapsed=0.;self.actions=0;self.done=False;self.reason='running'
        self.repositions=0;self.switches=0;self.total_transfer=0.;self.total_reward=0.
        return self.observation()
    def observation(self):
        tau,valid,reposition=self.bank.row(self.name,self.mode,self.contact)
        return dict(shape=self.bank.names.index(self.name),q=self.q.copy(),pusher=self.pusher.copy(),mode=self.mode,contact=self.contact,remaining=max(self.cfg.max_seconds-self.elapsed,0),tau=tau,reachable=valid,reposition=reposition)
    def step(self,index):
        if self.done:raise RuntimeError('Reset a terminated environment before stepping')
        cfg=self.cfg;K=len(cfg.travels);mode,k=divmod(action_index(index),K)
        if mode<0 or mode>=12:raise ValueError('Action index outside mode/travel grid')
        finite_vector(self.q,3,'Plant pose');finite_vector(self.pusher,2,'Pusher')
        if not np.isfinite(self.elapsed) or not 0<=self.elapsed<=cfg.max_seconds+1e-8:
            raise ValueError('Invalid plant clock')
        obs=self.observation();stages={};Eint=0.;input_integral=0.;start=self.elapsed
        def idle(label,duration):
            nonlocal Eint
            duration=min(duration,max(cfg.max_seconds-self.elapsed,0))
            stages[label]=duration;Eint+=energy(self.q,cfg)*duration;self.elapsed+=duration
            return duration
        if reached(self.spec,self.q):
            hold=idle('terminal_hold',cfg.terminal_hold);self.reason='success' if hold>=cfg.terminal_hold-1e-9 else 'time_limit';self.done=True
        elif not obs['reachable'][mode]:self.done=True;self.reason='unreachable_mode'
        else:
            self.actions+=1
            if obs['reposition'][mode]:
                self.repositions+=1;self.switches+=int(self.mode>=0 and mode!=self.mode)
                idle('pre_transfer_settle',cfg.settle_before_transfer)
                actual=idle('transfer',obs['tau'][mode]);self.total_transfer+=actual
                idle('engage',cfg.engage_seconds)
            before=self.q.copy();duration=min(cfg.travels[k]/cfg.push_speed,max(cfg.max_seconds-self.elapsed,0))
            travel=duration*cfg.push_speed;response=self.spec['C'][mode]
            if duration>0:
                self.q[:2]+=m.base.rot(before[2])@response[:2]*travel
                self.q[2]=float(np.arctan2(np.sin(before[2]+response[2]*travel),np.cos(before[2]+response[2]*travel)))
            # Trapezoidal running-error integral for the first-order response plant.
            Eint+=.5*(energy(before,cfg)+energy(self.q,cfg))*duration
            input_integral+=duration;self.elapsed+=duration;stages['push']=duration
            idle('post_push_settle',cfg.settle_after_push)
            if duration>0:
                self.mode=mode;self.contact=True
                self.pusher=self.q[:2]+m.base.rot(self.q[2])@(self.spec['points'][mode]+self.spec['normals'][mode]*.0041)
            if reached(self.spec,self.q):
                hold=idle('terminal_hold',cfg.terminal_hold)
                self.done=True;self.reason='success' if hold>=cfg.terminal_hold-1e-9 else 'time_limit'
            elif self.elapsed>=cfg.max_seconds-1e-9:self.done=True;self.reason='time_limit'
            elif self.actions>=cfg.max_actions:self.done=True;self.reason='action_limit'
            elif np.linalg.norm(self.q[:2])>.06:self.done=True;self.reason='workspace'
        dt=self.elapsed-start
        reward=-cfg.reward_time*dt/cfg.time_reference-cfg.reward_error*Eint-cfg.reward_input*input_integral
        if self.done:reward+=cfg.success_bonus if self.reason=='success' else -cfg.failure_penalty
        if (not np.isfinite([reward,Eint,dt,self.elapsed,self.total_transfer]).all() or
            not np.isfinite(self.q).all() or not np.isfinite(self.pusher).all() or dt<0 or
            any(not np.isfinite(value) or value<0 for value in stages.values())):
            raise FloatingPointError('Non-finite plant transition or invalid execution clock')
        self.total_reward+=reward
        if not np.isfinite(self.total_reward):raise FloatingPointError('Non-finite episode reward')
        info=dict(dt=dt,stages=stages,integrated_error=Eint,success=self.reason=='success',reason=self.reason,elapsed_seconds=self.elapsed)
        return self.observation(),float(reward),self.done,info
    def state_dict(self):
        return dict(name=self.name,q=self.q.tolist(),pusher=self.pusher.tolist(),mode=int(self.mode),contact=bool(self.contact),elapsed=float(self.elapsed),actions=int(self.actions),done=bool(self.done),reason=self.reason,repositions=int(self.repositions),switches=int(self.switches),total_transfer=float(self.total_transfer),total_reward=float(self.total_reward),rng=self.rng.bit_generator.state)
    def load_state_dict(self,d):
        required={'name','q','pusher','mode','contact','elapsed','actions','done','reason',
            'repositions','switches','total_transfer','total_reward','rng'}
        if not isinstance(d,dict) or set(d)!=required:raise ValueError('Invalid plant checkpoint fields')
        if d['name'] not in self.names:raise ValueError('Checkpoint shape outside declared split')
        state=dict(d,q=finite_vector(d['q'],3,'Checkpoint pose'),
            pusher=finite_vector(d['pusher'],2,'Checkpoint pusher'))
        for key in ('mode','actions','repositions','switches'):
            state[key]=action_index(d[key])
        if (not -1<=state['mode']<12 or not 0<=state['actions']<=self.cfg.max_actions or
            not 0<=state['switches']<=state['repositions']<=state['actions']):
            raise ValueError('Invalid checkpoint action counters')
        if any(not isinstance(d[key],(bool,np.bool_)) for key in ('contact','done')):
            raise ValueError('Invalid checkpoint booleans')
        if not isinstance(d['reason'],str) or (d['contact'] and state['mode']<0):
            raise ValueError('Invalid checkpoint contact or reason')
        for key in ('elapsed','total_transfer','total_reward'):
            if isinstance(d[key],(bool,np.bool_)) or not np.isscalar(d[key]) or not np.isfinite(d[key]):
                raise ValueError(f'Invalid checkpoint scalar: {key}')
        if not 0<=d['total_transfer']<=d['elapsed']+1e-8 or not 0<=d['elapsed']<=self.cfg.max_seconds+1e-8:
            raise ValueError('Invalid checkpoint execution clock')
        # Validate RNG before applying any state so a rejected checkpoint leaves
        # the live environment intact.
        rng=np.random.default_rng();rng.bit_generator.state=d['rng']
        for key,value in state.items():
            if key!='rng':setattr(self,key,value)
        self.rng=rng;self.spec=self.bank.specs[self.name]
