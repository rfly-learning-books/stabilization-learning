"""Reduced response plant with an explicit execution-time clock (not MuJoCo).

The object remains stationary during repositioning; only the push changes pose.
Every stage consumes the deadline and contributes to integrated task error.
"""
import numpy as np
from shapely.affinity import rotate, translate
from .timing import HOME
import so101_multishape as m

def energy(q,cfg):return float(np.dot(q[:2],q[:2])/cfg.length_scale**2+2*np.sin(q[2]/2)**2)
def overlap(spec,q):
    moved=translate(rotate(spec['poly'],q[2],origin=(0,0),use_radians=True),*q[:2])
    return moved.intersection(spec['poly']).area/spec['poly'].area
def reached(spec,q):return np.linalg.norm(q[:2])<.002 and abs(q[2])<np.deg2rad(4) and overlap(spec,q)>.9

class TimedResponseEnv:
    def __init__(self,bank,shapes=None,seed=0):
        self.bank=bank;self.cfg=bank.config;self.names=shapes or bank.names
        if not self.names or any(n not in bank.names for n in self.names):raise ValueError('Unknown/empty shape split')
        self.rng=np.random.default_rng(seed);self.reset()
    def reset(self,shape=None,pose=None):
        self.name=shape or self.names[int(self.rng.integers(len(self.names)))];self.spec=self.bank.specs[self.name]
        self.q=np.asarray(pose,float).copy() if pose is not None else np.r_[self.rng.uniform(-.012,.012,2),self.rng.uniform(-.35,.35)]
        self.mode=-1;self.contact=False;self.pusher=HOME.copy();self.elapsed=0.;self.actions=0;self.done=False;self.reason='running'
        self.repositions=0;self.switches=0;self.total_transfer=0.;self.total_reward=0.
        return self.observation()
    def observation(self):
        tau,valid,reposition=self.bank.row(self.name,self.mode,self.contact)
        return dict(shape=self.bank.names.index(self.name),q=self.q.copy(),pusher=self.pusher.copy(),mode=self.mode,contact=self.contact,remaining=max(self.cfg.max_seconds-self.elapsed,0),tau=tau,reachable=valid,reposition=reposition)
    def step(self,index):
        if self.done:raise RuntimeError('Reset a terminated environment before stepping')
        cfg=self.cfg;K=len(cfg.travels);mode,k=divmod(int(index),K)
        if mode<0 or mode>=12:raise ValueError('Action index outside mode/travel grid')
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
        self.total_reward+=reward
        info=dict(dt=dt,stages=stages,integrated_error=Eint,success=self.reason=='success',reason=self.reason,elapsed_seconds=self.elapsed)
        return self.observation(),float(reward),self.done,info
    def state_dict(self):
        return dict(name=self.name,q=self.q.tolist(),pusher=self.pusher.tolist(),mode=int(self.mode),contact=bool(self.contact),elapsed=float(self.elapsed),actions=int(self.actions),done=bool(self.done),reason=self.reason,repositions=int(self.repositions),switches=int(self.switches),total_transfer=float(self.total_transfer),total_reward=float(self.total_reward),rng=self.rng.bit_generator.state)
    def load_state_dict(self,d):
        for key,value in d.items():
            if key=='rng':self.rng.bit_generator.state=value
            else:setattr(self,key,np.array(value,float) if key in ['q','pusher'] else value)
        self.spec=self.bank.specs[self.name]
