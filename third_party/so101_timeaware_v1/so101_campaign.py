"""Paired ablations, persistence, disturbance/boundary tests and actuated arm.

All rollout states come from MuJoCo. IK for the actuated arm runs on a separate
MjData; it never sets the live robot joint positions after reset.
"""
import argparse
import collections
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import so101_contact_demo as base
import numpy as np
import torch
import mujoco
from shapely.geometry import Point, LineString
from dl_push_contact.envs import so101_push_t_env as arm_module

PACKAGE=Path(__file__).resolve().parents[1]
DEFAULT_OUT=PACKAGE/'media/so101_campaign'
SOURCE_TEXT=Path(__file__).read_text()
SOURCE_SHA=hashlib.sha256(SOURCE_TEXT.encode()).hexdigest()

@dataclass(frozen=True)
class Config:
    metric: str='finetuned'
    persistence: bool=True
    plant: str='planar'
    damping: float=1.
    mass: float=1.
    friction: float=.6
    delay_ticks: int=0
    noise_mm: float=0.
    noise_deg: float=0.
    max_actions: int=100
    max_time: float=240.
    speed: float=.025
    joint_speed: float=.8
    joint_accel: float=2.
    torque_limit: float=3.
    actual_accel_stop: float=50.

def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False))

def dynamic_xml(cfg):
    xml,mesh=arm_module._build_pusht_xml()
    root=ET.fromstring(xml)
    root.find('option').set('timestep','.0025')
    block=root.find(".//body[@name='block']")
    for joint in list(block.findall('joint')):block.remove(joint)
    ET.SubElement(block,'freejoint',name='block_free')
    for g in block.findall('geom'):
        size=g.get('size').split();size[2]='.004';g.set('size',' '.join(size))
        g.set('mass',str(float(g.get('mass'))*cfg.mass))
        g.set('contype','8');g.set('conaffinity','7');g.set('friction',f'{cfg.friction} .001 .0001')
        g.set('condim','4');g.set('priority','1')
    root.find(".//geom[@name='table']").set('conaffinity','14')
    root.find(".//geom[@name='table']").set('friction',f'{cfg.friction} .001 .0001')
    for name in ['pusher_collision','pusher_marker']:
        g=root.find(f".//geom[@name='{name}']");g.set('contype','0');g.set('conaffinity','0');g.set('rgba','0 0 0 0')
    masses={'r_shoulder':.15,'r_upper_arm':.16,'r_lower_arm':.14,'r_wrist':.10,'r_gripper':.12,'r_moving_jaw':.025}
    for body_name,mass in masses.items():
        body=root.find(f".//body[@name='{body_name}']");body.set('gravcomp','0')
        geoms=body.findall('geom')
        for g in geoms:
            g.set('mass',str(mass/len(geoms)));g.set('contype','2');g.set('conaffinity','11')
        j=body.find('joint');j.set('damping','.12');j.set('armature','.002');j.set('frictionloss','.005')
    rod=root.find(".//geom[@name='r_tool_rod']");rod.set('contype','4');rod.set('conaffinity','11')
    gripper=root.find(".//body[@name='r_gripper']")
    # Bottom of the real rod, in the gripper frame. This site is measurement only.
    ET.SubElement(gripper,'site',name='physical_tip',pos='-.1119 -.000218121 -.0981274',size='.001',rgba='1 0 0 1')
    # Adjacent/assembled parts intentionally overlap in this procedural visual
    # model; exclude those assembly pairs, retain nonadjacent arm collisions.
    contact=ET.SubElement(root,'contact')
    assembly=[('r_base','r_shoulder'),('r_shoulder','r_upper_arm'),('r_upper_arm','r_lower_arm'),('r_lower_arm','r_wrist'),('r_wrist','r_gripper'),('r_gripper','r_moving_jaw'),('r_wrist','r_moving_jaw'),('r_lower_arm','r_gripper')]
    for a,b in assembly:ET.SubElement(contact,'exclude',body1=a,body2=b)
    actuators=root.find('actuator')
    for i in range(1,7):ET.SubElement(actuators,'motor',name=f'joint_motor_{i}',joint=f'r_j{i}',gear='1',ctrllimited='true',ctrlrange=f'{-cfg.torque_limit} {cfg.torque_limit}')
    return ET.tostring(root,encoding='unicode'),mesh

class Plant:
    def __init__(self,cfg,seed,video=None):
        self.cfg=cfg;self.rng=np.random.default_rng(seed);self.video=video;self.writer=None
        if cfg.plant=='arm':
            xml,mesh=dynamic_xml(cfg)
            # Reuse rendering and IK methods, initialize against the new model.
            self.env=object.__new__(arm_module.PushTEnv)
            self.env.model=mujoco.MjModel.from_xml_string(xml);self.env.data=mujoco.MjData(self.env.model)
            for attr,name,kind in [('_block_body','block',mujoco.mjtObj.mjOBJ_BODY),('_goal_body','goal',mujoco.mjtObj.mjOBJ_BODY),('_pusher_body','pusher',mujoco.mjtObj.mjOBJ_BODY),('_pusher_site','pusher',mujoco.mjtObj.mjOBJ_SITE)]:
                setattr(self.env,attr,mujoco.mj_name2id(self.env.model,kind,name))
            self.env._init_visual_overlay('rgb_array',100000,mesh)
        else:
            self.env=arm_module.PushTEnv(render_mode='rgb_array',max_episode_steps=100000)
        self.model,self.data=self.env.model,self.env.data
        self.block_geoms={self.gid(n) for n in ['block_stem','block_top']}
        if cfg.plant=='planar':
            for g in self.block_geoms:self.model.geom_size[g,2]=.004
            self.model.dof_damping[:3]=np.array([.8,.8,.002])*cfg.damping
            body=self.env._block_body
            self.model.body_mass[body]*=cfg.mass;self.model.body_inertia[body]*=cfg.mass
        self.rod=self.gid('r_tool_rod' if cfg.plant=='arm' else 'pusher_collision')
        self.arm_geoms={i for i in range(self.model.ngeom) if (mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,i) or '').startswith('r_') and self.model.geom_bodyid[i]!=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'r_base')}
        self.qadr=self.env._arm_qpos_adr;self.dadr=self.env._arm_dof_adr
        if cfg.plant=='arm':
            self.tip_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,'physical_tip')
            self.ikdata=mujoco.MjData(self.model)
        self.queue=collections.deque([np.zeros(2) for _ in range(cfg.delay_ticks)])
        self.phase='RESET';self.mode=-1;self.tick=0;self.trace=[];self.transfers=[]
        self.distance=0.;self.transit_contacts=0;self.arm_contacts=0;self.max_speed=0.;self.max_accel=0.;self.max_torque=0.;self.max_track=0.;self.min_clearance=np.inf
        self.saturated=0;self.unsafe_pairs=set();self.last_joint_velocity=np.zeros(6)
        if video:
            import imageio.v2 as imageio
            self.writer=imageio.get_writer(str(video),fps=25,codec='libx264',quality=8,macro_block_size=16)
    def gid(self,name):return mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,name)
    def pose(self):
        if self.cfg.plant=='planar':return self.data.qpos[:3].copy()
        b=self.env._block_body;R=self.data.xmat[b].reshape(3,3)
        return np.r_[self.data.xpos[b,:2],np.arctan2(R[1,0],R[0,0])]
    def observe(self):
        q=self.pose();q[:2]+=self.rng.normal(0,self.cfg.noise_mm*.001,2);q[2]+=self.rng.normal(0,np.deg2rad(self.cfg.noise_deg));return q
    def pusher(self):
        return self.data.site_xpos[self.tip_id,:2].copy() if self.cfg.plant=='arm' else self.data.qpos[3:5].copy()
    def contact(self):
        return any((c.geom1==self.rod and c.geom2 in self.block_geoms) or (c.geom2==self.rod and c.geom1 in self.block_geoms) for c in self.data.contact)
    def solve(self,target):
        # Every IK qpos write is isolated from the physical simulation state.
        live=self.env.data;self.ikdata.qpos[:]=live.qpos;self.ikdata.qvel[:]=0
        self.env.data=self.ikdata
        try:q=self.env._solve_ik_dls(np.r_[target,.107],self.ik_warm)
        finally:self.env.data=live
        self.ik_warm=q.copy();return np.r_[q,.25]
    def reset(self,q,p=None):
        p=np.array([-.05,-.055]) if p is None else np.asarray(p)
        if self.cfg.plant=='planar':self.env.reset(options={'block_xy':q[:2],'block_yaw':q[2],'pusher_xy':p})
        else:
            mujoco.mj_resetData(self.model,self.data)
            # The free block and robot are initialized once per episode.
            self.data.qpos[:7]=[*q[:2],.0041,np.cos(q[2]/2),0,0,np.sin(q[2]/2)]
            self.ik_warm=self.env._arm_init_q.copy();self.data.qpos[self.qadr]=np.r_[self.ik_warm,.25]
            mujoco.mj_forward(self.model,self.data)
            self.data.qpos[self.qadr]=self.solve(p)
            mujoco.mj_forward(self.model,self.data)
            self.target_xy=p.copy();self.qref=self.data.qpos[self.qadr].copy();self.vref=np.zeros(6)
        self.last_joint_velocity=self.data.qvel[self.dadr].copy()
    def step(self,v):
        if self.data.time>self.cfg.max_time:raise RuntimeError('time_limit')
        v=np.asarray(v);self.queue.append(v.copy());cmd=self.queue.popleft()
        before=self.pusher();substeps=int(round(.02/self.model.opt.timestep))
        if self.cfg.plant=='arm':
            self.target_xy+=cmd*.02
            # Bounded tracking reference avoids integrating a large error into
            # an unreachable target if contact or actuator limits block motion.
            offset=self.target_xy-before
            if np.linalg.norm(offset)>.004:self.target_xy=before+offset/np.linalg.norm(offset)*.004
            desired=self.solve(self.target_xy)
            err=desired-self.qref
            vdes=np.sign(err)*np.minimum(self.cfg.joint_speed,np.minimum(np.sqrt(2*self.cfg.joint_accel*abs(err)),abs(err)/.02))
            self.vref+=np.clip(vdes-self.vref,-self.cfg.joint_accel*.02,self.cfg.joint_accel*.02)
            self.qref+=self.vref*.02
        for _ in range(substeps):
            if self.cfg.plant=='arm':
                torque=40*(self.qref-self.data.qpos[self.qadr])-.35*self.data.qvel[self.dadr]+self.data.qfrc_bias[self.dadr]
                self.saturated+=int(np.any(abs(torque)>self.cfg.torque_limit))
                torque=np.clip(torque,-self.cfg.torque_limit,self.cfg.torque_limit)
                self.data.ctrl[:2]=0;self.data.ctrl[2:]=torque
                self.max_torque=max(self.max_torque,float(max(abs(torque))))
            else:self.data.ctrl[:]=np.clip(cmd,-.04,.04)
            mujoco.mj_step(self.model,self.data)
            if self.phase=='TRANSIT' and self.contact():self.transit_contacts+=1
            if self.cfg.plant=='arm':
                velocity=self.data.qvel[self.dadr].copy()
                self.max_speed=max(self.max_speed,float(max(abs(velocity))))
                self.max_accel=max(self.max_accel,float(max(abs(velocity-self.last_joint_velocity)))/self.model.opt.timestep)
                self.last_joint_velocity=velocity
                for c in self.data.contact:
                    pair={int(c.geom1),int(c.geom2)}
                    intentional=self.rod in pair and bool(pair & self.block_geoms)
                    if pair & self.arm_geoms and not intentional:
                        self.arm_contacts+=1;self.unsafe_pairs.add(tuple(sorted(mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,g) for g in pair)))
        mujoco.mj_forward(self.model,self.data)
        if self.cfg.plant=='planar' and self.writer:self.env._track_pusher_with_arm()
        self.distance+=float(np.linalg.norm(self.pusher()-before));self.tick+=1
        if self.cfg.plant=='arm':self.max_track=max(self.max_track,float(np.linalg.norm(self.pusher()-self.target_xy)))
        if self.phase=='TRANSIT':
            clearance=base.shape(self.pose()).distance(Point(self.pusher()))-.004
            self.min_clearance=min(self.min_clearance,clearance)
            if self.transit_contacts:raise RuntimeError('transit_collision')
        if self.cfg.plant=='arm' and self.arm_contacts:raise RuntimeError('arm_collision')
        if self.cfg.plant=='arm':
            if self.max_speed>self.cfg.joint_speed*1.05:raise RuntimeError('joint_speed_limit')
            if self.max_accel>self.cfg.actual_accel_stop:raise RuntimeError('joint_acceleration_limit')
            limits=self.model.jnt_range[self.env._arm_joint_ids]
            if np.any(self.data.qpos[self.qadr]<limits[:,0]) or np.any(self.data.qpos[self.qadr]>limits[:,1]):raise RuntimeError('joint_position_limit')
            if self.data.xpos[self.env._block_body,2]>.015:raise RuntimeError('object_lifted')
            R=self.data.xmat[self.env._block_body].reshape(3,3)
            if R[2,2]<np.cos(np.deg2rad(15)):raise RuntimeError('object_tipped')
        q=self.pose()
        self.trace.append([float(self.data.time),*q,*self.pusher(),self.mode,self.phase,int(self.contact()),*self.data.qpos[self.qadr]])
        if self.writer and self.tick%2==0:self.frame()
    def frame(self):
        from PIL import Image,ImageDraw,ImageFont
        a=self.env.render(camera='3d');b=self.env.render(camera='2d');im=Image.new('RGB',(1024,608),'#131b27')
        im.paste(Image.fromarray(a),(0,96));im.paste(Image.fromarray(b),(512,96));d=ImageDraw.Draw(im)
        f=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17);q=self.pose()
        title='ACTUATED JOINTS + FREE RIGID BODY' if self.cfg.plant=='arm' else 'PLANAR CONTACT + IK DISPLAY'
        lines=[f'{title} | {self.cfg.metric} | {self.phase} | mode {self.mode+1}',f't={self.data.time:.2f}s  position={np.linalg.norm(q[:2])*1000:.2f}mm  angle={np.rad2deg(q[2]):.2f}deg  overlap={base.coverage(q):.1%}',f'transit collisions={self.transit_contacts} | arm collisions={self.arm_contacts} | travel={self.distance:.3f}m']
        for i,line in enumerate(lines):d.text((12,8+i*28),line,font=f,fill='white')
        self.writer.append_data(np.asarray(im))
    def move(self,target,phase,speed=None):
        self.phase=phase;speed=self.cfg.speed if speed is None else speed
        for _ in range(650):
            delta=target-self.pusher()
            if np.linalg.norm(delta)<(.0003 if self.cfg.plant=='arm' else .00015):return
            v=delta*8.;v*=min(1.,speed/max(np.linalg.norm(v),1e-12));self.step(v)
        raise RuntimeError('waypoint_timeout')
    def settle(self):
        self.phase='SETTLE'
        for _ in range(25 if self.cfg.plant=='arm' else 15):self.step(np.zeros(2))
    def close(self):
        if self.writer:self.writer.close()
        self.env.close()

class Policy:
    def __init__(self,metric):
        artifact=PACKAGE/'media/so101_contact'
        self.C=np.load(artifact/'physical_training.npz')['C'];self.metric=metric
        self.Ct=torch.tensor(self.C[None],dtype=torch.float32);self.Ct[:,:,2]*=2
        self.valid=torch.ones((1,12),dtype=torch.bool)
        if metric!='identity':
            file=artifact/'contact_full_m.pt' if metric=='finetuned' else base.VENDOR/'pretrained_full_m.pt'
            ck=torch.load(file,map_location='cpu',weights_only=True);self.actor=base.FullMActor(base.FullMConfig(**ck['cfg']));self.actor.load_state_dict(ck['actor']);self.actor.eval()
            with torch.no_grad():self.cond=self.actor.encoder(self.Ct,self.valid).detach()
    def value(self,q):
        e=torch.tensor(np.array([base.error(x) for x in np.atleast_2d(q)]),dtype=torch.float32);D=base.coordinates(e)
        if self.metric=='identity':return D.square().sum(-1).numpy()
        with torch.no_grad():return self.actor.value(D,self.cond.expand(len(D),-1)).numpy()
    def choose(self,q,previous,hold,pusher):
        travels=np.array([.0004,.001,.002,.0035]);world=(base.rot(q[2])@self.C[:,:2].T).T
        pred=q[None,None,:]+np.c_[world,self.C[:,2]][:,None,:]*travels[None,:,None]
        D=base.coordinates(torch.tensor(base.error(q)[None],dtype=torch.float32)).requires_grad_(True)
        value=D.square().sum() if self.metric=='identity' else self.actor.value(D,self.cond).sum()
        grad,=torch.autograd.grad(value,D)
        flow=np.c_[world/.05,np.cos(q[2]/2)*self.C[:,2]/np.sqrt(2)]@grad.detach().numpy()[0]
        V=float(value.detach());vp=self.value(pred.reshape(-1,3)).reshape(12,4)
        physical=(np.linalg.norm(pred[:,:,:2],axis=-1)<.060)&self.valid.cpu().numpy()[0,:,None]
        score=vp.copy();score[(flow[:,None]>=0)|(vp>=V-1e-8)|~physical]=np.inf
        recovery=False
        if not np.isfinite(score).any():
            recovery=True;score=abs(pred[:,:,2]);score[(score>=abs(q[2]))|~physical]=np.inf
            if not np.isfinite(score).any():raise RuntimeError('no_descending_mode')
        if hold and previous is not None and np.isfinite(score[previous]).any():
            best=np.unravel_index(np.argmin(score),score.shape);k=int(np.argmin(score[previous]))
            # Keep a feasible contact if it delivers at least 25% of the best
            # predicted descent; this is explicit and identical across metrics.
            reference=abs(q[2]) if recovery else V
            if reference-score[previous,k]>=.25*(reference-score[best]):return previous,float(travels[k]),V,recovery
        # Among near-equivalent descent candidates prefer shorter repositioning.
        if hold and not recovery:
            centers=q[:2]+(base.rot(q[2])@(base.POINTS+base.NORMALS*.0058).T).T
            best=float(np.min(score));budget=.05*max(V-best,0.)
            score+=budget*np.minimum(np.linalg.norm(centers-pusher,axis=1)/.08,1.)[:,None]
        m,k=np.unravel_index(np.argmin(score),score.shape)
        return int(m),float(travels[k]),V,recovery

def transfer(s,m,previous,policy):
    s.settle();q0=s.pose();t=s.data.time;v0=float(policy.value(q0)[0])
    if previous is not None:
        s.move(s.pusher()+base.rot(q0[2])@base.NORMALS[previous]*.004,'RETRACT',.010)
    q=s.observe();end=q[:2]+base.rot(q[2])@(base.POINTS[m]+base.NORMALS[m]*.0058)
    route=base.path(s.pusher(),end,q)
    for point in route[1:]:s.move(point,'TRANSIT')
    drift=s.pose()-q0;v1=float(policy.value(s.pose())[0])
    s.transfers.append(dict(time_s=float(s.data.time-t),drift_mm=float(np.linalg.norm(drift[:2])*1000),drift_deg=float(abs(np.rad2deg(drift[2]))),V_before=v0,V_after=v1,mode_from=previous,mode_to=m))
    if np.linalg.norm(drift[:2])>.0002:raise RuntimeError('transfer_drift_over_0.2mm')
    s.mode=m

def push(s,m,travel):
    inward=-base.rot(s.observe()[2])@base.NORMALS[m]
    s.phase='APPROACH'
    for _ in range(100):
        if s.contact():break
        s.step(inward*.004)
    if not s.contact():raise RuntimeError('contact_not_established')
    s.phase='PUSH'
    for _ in range(max(1,int(round(travel/.006/.02)))):s.step(inward*.006)
    s.settle()

def rollout(cfg,q,seed,policy=None,video=None,telemetry=None):
    policy=policy or Policy(cfg.metric);s=Plant(cfg,seed,video);previous=None;reason='action_limit';retained=0;changed=0;descent=[];initial=np.asarray(q).copy();actions=0
    try:
        s.reset(initial)
        for i in range(cfg.max_actions):
            if base.reached(s.pose()):reason='success';break
            obs=s.observe();m,travel,V,recovery=policy.choose(obs,previous,cfg.persistence,s.pusher())
            surface_gap=base.shape(s.pose()).distance(Point(s.pusher()))-.004
            same=m==previous and (s.contact() or -.0003<=surface_gap<=.0006)
            if same:
                local=base.rot(-s.pose()[2])@(s.pusher()-s.pose()[:2]);tangent=np.array([-base.NORMALS[m,1],base.NORMALS[m,0]])
                same=abs(float((local-base.POINTS[m])@tangent))<.0015
            if cfg.persistence and same:retained+=1
            else:
                if previous is not None and previous!=m:changed+=1
                transfer(s,m,previous,policy)
            before=float(policy.value(s.pose())[0]);push(s,m,travel);after=float(policy.value(s.pose())[0]);descent.append(after-before)
            previous=m;actions=i+1
        s.settle()
        if base.reached(s.pose()):reason='success'
        s.phase='HOLD'
        # Verify the same one-second terminal hold with and without rendering.
        for _ in range(50):s.step(np.zeros(2))
        if reason=='success' and not base.reached(s.pose()):reason='terminal_hold_failed'
    except RuntimeError as ex:reason=str(ex)
    final=s.pose()
    report=dict(config=asdict(cfg),seed=seed,initial=initial.tolist(),final=final.tolist(),success=reason=='success' and s.transit_contacts==0 and s.arm_contacts==0,reason=reason,actions=actions,sim_time_s=float(s.data.time),distance_m=s.distance,mode_changes=changed,repositions=len(s.transfers),retained_actions=retained,position_mm=float(np.linalg.norm(final[:2])*1000),angle_deg=float(np.rad2deg(final[2])),coverage=base.coverage(final),transit_contact_substeps=s.transit_contacts,arm_contact_substeps=s.arm_contacts,unsafe_pairs=sorted(s.unsafe_pairs),min_transit_clearance_mm=None if not np.isfinite(s.min_clearance) else s.min_clearance*1000,max_joint_speed_rad_s=s.max_speed,max_joint_acceleration_rad_s2=s.max_accel,max_torque_nm=s.max_torque,torque_saturation_substeps=s.saturated,max_tracking_error_mm=s.max_track*1000,transfers=s.transfers,push_delta_V=descent)
    if video:
        s.phase='SUCCESS' if report['success'] else 'FAILED';s.frame();save(video.with_suffix('.json'),report)
        with video.with_suffix('.csv').open('w') as f:
            f.write('time,x,y,yaw,pusher_x,pusher_y,mode,phase,contact,j1,j2,j3,j4,j5,j6\n')
            for row in s.trace:f.write(','.join(map(str,row))+'\n')
    if telemetry is not None:
        t=np.r_[0.,[row[0] for row in s.trace]]
        poses=np.vstack([initial,np.array([row[1:4] for row in s.trace]).reshape(-1,3)])
        # Include a failure state even when a safety stop occurs before trace append.
        if s.data.time>t[-1]+1e-10:
            t=np.r_[t,s.data.time];poses=np.vstack([poses,final])
        values=np.concatenate([policy.value(part) for part in np.array_split(poses,max(1,len(poses)//1024+1))])
        err=np.sum((poses[:,:2]/.05)**2,axis=1)+2*np.sin(poses[:,2]/2)**2
        report['integrated_error']=float(np.sum(.5*(err[:-1]+err[1:])*np.diff(t)))
        report['complete_V_increase_fraction']=float(np.mean(np.diff(values)>1e-7)) if len(values)>1 else 0.
        np.savez_compressed(telemetry,time=t,pose=poses,V=values,error=err)
    s.close();return report

def cases(seed,n,extent=.018,angle=.55):
    rng=np.random.default_rng(seed);return [np.r_[rng.uniform(-extent,extent,2),rng.uniform(-angle,angle)] for _ in range(n)]

def run_group(out,name,cfg,initials,seed):
    file=out/f'{name}.json'
    if file.exists():
        existing=json.loads(file.read_text())
        if asdict(Config(**existing['config']))!=asdict(cfg) or existing['initials']!=[q.tolist() for q in initials] or existing.get('source_sha256')!=SOURCE_SHA:raise ValueError('Refusing to mix experiment configurations or source versions')
        rows=existing['episodes']
    else:rows=[]
    policy=Policy(cfg.metric)
    for i,q in enumerate(initials[len(rows):],start=len(rows)):
        r=rollout(cfg,q,seed+i,policy);r['episode']=i;rows.append(r)
        save(file,dict(config=asdict(cfg),seed=seed,source_sha256=SOURCE_SHA,initials=[q.tolist() for q in initials],episodes=rows))
        print(name,i,r['success'],r['reason'],f"{r['sim_time_s']:.1f}s",flush=True)
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('suite',choices=['ablation','robustness','arm','smoke']);ap.add_argument('--output',type=Path,default=DEFAULT_OUT);ap.add_argument('--n',type=int,default=30);ap.add_argument('--video',action='store_true');a=ap.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
    (out/'source_snapshot.py').write_text(SOURCE_TEXT)
    if a.suite=='smoke':
        for plant in ['planar','arm']:
            r=rollout(Config(plant=plant,max_actions=10,max_time=40),np.array([-.01,0,.1]),393,video=out/f'{plant}_smoke.mp4' if a.video else None)
            save(out/f'{plant}_smoke.json',r);print(plant,r['reason'],r['unsafe_pairs'],r['final'],flush=True)
    elif a.suite=='ablation':
        qs=cases(20260925,a.n)
        for metric in ['identity','pretrained','finetuned']:
            for hold in [False,True]:run_group(out,f'ablation_{metric}_{"hold" if hold else "reset"}',Config(metric=metric,persistence=hold),qs,26000)
    elif a.suite=='robustness':
        qs=cases(20260926,a.n)
        settings={'nominal':{},'damping_low':{'damping':.5},'damping_high':{'damping':2.},'mass_low':{'mass':.5},'mass_high':{'mass':2.},'delay_60ms':{'delay_ticks':3},'noise':{'noise_mm':.4,'noise_deg':1.},'combined':{'damping':1.5,'mass':1.5,'delay_ticks':2,'noise_mm':.3,'noise_deg':.8}}
        for name,settings in settings.items():run_group(out,'robust_'+name,Config(**settings),qs,27000)
        qs=[np.array([x,y,np.deg2rad(a)]) for x in [-.025,0,.025] for y in [-.025,0,.025] for a in [-120,-60,0,60,120]]
        run_group(out,'boundary',Config(),qs,28000)
    elif a.suite=='arm':
        qs=cases(20260927,a.n,extent=.012,angle=.35)
        for name,settings in {'nominal':{},'friction_low':{'friction':.3},'friction_high':{'friction':.9},'mass_high':{'mass':2.}}.items():
            rows=run_group(out,'arm_'+name,Config(plant='arm',speed=.012,**settings),qs,29000)
            if a.video:
                r=next((r for r in rows if r['success']),rows[0]);rollout(Config(**r['config']),np.array(r['initial']),r['seed'],video=out/f'arm_{name}_demo.mp4')
    save(out/'provenance.json',dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),mujoco=mujoco.__version__,torch=torch.__version__,notes='Arm inertias, limits and PD gains are explicit simulation assumptions, not identified SO101 hardware parameters. Planar damping sweep is not a physical friction sweep.'))

if __name__=='__main__':main()
