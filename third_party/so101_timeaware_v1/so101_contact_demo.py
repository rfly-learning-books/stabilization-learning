"""Physical-contact adaptation of the paper's Full-M certificate and SO101 demo.

The arm is an IK visual overlay; the planar velocity-driven rod is physical.
Only resets/calibration resets set object poses. Rollouts never teleport bodies.
"""
import os
os.environ.setdefault('MUJOCO_GL', 'glfw')
import sys, json, argparse, hashlib, heapq
from pathlib import Path
import numpy as np
import torch
import mujoco
from shapely.geometry import Polygon, Point, LineString
from shapely.affinity import rotate, translate
ROOT = Path(__file__).resolve().parents[1] / 'push_original'
VENDOR=Path(__file__).resolve().parent/'so101_vendor'
sys.path[:0] = [str(VENDOR), str(ROOT), str(ROOT/'contact_tasks')]
from dl_push_contact.envs.so101_push_t_env import PushTEnv
from dl_push.certificate.full_m_control import FullMActor, FullMConfig, coordinates

torch.set_num_threads(2)
POLY = Polygon([(-.004,-.015),(.004,-.015),(.004,.015),(.014,.015),(.014,.023),(-.014,.023),(-.014,.015),(-.004,.015)])
# Surface points and outward normals; 12 physically accessible contact modes.
POINTS = np.array([[-.01,.023],[0,.023],[.01,.023],[-.002,-.015],[0,-.015],[.002,-.015],[-.014,.017],[-.014,.021],[.014,.017],[.014,.021],[-.004,-.010],[.004,-.010]])
NORMALS = np.array([[0,1]]*3+[[0,-1]]*3+[[-1,0]]*2+[[1,0]]*2+[[-1,0],[1,0]])
DT=.02
def rot(a): return np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
def shape(q): return translate(rotate(POLY,q[2],origin=(0,0),use_radians=True),*q[:2])
def error(q): return np.r_[q[:2]/.05,np.arctan2(np.sin(q[2]),np.cos(q[2]))]
def coverage(q): return shape(q).intersection(POLY).area/POLY.area
def reached(q): return np.linalg.norm(q[:2])<.002 and abs(error(q)[2])<np.deg2rad(4) and coverage(q)>.90

class Scene:
    def __init__(self, record=None):
        self.env=PushTEnv(render_mode='rgb_array',max_episode_steps=100000)
        self.model,self.data=self.env.model,self.env.data
        # Correct the original 0.6-mm visual slab to the specified 8-mm part.
        for name in ['block_stem','block_top']:
            self.model.geom_size[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,name),2]=.004
        self.model.dof_damping[:3]=[.8,.8,.002]
        self.record=record; self.writer=None; self.rows=[]; self.tick=0; self.phase='RESET'; self.mode=-1
        self.unsafe=0; self.max_ik=0.; self.max_jump=0.; self.switches=[]
        if record:
            import imageio.v2 as imageio
            self.writer=imageio.get_writer(str(record),fps=25,codec='libx264',quality=8,macro_block_size=16)
    def reset(self,q,p=None):
        self.env.reset(options={'block_xy':q[:2],'block_yaw':q[2],'pusher_xy':[-.05,-.055] if p is None else p})
    def contacts(self):
        names=[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,n) for n in ['pusher_collision','block_stem','block_top']]
        return any(names[0] in [c.geom1,c.geom2] and (names[1] in [c.geom1,c.geom2] or names[2] in [c.geom1,c.geom2]) for c in self.data.contact)
    def step(self,v):
        old=self.data.qpos[3:5].copy()
        self.data.ctrl[:]=np.clip(v,-.04,.04)
        for _ in range(4):
            mujoco.mj_step(self.model,self.data)
            if self.phase=='TRANSIT' and self.contacts(): self.unsafe+=1
        if self.record:
            self.env._track_pusher_with_arm()
            self.max_ik=max(self.max_ik,self.env._arm_tcp_xy_error())
        self.max_jump=max(self.max_jump,float(np.linalg.norm(self.data.qpos[3:5]-old)))
        self.tick+=1
        q=self.data.qpos[:3].copy()
        self.rows.append([self.data.time,*q,*self.data.qpos[3:5],self.mode,self.phase,int(self.contacts())])
        if self.writer and self.tick%2==0:
            from PIL import Image,ImageDraw,ImageFont
            a=self.env.render(camera='3d'); b=self.env.render(camera='2d')
            im=Image.new('RGB',(1024,608),(19,27,39)); im.paste(Image.fromarray(a),(0,96));im.paste(Image.fromarray(b),(512,96))
            d=ImageDraw.Draw(im); f=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
            lines=[f'SO101 | Full-M contact adaptation | {self.phase} | mode {self.mode+1:02d}',
                f't={self.data.time:6.2f}s   position={np.linalg.norm(q[:2])*1000:5.2f}mm   angle={np.rad2deg(error(q)[2]):6.2f}deg   overlap={coverage(q)*100:5.1f}%',
                f'Continuous collision-checked transit | transit contacts: {self.unsafe} | planar contact + visual IK arm']
            for i,line in enumerate(lines):d.text((16,7+i*28),line,font=f,fill=(227,236,248))
            self.writer.append_data(np.asarray(im))
    def move(self,target,phase,speed=.025):
        self.phase=phase
        for _ in range(800):
            delta=target-self.data.qpos[3:5]
            if np.linalg.norm(delta)<.00015:return
            v=delta*12.; v*=min(1.,speed/max(np.linalg.norm(v),1e-12))
            self.step(v)
        raise RuntimeError('Waypoint unreachable')
    def settle(self):
        self.phase='SETTLE'
        for _ in range(15):self.step(np.zeros(2))
    def close(self):
        if self.writer:self.writer.close()
        self.env.close()

def path(start,end,q):
    # Visibility graph around the Minkowski-expanded T; 0.9-mm extra margin.
    obstacle=shape(q).buffer(.0049,quad_segs=4)
    vertices=np.asarray(obstacle.buffer(.00025,join_style=2).exterior.coords)[:-1]
    nodes=np.vstack([start,end,vertices]); n=len(nodes)
    if obstacle.contains(Point(start)) or obstacle.contains(Point(end)):raise RuntimeError('Path endpoint inside safety envelope')
    edges=[[] for _ in nodes]
    for i in range(n):
        for j in range(i):
            if np.any(nodes[[i,j],0]<-.064) or np.any(nodes[[i,j],0]>.069) or np.any(abs(nodes[[i,j],1])>.069):continue
            if not LineString([nodes[i],nodes[j]]).intersects(obstacle):
                w=float(np.linalg.norm(nodes[i]-nodes[j]));edges[i].append((j,w));edges[j].append((i,w))
    queue=[(0.,0,[])];visited=set()
    while queue:
        cost,i,route=heapq.heappop(queue)
        if i in visited:continue
        if i==1:return nodes[route+[i]]
        visited.add(i)
        for j,w in edges[i]:heapq.heappush(queue,(cost+w,j,route+[i]))
    raise RuntimeError('No collision-free path in workspace')

def transfer(s,m,previous):
    s.settle(); q0=s.data.qpos[:3].copy(); t0=s.data.time
    if previous is not None:
        outward=rot(q0[2])@NORMALS[previous]
        s.move(s.data.qpos[3:5].copy()+outward*.004,'RETRACT',.012)
    q=s.data.qpos[:3].copy(); R=rot(q[2]); end=q[:2]+R@(POINTS[m]+NORMALS[m]*.0058)
    route=path(s.data.qpos[3:5].copy(),end,q)
    for waypoint in route[1:]:s.move(waypoint,'TRANSIT')
    drift=s.data.qpos[:3]-q0
    s.switches.append(dict(time_s=float(s.data.time-t0),position_drift_mm=float(np.linalg.norm(drift[:2])*1000),angle_drift_deg=float(abs(np.rad2deg(drift[2])))))
    s.mode=m

def push(s,m,travel):
    q=s.data.qpos[:3].copy(); inward=-rot(q[2])@NORMALS[m]
    # Contact is rebuilt continuously, including the pre-contact gap.
    s.phase='APPROACH'
    for _ in range(80):
        if s.contacts():break
        s.step(inward*.006)
    s.phase='PUSH'
    for _ in range(max(1,int(round(travel/.008/DT)))):s.step(inward*.008)
    s.settle()

def collect(out):
    s=Scene(); responses=[]; inputs=[]; successors=[]
    rng=np.random.default_rng(193)
    for m in range(12):
        trials=[]
        for j in range(12):
            q=np.r_[rng.uniform(-.018,.018,2),rng.uniform(-.65,.65)]
            p=q[:2]+rot(q[2])@(POINTS[m]+NORMALS[m]*.0058)
            s.reset(q,p);s.mode=m
            push(s,m,.002)
            qn=s.data.qpos[:3].copy();delta=qn-q
            trials.append(np.r_[rot(-q[2])@delta[:2],delta[2]]/.002)
            inputs.append(q);successors.append(qn)
        responses.append(np.median(trials,axis=0))
        print('calibrated mode',m+1,np.round(responses[-1],3),flush=True)
    s.close(); C=np.array(responses)
    np.savez(out/'physical_training.npz',C=C,q=inputs,qn=successors)
    return C,np.array(inputs),np.array(successors)

def adapt(out,C,q,qn,checkpoint):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    actor=FullMActor(FullMConfig(**ck['cfg']));actor.load_state_dict(ck['actor'])
    # Keep the paper's architecture/codec; normalize the 38-mm workpiece scale.
    Ct=torch.tensor(C[None],dtype=torch.float32);Ct[:,:,2]*=2
    valid=torch.ones((1,12),dtype=torch.bool)
    e=torch.tensor(np.array([error(x) for x in q]),dtype=torch.float32)
    en=torch.tensor(np.array([error(x) for x in qn]),dtype=torch.float32)
    favorable=(en.square().sum(-1)<e.square().sum(-1))
    optimizer=torch.optim.Adam(actor.parameters(),lr=1e-4);losses=[]
    for epoch in range(150):
        cond=actor.encoder(Ct,valid).expand(len(e),-1)
        V=actor.value(coordinates(e),cond);Vn=actor.value(coordinates(en),cond)
        descent=torch.relu((Vn-V)[favorable]+.001).square().mean()
        _,M=actor.conditioned(coordinates(e),cond)
        # Scale regularization and smoothness retain a useful translational metric.
        fit=(V-e.square().sum(-1)).square().mean()
        _,Mn=actor.conditioned(coordinates(e+.01),cond)
        loss=descent+.05*fit+.001*(Mn-M).square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
    torch.save(dict(actor=actor.state_dict(),cfg=ck['cfg'],C=torch.tensor(C),error_length_scale_m=.05,encoder_omega_multiplier=2.,source=str(checkpoint),training='contact-transition certificate fine-tuning; not a new PPO run'),out/'contact_full_m.pt')
    (out/'training.json').write_text(json.dumps(dict(transitions=len(q),epochs=len(losses),loss=losses,source_checkpoint=str(checkpoint),source_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()),indent=2))
    return actor.eval(),Ct,valid

def choose(actor,C,Ct,valid,q):
    travels=np.array([.0004,.001,.002,.0035])
    pred=q[None,None,:]+np.concatenate([np.broadcast_to((rot(q[2])@C[:,:2].T).T[:,None,:],(12,4,2)),np.broadcast_to(C[:,None,2:3],(12,4,1))],axis=2)*travels[None,:,None]
    cond=actor.encoder(Ct,valid).detach()
    e=torch.tensor(error(q)[None],dtype=torch.float32)
    D=coordinates(e).requires_grad_(True)
    value=actor.value(D,cond)
    grad,=torch.autograd.grad(value.sum(),D)
    world=(rot(q[2])@C[:,:2].T).T
    tangent=np.c_[world/.05,np.cos(q[2]/2)*C[:,2]/np.sqrt(2)]
    flow=tangent@grad.detach().numpy()[0]
    with torch.no_grad():
        ep=torch.tensor(np.array([error(p) for p in pred.reshape(-1,3)]),dtype=torch.float32)
        V=float(actor.value(coordinates(e),cond)[0]); vp=actor.value(coordinates(ep),cond.expand(len(ep),-1)).numpy().reshape(12,4)
    score=vp.copy()
    score[(flow[:,None]>=0)|(vp>=V-1e-8)]=np.inf
    score[np.linalg.norm(pred[:,:,:2],axis=-1)>.036]=np.inf
    if not np.isfinite(score).any():
        # The paper's separate attitude recovery when the certificate grid fails.
        score=abs(pred[:,:,2])
        score[abs(pred[:,:,2])>=abs(q[2])]=np.inf
        if not np.isfinite(score).any():raise RuntimeError('No descending physical mode')
    m,k=np.unravel_index(np.argmin(score),score.shape)
    return m,float(travels[k]),V

def rollout(actor,C,Ct,valid,q,record=None,max_actions=100):
    s=Scene(record);s.reset(q);previous=None;reason='timeout';initial=q.copy()
    try:
        for action in range(max_actions):
            q=s.data.qpos[:3].copy()
            if reached(q):reason='success';break
            m,travel,V=choose(actor,C,Ct,valid,q)
            transfer(s,m,previous);push(s,m,travel);previous=m
        s.settle()
        if reached(s.data.qpos[:3]):reason='success'
        s.phase=reason.upper()
        if record:
            for _ in range(50):s.step(np.zeros(2))
    except RuntimeError as ex:reason=str(ex)
    q=s.data.qpos[:3].copy()
    report=dict(success=reason=='success' and reached(q) and s.unsafe==0,reason=reason,initial=initial.tolist(),final=q.tolist(),position_mm=float(np.linalg.norm(q[:2])*1000),angle_deg=float(np.rad2deg(error(q)[2])),coverage=coverage(q),actions=action+1,sim_time_s=float(s.data.time),transit_contact_substeps=s.unsafe,max_step_mm=s.max_jump*1000,max_ik_error_mm=s.max_ik*1000,switches=s.switches)
    if record:
        with record.with_suffix('.json').open('w') as f:json.dump(report,f,indent=2)
        with record.with_suffix('.csv').open('w') as f:
            f.write('time,x,y,yaw,pusher_x,pusher_y,mode,phase,contact\n')
            for row in s.rows:f.write(','.join(map(str,row))+'\n')
    s.close();return report

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--episodes',type=int,default=20);ap.add_argument('--seed',type=int,default=20260921);ap.add_argument('--reuse',action='store_true');ap.add_argument('--video',action='store_true');a=ap.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    checkpoint=VENDOR/'pretrained_full_m.pt'
    if not checkpoint.exists():checkpoint=ROOT/'runs/full_m_final_20260913/final.pt'
    if a.reuse:
        data=np.load(out/'physical_training.npz');C,q,qn=data['C'],data['q'],data['qn']
    else:C,q,qn=collect(out)
    torch.manual_seed(193);actor,Ct,valid=adapt(out,C,q,qn,checkpoint)
    rng=np.random.default_rng(a.seed); reports=[]
    for i in range(a.episodes):
        q=np.r_[rng.uniform(-.018,.018,2),rng.uniform(-.55,.55)]
        r=rollout(actor,C,Ct,valid,q);r['episode']=i;reports.append(r)
        print('evaluation',i,r['success'],r['reason'],round(r['position_mm'],2),round(r['angle_deg'],2),flush=True)
        (out/'evaluation.json').write_text(json.dumps(dict(seed=a.seed,episodes=reports,success_rate=sum(x['success'] for x in reports)/len(reports)),indent=2))
    if a.video:
        successes=[r for r in reports if r['success']]
        for r in sorted(successes,key=lambda x:abs(x['sim_time_s']-45))[:2]:
            print('rendering episode',r['episode'],flush=True)
            rollout(actor,C,Ct,valid,np.array(r['initial']),out/f"success_{r['episode']:02d}.mp4")

if __name__=='__main__':main()
