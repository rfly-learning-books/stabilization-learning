"""Measured J/A/P instantaneous quadratic-CLF demonstrations.

Uses a fixed Q and calibrated primitives; no integral/history term or learned
actor. Mode changes are executed through real release/transit/engage motion.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import mujoco
from shapely.geometry import shape as read_shape,Point,Polygon
from .library import ModeLibrary
from .fixture import Fixture
from .math import rot2,shape_error,response_in_error_frame
from .material_contact import material_contact_velocity
from .transfer import _visibility_path
from .evaluation import json_safe


def simulate(name,output,initial=(-.045,-.035,-.22),goal=(.04,.035,.12),max_time=70.,render=False,workspace_guard=False):
    source=Path('runs/mode_library_safe_reselected_20260911')/name
    lib=ModeLibrary.load(source/'library');poly=read_shape(json.loads((source/'polygon.json').read_text()))
    model=mujoco.MjModel.from_xml_path(str(source/'model.xml'));f=Fixture(model)
    Q=np.diag([.75,.75,1.5]);ell=.1;rate=.06;dt=model.opt.timestep;control_dt=.02
    pure=lib.valid&(np.abs(lib.c[2])/np.maximum(np.linalg.norm(lib.c[:2],axis=0),1e-9)<.15)
    radius=.018
    f.reset(lib.contact_points[0],lib.normals[0],radius,initial[2])
    f.d.qpos[f.oq[:2]]+=np.array(initial[:2]);f.d.qpos[f.pq]=[-.19,-.18];f.d.qvel[f.pv]=0
    mujoco.mj_forward(model,f.d)
    goal=np.asarray(goal);epoch=float(f.d.time)
    stage='JOINT';execution='TRANSIT';mode=-1;requested=None;path=[];wp=0
    switches=0;events=[];rows=[];status='TIMEOUT';stage_enter=0.;hold=0.;push_age=0.;last_command=np.zeros(2)
    valid_violations=0;valid_checks=0;first=True;transit_contact_steps=0;transit_peak_force=0.
    def error():
        p,yaw,v,omega,ball=f.state();D=shape_error(np.r_[p,yaw],goal,ell)
        return p,yaw,v,omega,ball,D
    def projection():return Q if stage=='JOINT' else (np.diag([0,0,Q[2,2]]) if stage=='ATTITUDE' else np.diag([Q[0,0],Q[1,1],0]))
    def anchor(i):return lib.contact_points[i]+lib.normals[i]*lib.contact_offset[i]
    def choose(D,p,yaw,ball):
        P=projection();V=float(D@P@D);options=[]
        for i in lib.active_indices:
            if stage=='POSITION' and not pure[i]:continue
            g=response_in_error_frame(D,lib.c[:,i],ell);beta=float(2*D@P@g)
            if beta>=-1e-9:continue
            u=max(float(lib.u_min[i]),rate*V/(-beta))
            if u>lib.u_max[i]+1e-10:continue
            future=D+g*u*.2
            predicted_p=p+rot2(yaw)@lib.c[:2,i]*u*.5
            predicted_yaw=yaw+lib.c[2,i]*u*.5
            if workspace_guard and np.max(np.abs(np.array(poly.exterior.coords)@rot2(predicted_yaw).T+predicted_p))>.224:continue
            if workspace_guard and np.max(np.abs(p+rot2(yaw)@(anchor(i)+lib.normals[i]*.014)))>.224:continue
            if stage=='POSITION' and abs(future[2])>np.deg2rad(3):continue
            if stage=='ATTITUDE' and np.linalg.norm(future[:2])*ell>.17:continue
            distance=np.linalg.norm(p+rot2(yaw)@anchor(i)-ball)
            cost=(u/.05)**2+.8*(i!=mode)+.5*distance/.2
            if stage=='ATTITUDE':cost+=.8*np.linalg.norm(future[:2])**2
            options.append((cost,int(i),float(u),beta))
        if not options:return None
        # Persistence is allowed only while the current mode is feasible.
        current=[x for x in options if x[1]==mode]
        best=min(options)
        return current[0] if current and current[0][0]<=best[0]+.35 else best
    def build_path(target):
        p,yaw,_,_,ball,_=error();R=rot2(yaw)
        start=R.T@(ball-p);end=anchor(target)+lib.normals[target]*.009
        # Reviewed T side modes have about 1.17 mm clearance to the shoulder.
        # Keep a 0.5 mm collision margin and track each corner precisely.
        obstacle=poly.buffer(radius+.0005,join_style=2)
        workspace=Polygon([R.T@(np.array(v)-p) for v in [(-.225,-.225),(.225,-.225),(.225,.225),(-.225,.225)]])
        result=_visibility_path(start,end,obstacle,workspace)
        return result[1:] if result else None
    for tick in range(round(max_time/control_dt)):
        t=float(f.d.time)-epoch;p,yaw,v,omega,ball,D=error();pos=np.linalg.norm(p-goal[:2]);ang=abs(D[2])
        if np.max(np.abs(np.array(poly.exterior.coords)@rot2(yaw).T+p))>.232:
            status='OBJECT_WORKSPACE';break
        if stage=='JOINT' and pos<.045:
            stage='ATTITUDE' if ang>np.deg2rad(1) else 'POSITION'
            events.append(dict(t=t,event='stage',stage=stage));stage_enter=t
        if stage=='ATTITUDE' and ang<=np.deg2rad(.8):
            stage='POSITION';events.append(dict(t=t,event='stage',stage=stage));stage_enter=t
        elif stage=='POSITION' and ang>np.deg2rad(3):
            stage='ATTITUDE';events.append(dict(t=t,event='stage',stage=stage));stage_enter=t
        near=pos<=.004 and ang<=np.deg2rad(1.5)
        if near:
            execution='HOLD'
        elif execution=='HOLD':execution='PUSH' if mode>=0 else 'TRANSIT'
        speed=0.;decision=None
        if execution in ('PUSH','TRANSIT') and requested is None:
            decision=choose(D,p,yaw,ball)
            if decision is None:status='NO_FEASIBLE_MODE';break
            _,target,speed,beta=decision
            if target!=mode or first:
                requested=target;first=False
                if mode>=0:
                    execution='RELEASE';path=[(anchor(mode)+lib.normals[mode]*.014).tolist()];wp=0
                else:
                    path=build_path(requested)
                    if path is None:status='NO_TRANSIT_PATH';break
                    execution='TRANSIT';wp=0
            else:execution='PUSH'
        elif execution=='PUSH':
            decision=choose(D,p,yaw,ball)
            if decision is None:status='NO_FEASIBLE_MODE';break
            speed=decision[2]
        startV=float(D@projection()@D)
        for substep in range(round(control_dt/dt)):
            p,yaw,v,omega,ball,D=error();R=rot2(yaw)
            if execution=='HOLD':cmd=np.zeros(2)
            elif execution in ('RELEASE','TRANSIT'):
                target=p+R@np.array(path[wp]);delta=target-ball
                cmd=delta*15.;norm=np.linalg.norm(cmd)
                if norm>.10:cmd*=.10/norm
            elif execution in ('ENGAGE','PUSH'):
                active=requested if execution=='ENGAGE' else mode
                u=.02 if execution=='ENGAGE' else speed
                virtual=lib.contact_points[active]+lib.normals[active]*(lib.contact_offset[active]-radius)
                cmd=material_contact_velocity(p,yaw,v,omega,ball,virtual,lib.normals[active],radius,u)
            else:raise RuntimeError(execution)
            f.step(cmd);last_command=cmd
            if execution=='TRANSIT':
                nowp,nowyaw,_,_,_,_=error()
                forces=f.contacts(nowp,rot2(nowyaw),np.zeros(2),np.array([1.,0.]))[2]
                force=float(sum(forces));transit_peak_force=max(transit_peak_force,force)
                transit_contact_steps+=int(force>.01)
        p,yaw,v,omega,ball,D=error();t=float(f.d.time)-epoch
        contacts=[]
        if execution in ('ENGAGE','PUSH'):
            active=requested if execution=='ENGAGE' else mode
            contacts=f.contacts(p,rot2(yaw),lib.contact_points[active],lib.normals[active])[2]
        if execution in ('RELEASE','TRANSIT'):
            target=p+rot2(yaw)@np.array(path[wp])
            if np.linalg.norm(target-ball)<.0001:
                wp+=1
                if wp>=len(path):
                    if execution=='RELEASE':
                        path=build_path(requested)
                        if path is None:status='NO_TRANSIT_PATH';break
                        execution='TRANSIT';wp=0
                    else:execution='ENGAGE'
        elif execution=='ENGAGE' and len([x for x in contacts if x>.05])>=int(lib.face_mask[requested].sum()):
            if mode>=0 and mode!=requested:switches+=1
            mode=requested;requested=None;execution='PUSH';push_age=0.
            events.append(dict(t=t,event='mode',mode=mode,stage=stage,switches=switches))
        elif execution=='PUSH':push_age+=control_dt
        pos=float(np.linalg.norm(p-goal[:2]));ang=float(abs(D[2]))
        if execution=='HOLD' and pos<=.004 and ang<=np.deg2rad(1.5) and np.linalg.norm(v)<=.005 and abs(omega)<=.02:
            hold+=control_dt
        else:hold=0.
        P=projection();dv=float(2*D@P@np.r_[rot2(-goal[2])@v/ell,omega])
        if execution=='PUSH' and push_age>.2 and contacts and max(contacts)>.05:
            valid_checks+=1;valid_violations+=int(dv>1e-5)
        rows.append(dict(t=t,q=[*p,float(yaw)],pusher=ball.tolist(),velocity=v.tolist(),omega=float(omega),D=D.tolist(),V=float(D@Q@D),active_V=float(D@P@D),Vdot=dv,stage=stage,execution=execution,mode=mode,requested=requested,u=speed,command=last_command.tolist(),position_error=pos,angle_error=ang,switches=switches,hold=hold))
        if hold>=1.:
            status='SUCCESS';break
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    report=dict(shape=name,status=status,Q=Q.tolist(),initial=initial,goal=goal.tolist(),rate=rate,control_dt=control_dt,physics_dt=dt,elapsed_sim_s=float(f.d.time)-epoch,switches=switches,events=events,final=rows[-1] if rows else None,instantaneous_positive_Vdot_checks=valid_violations,contact_push_checks=valid_checks,transit_contact_steps=transit_contact_steps,transit_peak_force=transit_peak_force,workspace_guard=workspace_guard,rows=rows,source_library=str(source.resolve()),policy='fixed Q, J/A/P projections; scalar CLF interval; switching cost + feasible-mode persistence; real release/transit/engage; no history integral; no Q learning')
    (out/'report.json').write_text(json.dumps(json_safe(report),indent=2,allow_nan=False))
    print(json.dumps({k:report[k] for k in ('shape','status','elapsed_sim_s','switches')},ensure_ascii=False),flush=True)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--shape',default='T');p.add_argument('--output',required=True);p.add_argument('--max-time',type=float,default=70)
    args=p.parse_args();simulate(args.shape,args.output,max_time=args.max_time)

if __name__=='__main__':main()
