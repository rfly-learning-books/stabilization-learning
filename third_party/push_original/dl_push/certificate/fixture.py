"""Measured T modes with absolute normal speed and material-point tracking.

This is a separately versioned planar calibration fixture, not a free-body
validation. No object state is overwritten after reset/settling.
"""
from pathlib import Path
import argparse
import json
import sys
import xml.etree.ElementTree as ET

import numpy as np
import mujoco
from dl_push.certificate.assets import load_geometry, certificate_polygon
from dl_push.certificate.model import certificate_xml
from dl_push.certificate.math import rot2
from dl_push.certificate.material_contact import material_contact_velocity


def build_model(path, radius=.008, dt=.0005, shape='T', actuator_kv=2000):
    geometry = load_geometry('assets/native_v1', shape)
    root = ET.fromstring(certificate_xml(geometry.xml, free_body=False,
        physics_dt=dt, actuator_kv=actuator_kv, actuator_force_limit=30))
    opt = root.find('option')
    opt.set('cone', 'elliptic')
    opt.set('iterations', '100')
    opt.set('tolerance', '1e-10')
    # T's rectangular convex pieces can be represented exactly as boxes.
    meshes = {m.get('name'): np.fromstring(m.get('vertex'), sep=' ').reshape(-1, 3)
              for m in root.findall('./asset/mesh')}
    for geom in root.findall('./worldbody/body[@name="object"]/geom'):
        vertices = meshes[geom.attrib.pop('mesh')]
        lo, hi = vertices.min(0), vertices.max(0)
        assert len(np.unique(vertices, axis=0)) == 8
        assert np.all(np.isclose(vertices,lo)|np.isclose(vertices,hi)), 'non-box mesh'
        geom.set('type', 'box')
        geom.set('pos', ' '.join(map(str, (lo+hi)/2)))
        geom.set('size', ' '.join(map(str, (hi-lo)/2)))
    root.find('.//geom[@name="pusher_geom"]').set('size', str(radius))
    # Wider overview, with physical renderer providing the camera projection.
    root.find('.//camera[@name="top"]').set('fovy', '45')
    path.write_text(ET.tostring(root, encoding='unicode'))
    return mujoco.MjModel.from_xml_path(str(path)), certificate_polygon(geometry)


class Fixture:
    def __init__(self, model):
        self.m = model
        self.d = mujoco.MjData(model)
        self.obj = model.body('object').id
        self.ball = model.body('pusher').id
        self.pq = [int(model.jnt_qposadr[model.joint(n).id]) for n in ('pusher_x','pusher_y')]
        self.pv = [int(model.jnt_dofadr[model.joint(n).id]) for n in ('pusher_x','pusher_y')]
        self.oq = [int(model.jnt_qposadr[model.joint(n).id]) for n in ('obj_x','obj_y','obj_z','obj_yaw')]
        self.ov = [int(model.jnt_dofadr[model.joint(n).id]) for n in ('obj_x','obj_y','obj_z','obj_yaw')]
        self.ballgeom = model.geom('pusher_geom').id

    def state(self):
        yaw = self.d.qpos[self.oq[3]]
        R = rot2(yaw)
        p = self.d.xipos[self.obj,:2].copy()
        om = self.d.qvel[self.ov[3]]
        off = R @ self.m.body_ipos[self.obj,:2]
        v = self.d.qvel[self.ov[:2]] + om*np.array([-off[1], off[0]])
        ball = self.d.xpos[self.ball,:2].copy()
        return p, yaw, v, om, ball

    def step(self, cmd):
        if np.linalg.norm(cmd) > .25:
            raise ValueError('pusher speed limit')
        self.d.ctrl[:] = cmd
        mujoco.mj_step(self.m, self.d)
        # Synchronize poses, contacts and velocities at the same timestamp.
        mujoco.mj_forward(self.m, self.d)

    def reset(self, point, normal, radius, yaw):
        mujoco.mj_resetData(self.m,self.d)
        self.d.qpos[self.oq[3]] = yaw
        self.d.qpos[self.oq[:2]] = -rot2(yaw)@self.m.body_ipos[self.obj,:2]
        self.d.qpos[self.pq] = [-.21,-.21]
        # obj_z is a displacement, not the body's absolute initial height.
        self.d.qpos[self.oq[2]] = 0
        mujoco.mj_forward(self.m,self.d)
        for _ in range(round(.4/self.m.opt.timestep)):
            self.step(np.zeros(2))
        p, angle, *_ = self.state()
        self.d.qpos[self.pq] = p+rot2(angle)@(point+(radius+.0003)*normal)
        self.d.qvel[self.pv] = 0
        mujoco.mj_forward(self.m,self.d)

    def contacts(self, p, R, point, normal):
        points, normals, forces = [], [], []
        for j in range(self.d.ncon):
            c = self.d.contact[j]
            if self.ballgeom not in (c.geom1,c.geom2):
                continue
            other = c.geom2 if c.geom1 == self.ballgeom else c.geom1
            if self.m.geom_bodyid[other] != self.obj:
                continue
            f = np.zeros(6)
            mujoco.mj_contactForce(self.m,self.d,j,f)
            nb = R.T@c.frame.reshape(3,3)[0,:2]
            if nb@normal < 0:
                nb = -nb
            points.append(R.T@(c.pos[:2]-p))
            normals.append(nb)
            forces.append(float(f[0]))
        if not points:
            return [], [], [], np.inf, np.inf
        pe = max(np.linalg.norm(x-point) for x in points)
        ne = max(np.arccos(np.clip(x@normal,-1,1)) for x in normals)
        return points, normals, forces, pe, ne


def run(fixture, point, normal, radius, speed=.04, yaw=0., seconds=2., render=None, faces=None, distance=.035,initial_offset_body=None):
    f = fixture
    faces=np.asarray([normal] if faces is None else faces,dtype=float)
    offset=radius if len(faces)==1 else radius/float(faces[0]@normal)
    virtual_point=point+(offset-radius)*normal
    f.reset(virtual_point,normal,radius,yaw)
    if initial_offset_body is not None:
        f.d.qpos[f.pq]+=rot2(yaw)@np.asarray(initial_offset_body)
        mujoco.mj_forward(f.m,f.d)
    dt = f.m.opt.timestep
    anchor = point+offset*normal
    tangent = np.array([-normal[1],normal[0]])
    rows=[]
    start=None
    start_yaw=None
    for k in range(round(seconds/dt)):
        p, angle, v, om, ball = f.state()
        R = rot2(angle)
        local=R.T@(ball-p)
        material_v=R.T@v+om*np.array([-anchor[1],anchor[0]])
        # Absolute normal drive. Tangential transport has zero normal projection.
        u = speed*min(1., (k*dt)/.15)
        cmd=material_contact_velocity(p,angle,v,om,ball,virtual_point,normal,radius,u)
        cmd_b=R.T@cmd
        f.step(cmd)
        p, angle, v, om, ball=f.state()
        R=rot2(angle)
        local=R.T@(ball-p)
        pts, ns, fs, pe, ne=f.contacts(p,R,point,normal)
        if len(faces)>1:
            # Match each actual face to its own material contact, never to
            # the concave vertex or the mean of two distinct contacts.
            used=set();errors=[];angles=[]
            for face in faces:
                candidates=[j for j in range(len(ns)) if j not in used and fs[j]>.05]
                if not candidates:
                    errors.append(np.inf);angles.append(np.inf);continue
                j=max(candidates,key=lambda j:float(ns[j]@face));used.add(j)
                errors.append(np.linalg.norm(pts[j]-(anchor-radius*face)))
                angles.append(np.arccos(np.clip(ns[j]@face,-1,1)))
            pe=max(errors);ne=max(angles)
        if k*dt >= .35 and start is None:
            start=p.copy(); start_yaw=angle
        rows.append([k*dt, *p, angle, *(R.T@v), om, *(R.T@(ball-p)),
                     np.linalg.norm(local-anchor), pe, ne, sum(fs),len(fs),
                     *cmd_b, *(R.T@f.d.qvel[f.pv]), *v])
        if render and k%round(1/(30*dt))==0:
            render(f,rows[-1],point,normal)
        if start is not None and np.linalg.norm(p-start)>=distance:
            break
    a=np.asarray(rows)
    s=a[a[:,0]>=.35]
    if not len(s):
        s=a[-1:]
    velocity=s[:,4:7]
    mean=velocity.mean(0)
    std=velocity.std(0)
    slope=np.array([np.polyfit(s[:,0],velocity[:,i],1)[0] for i in range(3)]) if len(s)>1 else np.zeros(3)
    travel=float(np.linalg.norm(s[-1,1:3]-s[0,1:3]))
    return a, dict(point=point.tolist(),normal=normal.tolist(),speed=speed,yaw=yaw,
        twist_mean=mean.tolist(),twist_std=std.tolist(),twist_slope=slope.tolist(),
        distance=travel,delta_yaw_deg=float(np.rad2deg(s[-1,3]-s[0,3])),
        max_anchor_error_mm=float(s[:,9].max()*1000),max_contact_error_mm=float(s[:,10].max()*1000),
        max_normal_error_deg=float(np.rad2deg(s[:,11].max())),hold_rate=float(np.mean(s[:,12]>.05)),
        required_faces=len(faces),face_hold_rate=float(np.mean(np.isfinite(s[:,10])&(s[:,11]<np.deg2rad(.5)))),
        max_tangent_error_mm=float(np.max(np.abs((s[:,7:9]-anchor)@tangent))*1000),
        normal_speed_error=float(np.max(np.abs(s[:,16:18]@(-normal)-speed))),
        c=(mean/speed).tolist())


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    model,polygon=build_model(out/'model.xml');f=Fixture(model)
    lo,bot,hi,top=polygon.bounds
    modes=[([0,top],[0,1]),([0,bot],[0,-1]),([-.0255,0],[-1,0]),([.0255,0],[1,0]),
           ([-.055,bot],[0,-1]),([.055,bot],[0,-1]),([lo,(bot-.0174487)/2],[-1,0]),([hi,(bot-.0174487)/2],[1,0])]
    results=[]
    for i,(pt,n) in enumerate(modes):
        a,r=run(f,np.array(pt),np.array(n),.008)
        np.savez_compressed(out/f'mode_{i:02}.npz',samples=a)
        r['mode']=i;results.append(r);print(json.dumps(r),flush=True)
    (out/'report.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':main()
