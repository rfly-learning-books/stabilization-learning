"""Frozen-metric multi-shape transfer on an actuated SO101 simulation.

Geometry and normal-contact response calibration may change; actor weights do
not. Each process installs exactly one geometry, preserving the original suite.
"""
import argparse
import os
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch
from shapely.geometry import box, Point, LineString, shape as geo_shape
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from shapely.affinity import translate, scale, rotate
from . import so101_campaign as c
base=c.base
ROOT=c.PACKAGE
CHECKPOINT=ROOT/'media/so101_contact/contact_full_m.pt'
SOURCE=Path(__file__).read_text(encoding='utf-8')
SHA=hashlib.sha256(SOURCE.encode()).hexdigest()
# Nonoverlapping rectangular cells, in mm, scaled to a common 38-mm span.
PARTS={
 'T':[(-4,-19,4,11),(-14,11,14,19)],
 'RectWide':[(-19,-11,0,11),(0,-11,19,11)],
 'Square':[(-15,-15,0,15),(0,-15,15,15)],
 'L':[(-17,-18,-9,18),(-9,-18,17,-10)],
 'Plus':[(-4,-19,4,19),(-19,-4,-4,4),(4,-4,19,4)],
 'UShape':[(-19,-18,19,-10),(-19,-10,-11,18),(11,-10,19,18)],
 'OffsetComb3':[(-19,-21,-11,21),(-11,-21,19,-15),(-11,-3,13,3),(-11,15,17,21)],
 'AsymThreeBranch':[(-4,-21,4,21),(-19,-17,-4,-10),(4,-2,18,5),(-15,13,-4,19)],
}
LABELS={'T':'T','RectWide':'长方形','Square':'正方形','L':'L 形','Plus':'十字形','UShape':'U 形','OffsetComb3':'偏置三齿梳','AsymThreeBranch':'非对称三分支'}

def geometry(name):
    cells=np.array(PARTS[name],float)*.001
    polygon=unary_union([box(*b) for b in cells])
    assert polygon.geom_type=='Polygon' and polygon.is_valid
    assert abs(sum(box(*b).area for b in cells)-polygon.area)<1e-12
    factor=.038/max(polygon.bounds[2]-polygon.bounds[0],polygon.bounds[3]-polygon.bounds[1])
    centroid=np.array(polygon.centroid.coords[0]);cells=(cells-np.tile(centroid,2))*factor
    polygon=orient(unary_union([box(*b) for b in cells]),sign=1)
    return cells,polygon

def modes(poly):
    ring=np.asarray(poly.exterior.coords);points=[];normals=[]
    for a,b in zip(ring[:-1],ring[1:]):
        delta=b-a;normal=np.array([delta[1],-delta[0]])/np.linalg.norm(delta)
        for t in [.18,.5,.82]:
            p=a+t*delta;near=p+normal*.0041;far=p+normal*.0058
            if poly.distance(Point(near))<.00405 or poly.distance(Point(far))<.0055:continue
            if LineString([far,far+normal*.003]).intersects(poly.buffer(.0049)):continue
            points.append(p);normals.append(normal)
    if len(points)<12:raise RuntimeError('Fewer than twelve accessible contacts')
    points=np.asarray(points);normals=np.asarray(normals)
    features=np.c_[points/.019,normals]
    selected=[int(np.argmin(points[:,0]))]
    while len(selected)<12:
        dist=np.min(((features[:,None]-features[selected][None])**2).sum(-1),axis=1)
        dist[selected]=-1;selected.append(int(np.argmax(dist)))
    return points[selected],normals[selected]

def novelty(poly,name):
    ck=torch.load(base.VENDOR/'pretrained_full_m.pt',map_location='cpu',weights_only=True)
    train_names=sorted(ck['fingerprints']);best=(0.,None)
    def norm(p):
        p=translate(p,-p.centroid.x,-p.centroid.y);return scale(p,xfact=1/np.sqrt(p.area),yfact=1/np.sqrt(p.area),origin=(0,0))
    a=norm(poly);refs={}
    for n in train_names:
        file=Path(os.environ.get('STABILIZATION_REFERENCE_POLYGONS', 'external_assets/reference_polygons'))/n/'polygon.json'
        if not file.exists():continue
        raw=json.loads(file.read_text(encoding='utf-8-sig'));refs[n]=raw;b=norm(geo_shape(raw))
        # Compare after centroid/area normalization, rotations and reflections.
        xy=np.asarray(b.exterior.coords);edges=np.diff(xy,axis=0)
        angles=np.unique(np.round(np.arctan2(edges[:,1],edges[:,0]),7))
        for reflection in [1,-1]:
            reflected=scale(b,xfact=reflection,yfact=1,origin=(0,0))
            for angle in angles:
                candidate=rotate(reflected,-angle if reflection==1 else angle,origin=(0,0),use_radians=True)
                iou=a.intersection(candidate).area/a.union(candidate).area
                if iou>best[0]:best=(iou,n)
    return dict(pretraining_names=train_names,known_pretraining_family=name in train_names,closest_training_polygon=best[1],max_similarity_iou=best[0],distinct_geometry_under_comparison=best[0]<.999,scope='Exact geometric comparison is not proof of a new semantic shape family.'),refs

@contextmanager
def install(name):
    cells,poly=geometry(name);points,normals=modes(poly)
    saved=(base.POLY,base.POINTS,base.NORMALS,c.dynamic_xml,c.Plant,c.transfer)
    original_xml=c.dynamic_xml;original_plant=c.Plant
    base.POLY=poly;base.POINTS=points;base.NORMALS=normals
    def xml(cfg):
        text,mesh=original_xml(cfg);root=ET.fromstring(text)
        for bodyname in ['block','goal']:
            body=root.find(f".//body[@name='{bodyname}']")
            for g in list(body.findall('geom')):body.remove(g)
            for i,(x0,y0,x1,y1) in enumerate(cells):
                attrs=dict(type='box',pos=f'{(x0+x1)/2} {(y0+y1)/2} 0',size=f'{(x1-x0)/2} {(y1-y0)/2} '+('.004' if bodyname=='block' else '.0003'))
                if bodyname=='block':
                    attrs.update(name=['block_stem','block_top'][i] if i<2 else f'block_part_{i}',mass=str(.025*cfg.mass*(x1-x0)*(y1-y0)/poly.area),contype='8',conaffinity='7',friction=f'{cfg.friction} .001 .0001',condim='4',priority='1',rgba='.36 .43 .55 1')
                else:attrs.update(name=f'goal_part_{i}',material='goal_hatch',contype='0',conaffinity='0',group='1')
                ET.SubElement(body,'geom',**attrs)
        return ET.tostring(root,encoding='unicode'),mesh
    class ShapePlant(original_plant):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.block_geoms=set(np.flatnonzero(self.model.geom_bodyid==self.env._block_body).tolist())
        def frame(self):
            from PIL import Image,ImageDraw,ImageFont
            a=self.env.render(camera='3d');b=self.env.render(camera='2d');im=Image.new('RGB',(1024,608),'#131b27')
            im.paste(Image.fromarray(a),(0,96));im.paste(Image.fromarray(b),(512,96));d=ImageDraw.Draw(im)
            f=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17);q=self.pose()
            lines=[f'SO101 ACTUATED | {name} | {self.cfg.metric} | {self.phase} | mode {self.mode+1}',f't={self.data.time:.2f}s | position={np.linalg.norm(q[:2])*1000:.2f}mm | yaw={np.rad2deg(q[2]):.2f}deg | overlap={base.coverage(q):.1%}',f'Frozen weights + shape response calibration | transit contacts={self.transit_contacts} | arm contacts={self.arm_contacts}']
            for i,line in enumerate(lines):d.text((12,8+28*i),line,font=f,fill='white')
            self.writer.append_data(np.asarray(im))
    def transfer(s,m,previous,policy):
        s.settle();q0=s.pose();t=s.data.time;v0=float(policy.value(q0)[0])
        if previous is not None:
            outward=base.rot(q0[2])@normals[previous];start=s.pusher();obstacle=base.shape(q0)
            # Narrow concavities may not accommodate a fixed 4-mm retreat.
            retreat=None
            for dist in [.004,.003,.002,.0015]:
                end=start+outward*dist
                if obstacle.distance(Point(end))>.00525 and not LineString([start+outward*.0006,end]).intersects(obstacle.buffer(.004)):
                    retreat=end;break
            if retreat is None:raise RuntimeError('no_safe_retreat')
            s.move(retreat,'RETRACT',.010)
        q=s.observe();end=q[:2]+base.rot(q[2])@(points[m]+normals[m]*.0058)
        for waypoint in base.path(s.pusher(),end,q)[1:]:s.move(waypoint,'TRANSIT')
        drift=s.pose()-q0
        s.transfers.append(dict(time_s=float(s.data.time-t),drift_mm=float(np.linalg.norm(drift[:2])*1000),drift_deg=float(abs(np.rad2deg(drift[2]))),V_before=v0,V_after=float(policy.value(s.pose())[0]),mode_from=previous,mode_to=m))
        if np.linalg.norm(drift[:2])>.0002:raise RuntimeError('transfer_drift_over_0.2mm')
        s.mode=m
    c.dynamic_xml=xml;c.Plant=ShapePlant;c.transfer=transfer
    try:yield cells,poly,points,normals
    finally:base.POLY,base.POINTS,base.NORMALS,c.dynamic_xml,c.Plant,c.transfer=saved

def calibrate(out,points,normals,repeat=6,seed=48031):
    cfg=c.Config(plant='arm',speed=.012,max_time=20)
    rng=np.random.default_rng(seed);records=[];responses=[];valid=[]
    for mode in range(12):
        trials=[]
        for j in range(repeat):
            s=c.Plant(cfg,seed+j);q=np.r_[rng.uniform(-.003,.003,2),rng.uniform(-.2,.2)]
            p=q[:2]+base.rot(q[2])@(points[mode]+normals[mode]*.0058)
            row=dict(mode=mode,trial=j,requested=q.tolist())
            try:
                s.reset(q,p);s.settle();before=s.pose();s.mode=mode;c.push(s,mode,.002);after=s.pose();delta=after-before
                response=np.r_[base.rot(-before[2])@delta[:2],delta[2]]/.002
                row.update(success=True,before=before.tolist(),after=after.tolist(),response=response.tolist());trials.append(response)
            except RuntimeError as ex:row.update(success=False,reason=str(ex))
            finally:s.close()
            records.append(row)
        isvalid=len(trials)>=max(2,repeat//2)
        responses.append(np.median(trials,axis=0) if isvalid else np.zeros(3));valid.append(isvalid)
        print('calibrate',out.name,mode,len(trials),'/',repeat,flush=True)
    data=dict(C=np.asarray(responses).tolist(),valid=valid,records=records,source_sha256=SHA)
    c.save(out/'calibration.json',data)
    return data

def policy(metric,data):
    p=c.Policy(metric);p.C=np.array(data['C']);p.Ct=torch.tensor(p.C[None],dtype=torch.float32);p.Ct[:,:,2]*=2
    p.valid=torch.tensor([data['valid']],dtype=torch.bool)
    if not p.valid.any():raise RuntimeError('No valid calibrated modes')
    if metric!='identity':
        for parameter in p.actor.parameters():parameter.requires_grad_(False)
        with torch.no_grad():p.cond=p.actor.encoder(p.Ct,p.valid).detach()
    return p

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shape',choices=list(PARTS),required=True);ap.add_argument('--output',type=Path,default=ROOT/'media/so101_multishape');ap.add_argument('--n',type=int,default=20);ap.add_argument('--seed',type=int,default=20261001);ap.add_argument('--calibration-repeats',type=int,default=6);ap.add_argument('--video',action='store_true');args=ap.parse_args()
    out=args.output/args.shape;out.mkdir(parents=True,exist_ok=True)
    initial_sha=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    with install(args.shape) as (cells,poly,points,normals):
        cfg=c.Config(plant='arm',speed=.012)
        c.save(out/'shape.json',dict(name=args.shape,label=LABELS[args.shape],cells=cells.tolist(),polygon=list(poly.exterior.coords),points=points.tolist(),normals=normals.tolist(),height_m=.008,mass_kg=.025,span_m=.038))
        (out/'scene.xml').write_text(c.dynamic_xml(cfg)[0], encoding='utf-8');(out/'source_snapshot.py').write_text(SOURCE, encoding='utf-8')
        audit,refs=novelty(poly,args.shape);c.save(out/'training_geometry_audit.json',audit);c.save(out/'training_reference_polygons.json',refs)
        if (out/'calibration.json').exists():
            data=json.loads((out/'calibration.json').read_text(encoding='utf-8-sig'))
            if data['source_sha256']!=SHA:raise ValueError('Calibration source mismatch')
        else:data=calibrate(out,points,normals,args.calibration_repeats)
        initials=c.cases(args.seed,args.n,extent=.012,angle=.35)
        for metric in ['identity','finetuned']:
            cfg=c.Config(plant='arm',speed=.012,metric=metric);p=policy(metric,data);rows=[];file=out/f'{metric}.json'
            if file.exists():
                prior=json.loads(file.read_text(encoding='utf-8-sig'))
                if prior['source_sha256']!=SHA or prior['initials']!=[q.tolist() for q in initials]:raise ValueError('Evaluation source/initial bank mismatch')
                rows=prior['episodes']
            for i,q in enumerate(initials[len(rows):],start=len(rows)):
                r=c.rollout(cfg,q,args.seed+i,policy=p);r['episode']=i;rows.append(r)
                c.save(file,dict(shape=args.shape,config=c.asdict(cfg),source_sha256=SHA,core_sha256=c.SOURCE_SHA,checkpoint_sha256=initial_sha,initials=[q.tolist() for q in initials],episodes=rows))
                print('evaluate',args.shape,metric,i,r['success'],r['reason'],round(r['sim_time_s'],1),flush=True)
            if args.video and metric=='finetuned':
                chosen=next((r for r in rows if r['success']),rows[0]);r=c.rollout(cfg,np.array(chosen['initial']),chosen['seed'],policy=p,video=out/'demo.mp4')
                assert r['success']==chosen['success'] and np.allclose(r['final'],chosen['final'],atol=1e-10), 'Render replay mismatch'
        assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()==initial_sha
        c.save(out/'provenance.json',dict(source_sha256=SHA,core_sha256=c.SOURCE_SHA,checkpoint_sha256=initial_sha,weights_unchanged=True,seed=args.seed,calibration_seed=48031,calibration_repeats=args.calibration_repeats,scope='Actuated SO101-style simulation; no physical hardware connected. Frozen T-adapted actor; new shapes only recalibrate response maps.'))

if __name__=='__main__':main()
