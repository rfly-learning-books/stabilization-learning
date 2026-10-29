"""Geometry -> measured candidates -> N-column response library.

No learning or Q search is performed. Missing authority is reported explicitly.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib,json,time
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import orient
from shapely.affinity import translate
from .fixture import Fixture,build_model,run
from .polygon_fixture import polygon_model
from .shape_catalog import NATIVE,split,catalog
from .assets import certificate_polygon,load_geometry
from .library import ModeLibrary
from .evaluation import json_safe


@dataclass(frozen=True)
class GeneratorConfig:
    radius:float=.018
    dt:float=.0005
    kv:float=500.
    speeds:tuple=(.03,.05)
    validation_speed:float=.04
    validation_yaw:float=.7
    candidate_budget:int=40
    distance:float=.035
    near_translation_ratio:float=.15  # |omega| / ||v|| in rad/m
    max_contact_error_mm:float=.5
    min_hold_rate:float=.995
    max_response_error:float=.15
    minimum_face_length:float=.036
    endpoint_margin:float=.008
    perturbation_m:float=.001
    allow_smooth_arcs:bool=False
    smooth_turn_max_deg:float=6.0
    smooth_radius_min_m:float=.025

    def __post_init__(self):
        if self.radius<=0 or self.dt<=0 or self.kv<=0 or self.distance<.03:
            raise ValueError('positive fixture scales and at least 30 mm distance required')
        if len(set(self.speeds))<2 or any(u<=0 or u>.25 for u in self.speeds):
            raise ValueError('at least two distinct positive fitting speeds up to .25 m/s required')
        if not 0<self.validation_speed<=.25 or not np.isfinite(self.validation_yaw):
            raise ValueError('invalid validation speed/yaw')
        if not 0<self.min_hold_rate<=1 or self.max_contact_error_mm<=0 or self.max_response_error<=0:
            raise ValueError('invalid calibration thresholds')
        if self.minimum_face_length<=0 or self.endpoint_margin<=0:raise ValueError('positive face margins required')


@dataclass
class ContactCandidate:
    point:np.ndarray
    normal:np.ndarray
    faces:np.ndarray
    role:str='sample'

    def record(self):
        return dict(point=self.point.tolist(),normal=self.normal.tolist(),faces=self.faces.tolist(),role=self.role)


def candidate(point,normal,role='sample',faces=None):
    n=np.asarray(normal,dtype=float);n/=np.linalg.norm(n)
    return ContactCandidate(np.asarray(point,dtype=float),n,np.asarray([n] if faces is None else faces,dtype=float),role)


def preferred(p,name):
    """Preserve reviewed source modes; other polygons use boundary sampling."""
    l,b,r,t=p.bounds
    if name in ('Square','RectWide','Plus','T','H'):
        if name=='T':
            base=[candidate([0,t],[0,1]),candidate([0,b],[0,-1]),candidate([-.0255,.0017196215],[-1,0]),candidate([.0255,.0017196215],[1,0])]
            extras=[candidate([-.055,b],[0,-1]),candidate([-.0255,-.01744871795],[-1,1],'corner',[[-1,0],[0,1]]),candidate([.0255,-.01744871795],[1,1],'corner',[[1,0],[0,1]]),candidate([r,-.04294871795],[1,0])]
        elif name=='H':
            base=[candidate([0,.0255],[0,1]),candidate([0,-.0255],[0,-1]),candidate([l,0],[-1,0]),candidate([r,0],[1,0])]
            extras=[candidate([l,.04],[-1,0]),candidate([r,.04],[1,0]),candidate([-.03,.0255],[1,1],'corner',[[1,0],[0,1]]),candidate([.03,.0255],[-1,1],'corner',[[-1,0],[0,1]])]
        else:
            base=[candidate([0,t],[0,1]),candidate([0,b],[0,-1]),candidate([l,0],[-1,0]),candidate([r,0],[1,0])]
            if name=='Plus':
                extras=[candidate([x,y],[np.sign(x),0],'rotation') for y in (-.065,.065) for x in (-.0255,.0255)]
                extras += [candidate([x,y],[0,np.sign(y)],'rotation') for x in (-.065,.065) for y in (-.0255,.0255)]
            else:
                extras=[candidate([x,b],[0,-1],'rotation') for x in (l*.5,r*.5)]+[candidate([x,t],[0,1],'rotation') for x in (l*.5,r*.5)]+[candidate([x,y],[np.sign(x),0],'rotation') for x in (l,r) for y in (b*.5,t*.5)]
        for i,c in enumerate(base):c.role=['translation_-Y','translation_+Y','translation_+X','translation_-X'][i]
        for c in extras:
            if c.role=='sample':c.role='reviewed'
        return base+extras
    return []


def geometry_safety(p,c,cfg):
    """Hard exclusion of vertex pushes and short/endpoint face contacts."""
    edges=[]
    for ring in [p.exterior,*p.interiors]:
        a=np.asarray(ring.coords)
        for start,end in zip(a[:-1],a[1:]):
            e=end-start;length=np.linalg.norm(e)
            if length>1e-10:edges.append((start,e/length,length,np.array([e[1],-e[0]])/length))
    if len(c.faces)==1:
        contacts=[(c.point,c.normal)]
    else:
        # Each actual contact must lie safely inside its support face. The
        # concave vertex itself is not either of these two physical contacts.
        anchor=c.point+cfg.radius/(c.faces[0]@c.normal)*c.normal
        contacts=[(anchor-cfg.radius*face,face) for face in c.faces]
    support=[]
    for point,normal in contacts:
        matches=[]
        for start,tangent,length,outward in edges:
            s=float((point-start)@tangent)
            if abs(float((point-start)@outward))<1e-7 and normal@outward>.999 and -1e-7<=s<=length+1e-7:
                matches.append((length,min(s,length-s)))
        if not matches:return dict(accepted=False,reason='not_a_supported_face_contact')
        length,margin=max(matches)
        if length<cfg.minimum_face_length or margin<cfg.endpoint_margin-1e-7:
            arc=smooth_arc_support(p,point,normal,cfg) if cfg.allow_smooth_arcs and len(c.faces)==1 else None
            if arc is not None:
                support.append(arc)
                continue
            if length<cfg.minimum_face_length:return dict(accepted=False,reason='short_face',face_length=length)
            return dict(accepted=False,reason='near_endpoint_or_convex_vertex',endpoint_distance=margin)
        support.append(dict(face_length=length,endpoint_distance=margin))
    return dict(accepted=True,reason='safe_faces',supports=support)


def smooth_arc_support(p,point,normal,cfg):
    """Accept facet interiors only on long, gently turning boundary chains.

    No polygon vertex is a candidate. Short edges are exempted only when both
    neighboring directions follow a resolved smooth arc, with radius >=25mm
    and >=18mm boundary travel available on each side before a sharp junction.
    """
    if not cfg.allow_smooth_arcs:return None
    for ring in [p.exterior,*p.interiors]:
        a=np.asarray(ring.coords)[:-1];e=np.roll(a,-1,axis=0)-a
        lengths=np.linalg.norm(e,axis=1);t=e/lengths[:,None]
        for j in range(len(a)):
            outward=np.array([t[j,1],-t[j,0]])
            s=float((point-a[j])@t[j])
            if abs(float((point-a[j])@outward))>1e-7 or normal@outward<.999:continue
            # Keep actual contact away from numerical polygon vertices too.
            if min(s,lengths[j]-s)<.2*lengths[j]:continue
            distances=[];radii=[]
            for direction,initial in [(-1,s),(1,lengths[j]-s)]:
                travel=initial;k=j
                for _ in range(len(a)-1):
                    nxt=(k+direction)%len(a)
                    angle=float(np.arccos(np.clip(t[k]@t[nxt],-1,1)))
                    radius=(lengths[k]+lengths[nxt])/(2*max(angle,1e-12))
                    if angle>np.deg2rad(cfg.smooth_turn_max_deg) or radius<cfg.smooth_radius_min_m:break
                    radii.append(radius);travel+=lengths[nxt];k=nxt
                    if travel>=max(cfg.minimum_face_length/2,cfg.endpoint_margin):break
                distances.append(travel)
            if min(distances)<max(cfg.minimum_face_length/2,cfg.endpoint_margin):continue
            return dict(kind='smooth_arc_facets',face_length=float(lengths[j]),
                endpoint_distance=float(min(s,lengths[j]-s)),smooth_chain_margin_m=float(min(distances)),
                minimum_local_radius_m=float(min(radii)) if radii else 1e6,
                sharp_vertex_contact=False,representation='polygonal approximation')
    return None


def sample_candidates(polygon,n=12,config=None,name='custom',audit=None):
    cfg=config or GeneratorConfig()
    if n<6:raise ValueError('N must be at least 6 for four translation and two rotation slots')
    if polygon.geom_type!='Polygon' or not polygon.is_valid or polygon.area<=0:raise ValueError('invalid polygon')
    p=orient(translate(polygon,xoff=-polygon.centroid.x,yoff=-polygon.centroid.y).simplify(1e-10,preserve_topology=True),1.)
    # Name alone must not silently substitute reviewed geometry for custom input.
    seeds=[]
    if name in ('T','Square','RectWide','Plus','H'):
        reference=catalog()[name]
        if p.symmetric_difference(reference).area<1e-10:seeds=preferred(p,name)
    rest=[]
    for ring in [p.exterior,*p.interiors]:
        a=np.array(ring.coords)
        for start,end in zip(a[:-1],a[1:]):
            edge=end-start;length=np.linalg.norm(edge)
            if length<cfg.minimum_face_length:
                if cfg.allow_smooth_arcs:
                    normal=np.array([edge[1],-edge[0]])/length
                    point=(start+end)/2
                    if smooth_arc_support(p,point,normal,cfg) is not None:
                        rest.append(candidate(point,normal,'smooth_arc'))
                if audit is not None:audit.append(dict(reason='short_face',start=start.tolist(),end=end.tolist(),length=float(length)))
                continue
            tangent=edge/length;normal=np.array([tangent[1],-tangent[0]])
            margin=max(.06,cfg.endpoint_margin/length)
            fractions=[np.clip(-start@edge/(length*length),margin,1-margin),.5,.15,.3,.7,.85]
            if n>24:fractions+=list(np.linspace(.06,.94,max(8,n//2)))
            for f in fractions:rest.append(candidate(start+np.clip(f,margin,1-margin)*edge,normal))
        for j,point in enumerate(a[:-1]):
            incoming=point-a[:-1][j-1];outgoing=a[(j+1)%(len(a)-1)]-point
            turn=incoming[0]*outgoing[1]-incoming[1]*outgoing[0]
            if turn< -1e-10:
                before=np.array([incoming[1],-incoming[0]])/np.linalg.norm(incoming)
                after=np.array([outgoing[1],-outgoing[0]])/np.linalg.norm(outgoing)
                if np.linalg.norm(before+after)>1e-8:rest.append(candidate(point,before+after,'concave_pair',[before,after]))
    # Near-COM normal lines first; far-offset candidates alternate torque signs.
    rest.sort(key=lambda c:abs(c.point[0]*c.normal[1]-c.point[1]*c.normal[0]))
    valid=[]
    for c in seeds+rest:
        quality=geometry_safety(p,c,cfg)
        if not quality['accepted']:
            if audit is not None:audit.append(dict(geometry=c.record(),**quality))
            continue
        offset=cfg.radius/(c.faces[0]@c.normal)
        centre=c.point+offset*c.normal
        if Point(centre).distance(p)<cfg.radius-1e-6:
            if audit is not None:audit.append(dict(geometry=c.record(),reason='ball_clearance'))
            continue
        if any(np.linalg.norm(c.point-v.point)<.0003 and c.normal@v.normal>.999 for v in valid):continue
        valid.append(c)
    budget=max(cfg.candidate_budget,3*n)
    if len(valid)>budget:
        front=valid[:max(len(seeds),budget//2)]
        remainder=valid[len(front):]
        front += [remainder[i] for i in np.linspace(0,len(remainder)-1,budget-len(front),dtype=int)]
        valid=front
    return p,valid


def measured_ok(r,cfg,distance):
    return (r.get('distance',0)>=distance and r.get('face_hold_rate',0)>=cfg.min_hold_rate
       and r.get('max_contact_error_mm',float('inf'))<=cfg.max_contact_error_mm)


def coverage(C,valid,cfg):
    v=C[:2].T;w=C[2];speed=np.linalg.norm(v,axis=1)
    pure=valid&(speed>1e-5)&(np.abs(w)/np.maximum(speed,1e-12)<=cfg.near_translation_ratio)
    directions={'-Y':np.array([0,-1]),'+Y':np.array([0,1]),'+X':np.array([1,0]),'-X':np.array([-1,0])}
    covered={key:np.flatnonzero(pure&(v@d/np.maximum(speed,1e-12)>.707)).tolist() for key,d in directions.items()}
    # Positive spanning requires the largest angular gap strictly below pi.
    angles=np.sort(np.arctan2(v[pure,1],v[pure,0]));gap=float('inf')
    if len(angles)>1:gap=float(np.max(np.diff(np.r_[angles,angles[0]+2*np.pi])))
    return dict(translation_modes=np.flatnonzero(pure).tolist(),directions=covered,
      missing_directions=[k for k,x in covered.items() if not x],positive_spans_xy=bool(gap<np.pi-1e-5),
      four_direction_translation=bool(len(angles)>=4 and all(covered.values()) and gap<np.pi-1e-5),
      rotation_positive=np.flatnonzero(valid&(w>.05)).tolist(),rotation_negative=np.flatnonzero(valid&(w<-.05)).tolist())


def generate_modes(polygon,n,output,name='custom',config=None,render_callback=None):
    """Build a measured 3xN library plus audit. Cache key includes all inputs.

    Returns ModeLibrary, report. Incomplete libraries retain invalid masks and
    explicit missing-authority status. C is column concatenation, not a sum.
    """
    cfg=config or GeneratorConfig();out=Path(output).resolve()
    geometry_rejections=[]
    p,candidates=sample_candidates(polygon,n,cfg,name,audit=geometry_rejections)
    if max(abs(x) for x in p.bounds)+cfg.radius+cfg.distance+.015>=.232:
        raise ValueError('polygon too large for the +/-0.232 m fixture; scale the metre-valued input')
    signature=hashlib.sha256((p.wkb_hex+json.dumps(asdict(cfg),sort_keys=True)+str(n)+name+
       Path(__file__).read_text()+Path(__file__).with_name('fixture.py').read_text()+
       Path(__file__).with_name('material_contact.py').read_text()+Path(__file__).with_name('polygon_fixture.py').read_text()+
       Path(__file__).with_name('model.py').read_text()+Path(__file__).with_name('shape_catalog.py').read_text()).encode()).hexdigest()
    if (out/'report.json').exists():
        old=json.loads((out/'report.json').read_text())
        if old['signature']!=signature:raise FileExistsError('output contains a different protocol; use a new directory')
        return ModeLibrary.load(out/'library'),old
    out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    native=name in NATIVE[:5] and p.symmetric_difference(certificate_polygon(load_geometry('assets/native_v1',name))).area<1e-10
    if native:
        model,_=build_model(out/'model.xml',radius=cfg.radius,dt=cfg.dt,shape=name,actuator_kv=cfg.kv)
    else:model,p=polygon_model(p,out/'model.xml',cfg.radius,cfg.dt,cfg.kv)
    fixture=Fixture(model);screen=[]
    def experiment(c,u,yaw=0,short=False,callback=None,offset=None):
        try:return run(fixture,c.point,c.normal,cfg.radius,speed=u,yaw=yaw,seconds=.9 if short else 4.,distance=.006 if short else cfg.distance,faces=c.faces,render=callback,initial_offset_body=offset)
        except (ValueError,RuntimeError,FloatingPointError) as e:return np.empty((0,20)),dict(failure=str(e),distance=0,twist_mean=[0,0,0],face_hold_rate=0)
    for index,c in enumerate(candidates):
        a,r=experiment(c,.04,short=True)
        screen.append(dict(candidate_id=index,geometry=c.record(),geometry_safety=geometry_safety(p,c,cfg),measurement=r,valid=measured_ok(r,cfg,.004)))
        np.savez_compressed(out/f'candidate_{index:03}.npz',samples=a)
    # Refine zero-angular-response points within the SAME straight face using
    # measured bracketing. Never move a point across a notch or into material.
    groups={}
    for i,c in enumerate(candidates):
        if len(c.faces)!=1 or not screen[i]['valid']:continue
        key=(*np.round(c.normal,6),round(float(c.normal@c.point),6))
        groups.setdefault(key,[]).append(i)
    for ids in groups.values():
        def ratio(i):
            v=np.array(screen[i]['measurement']['twist_mean'])
            return abs(v[2])/max(np.linalg.norm(v[:2]),1e-12)
        if min(map(ratio,ids))<cfg.near_translation_ratio*.4:continue
        best=min(ids,key=ratio);normal=candidates[best].normal
        tangent=np.array([-normal[1],normal[0]])
        ids.sort(key=lambda i:float(candidates[i].point@tangent))
        pairs=[(a,b) for a,b in zip(ids[:-1],ids[1:]) if screen[a]['measurement']['twist_mean'][2]*screen[b]['measurement']['twist_mean'][2]<0]
        if not pairs:continue
        left,right=min(pairs,key=lambda pair:np.linalg.norm(candidates[pair[0]].point-candidates[pair[1]].point))
        for iteration in range(4):
            wl=screen[left]['measurement']['twist_mean'][2];wr=screen[right]['measurement']['twist_mean'][2]
            alpha=float(np.clip(-wl/(wr-wl),.1,.9))
            point=(1-alpha)*candidates[left].point+alpha*candidates[right].point
            if Point(point).distance(p.boundary)>1e-7 or Point(point+cfg.radius*normal).distance(p)<cfg.radius-1e-6:break
            c=candidate(point,normal,'refined_translation');i=len(candidates)
            if not geometry_safety(p,c,cfg)['accepted']:break
            a,r=experiment(c,.04,short=True);candidates.append(c)
            screen.append(dict(candidate_id=i,geometry=c.record(),measurement=r,valid=measured_ok(r,cfg,.004),refinement_parents=[left,right]))
            np.savez_compressed(out/f'candidate_{i:03}.npz',samples=a)
            if not screen[i]['valid'] or ratio(i)<cfg.near_translation_ratio*.4:break
            if wl*r['twist_mean'][2]<0:right=i
            else:left=i
    # Reserve measured four-direction translations, then both rotation signs.
    selected=[]
    for d in ([0,-1],[0,1],[1,0],[-1,0]):
        choices=[]
        for i,s in enumerate(screen):
            v=np.array(s['measurement']['twist_mean']);mag=np.linalg.norm(v[:2])
            if s['valid'] and i not in selected and mag>1e-6 and v[:2]@d/mag>.707:
                ratio=abs(v[2])/mag
                preferred_seed=candidates[i].role.startswith('translation') and ratio<=cfg.near_translation_ratio
                choices.append((not preferred_seed,ratio,i))
        if choices:selected.append(min(choices)[2])
    # Preserve reviewed source slots after the four translation selections.
    for i,c in enumerate(candidates):
        if c.role not in ('sample','smooth_arc','refined_translation','vertex','refined_vertex') and i not in selected and screen[i]['valid'] and len(selected)<n:selected.append(i)
    for sign in (1,-1):
        choices=[i for i,s in enumerate(screen) if s['valid'] and i not in selected and sign*s['measurement']['twist_mean'][2]>.002]
        if choices and len(selected)<n:selected.append(max(choices,key=lambda i:sign*screen[i]['measurement']['twist_mean'][2]))
    while len(selected)<min(n,len(candidates)):
        choices=[i for i in range(len(candidates)) if i not in selected]
        def score(i):
            c=candidates[i]
            separation=min((np.linalg.norm(c.point-candidates[j].point)+.03*(1-c.normal@candidates[j].normal) for j in selected),default=1)
            return (screen[i]['valid'],separation)
        selected.append(max(choices,key=score))
    if len(selected)<n:
        raise ValueError(f'only {len(selected)} distinct reachable candidates for N={n}; reduce N or radius')
    records=[];cols=[];valid=[];C_history=[];history_times=[];rejected_modes=[]
    for mode,i in enumerate(selected):
        attempted=set()
        while True:
            c=candidates[i];trials=[]
            conditions=[(u,0.,'fit') for u in cfg.speeds]+[(cfg.validation_speed,cfg.validation_yaw,'validation')]
            for trial,(u,yaw,kind) in enumerate(conditions):
                a,r=experiment(c,u,yaw)
                np.savez_compressed(out/f'mode_{mode:02}_trial_{trial}.npz',samples=a)
                r.update(kind=kind);trials.append(r)
            for sign in (-1,1):
                offset=sign*cfg.perturbation_m*np.array([-c.normal[1],c.normal[0]])
                a,r=experiment(c,cfg.validation_speed,cfg.validation_yaw,offset=offset)
                np.savez_compressed(out/f'mode_{mode:02}_perturb_{sign:+d}.npz',samples=a)
                r.update(kind='perturbation',initial_offset_body=offset.tolist());trials.append(r)
            good=all(measured_ok(r,cfg,.03) for r in trials)
            inputs=np.array(cfg.speeds);values=np.array([r['twist_mean'] for r in trials if r['kind']=='fit'])
            ci=(inputs[:,None]*values).sum(0)/(inputs@inputs)
            pred=ci*cfg.validation_speed;weight=np.array([1,1,.1]);errors=[]
            for trial in trials[len(cfg.speeds):]:
                actual=np.array(trial['twist_mean'])
                errors.append(float(np.linalg.norm((actual-pred)*weight)/max(np.linalg.norm(actual*weight),1e-12)))
            err=max(errors)
            good=good and err<=cfg.max_response_error
            attempted.add(i)
            if good:break
            rejected_modes.append(dict(mode=mode,candidate_id=i,geometry=c.record(),trials=trials,validation_error=err))
            alternatives=[j for j in range(len(candidates)) if j not in selected and j not in attempted and screen[j]['valid']]
            if not alternatives or len(attempted)>=8:break
            # Preserve every failed physical record before trying a replacement.
            import shutil
            archive=out/'rejected'/f'mode_{mode:02}_candidate_{i:03}'
            archive.mkdir(parents=True,exist_ok=True)
            for path in out.glob(f'mode_{mode:02}_*.npz'):shutil.copy2(path,archive/path.name)
            original_normal=c.normal
            def replacement_score(j):
                v=np.array(screen[j]['measurement']['twist_mean'])
                if mode<4:
                    return (candidates[j].normal@original_normal, -abs(v[2])/max(np.linalg.norm(v[:2]),1e-12))
                return (len(candidates[j].faces)==1, abs(v[2]))
            i=max(alternatives,key=replacement_score);selected[mode]=i
        cols.append(ci);valid.append(good)
        snapshot=np.full((3,n),np.nan);snapshot[:,:len(cols)]=np.array(cols).T
        C_history.append(snapshot);history_times.append(time.perf_counter()-start)
        records.append(dict(mode=mode,candidate_id=i,geometry=c.record(),geometry_safety=geometry_safety(p,c,cfg),c=ci.tolist(),valid=good,validation_error=err,trials=trials))
        # Callback runs a separate clearly labelled u=.04/yaw=0 demonstration.
        if render_callback is not None:
            cb=lambda fx,row,point,normal:render_callback(mode,fx,row,c,p)
            a,r=experiment(c,.04,callback=cb)
            np.savez_compressed(out/f'mode_{mode:02}_demo.npz',samples=a)
            records[-1]['demo']=r
    C=np.array(cols).T;mask=np.array(valid);authority=coverage(C,mask,cfg)
    cs=[candidates[i] for i in selected];faces=np.zeros((n,2,2));fm=np.zeros((n,2),bool)
    for i,c in enumerate(cs):faces[i,:len(c.faces)]=c.faces;fm[i,:len(c.faces)]=True
    meta=dict(signature=signature,protocol='safe_face_material_modes_v2',shape_split=split(name),
       physics='planar MuJoCo, normal-only sphere contact; not free-body validation',config=asdict(cfg),
       input='absolute normal pusher command u in m/s; not full XY pusher speed',
       response='[body COM vx, body COM vy, omega_z] = c_i*u; C=[c_0 ... c_(N-1)]',
       controller='dl_push.certificate.material_contact.material_contact_velocity',
       authority=authority,model_sha256=hashlib.sha256((out/'model.xml').read_bytes()).hexdigest())
    library=ModeLibrary(name,C,np.array([c.point for c in cs]),np.array([c.normal for c in cs]),np.array([[-c.normal[1],c.normal[0]] for c in cs]),
       np.full(n,min(cfg.speeds)),np.full(n,max(cfg.speeds)),mask,face_normals=faces,face_mask=fm,
       contact_offset=np.array([cfg.radius/(c.faces[0]@c.normal) for c in cs]),
       fit_error=np.array([r['validation_error'] for r in records]),hold_rate=np.array([min(t.get('face_hold_rate',0) for t in r['trials']) for r in records]),metadata=json_safe(meta))
    library.save(out/'library');np.save(out/'C.npy',C)
    np.savez_compressed(out/'C_history.npz',C=np.array(C_history),elapsed_s=np.array(history_times),
        selected_candidate_ids=np.array(selected),final_valid=mask)
    np.savez_compressed(out/'C_candidates.npz',C=np.array([s['measurement'].get('c',[0,0,0]) for s in screen]).T,
        valid=np.array([s['valid'] for s in screen]),candidate_ids=np.arange(len(screen)))
    report=dict(signature=signature,shape=name,N=n,elapsed_s=time.perf_counter()-start,config=asdict(cfg),
      metadata=meta,authority=authority,valid_modes=int(mask.sum()),candidates=screen,modes=records,
      geometry_rejections=geometry_rejections,rejected_modes=rejected_modes,
      status='COMPLETE' if mask.all() and authority['four_direction_translation'] and authority['rotation_positive'] and authority['rotation_negative'] else 'AUTHORITY_OR_CALIBRATION_INCOMPLETE')
    (out/'report.json').write_text(json.dumps(json_safe(report),indent=2,allow_nan=False))
    (out/'polygon.json').write_text(json.dumps(dict(type='Polygon',coordinates=[list(p.exterior.coords),*[list(r.coords) for r in p.interiors]])))
    return library,report
