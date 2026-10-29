"""Geometry-derived transition times, with optional independent robot calibration."""
from dataclasses import dataclass, asdict
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path
import numpy as np
from scipy.sparse.csgraph import shortest_path
from shapely.geometry import Polygon, Point, LineString
from .. import so101_multishape as m

HOME=np.array([-.05,-.055])

@dataclass(frozen=True)
class TimeConfig:
    length_scale: float=.05
    encoder_omega_multiplier: float=2.
    travels: tuple=(.0004,.001,.002,.0035)
    push_speed: float=.006
    retreat_speed: float=.010
    transit_speed: float=.012
    waypoint_gain: float=8.
    waypoint_tolerance: float=.0003
    tracking_seconds_per_segment: float=.10
    engage_seconds: float=.45
    settle_after_push: float=.5
    settle_before_transfer: float=.5
    terminal_hold: float=1.5
    max_seconds: float=240.
    max_actions: int=100
    time_reference: float=1.
    reward_time: float=1.
    reward_error: float=1.
    reward_input: float=.02
    success_bonus: float=200.
    failure_penalty: float=600.
    reward_scale: float=.01
    selection_time_weight: float=.02
    logit_time_weight: float=.15
    residual_tolerance: float=1e-8
    residual_temperature: float=.05
    progress_temperature: float=.1
    gamma: float=1.
    gae_lambda: float=.95
    discount_reference_seconds: float=1.

    def __post_init__(self):
        # Every scalar is checked before it reaches the plant or policy.
        # In particular omega scaling and terminal rewards previously escaped
        # validation and could poison every candidate or episode at step one.
        for key,value in asdict(self).items():
            if key in ('travels','max_actions'):continue
            if isinstance(value,bool) or not isinstance(value,Real) or not np.isfinite(value):
                raise ValueError(f'Invalid finite scalar: {key}')
        positive=['length_scale','encoder_omega_multiplier','push_speed','retreat_speed','transit_speed','waypoint_gain','waypoint_tolerance','max_seconds','time_reference','reward_scale','discount_reference_seconds','residual_temperature','progress_temperature']
        if any(not np.isfinite(getattr(self,k)) or getattr(self,k)<=0 for k in positive):raise ValueError('Invalid positive scale')
        if (isinstance(self.max_actions,bool) or not isinstance(self.max_actions,Integral) or self.max_actions<1 or
            not isinstance(self.travels,(list,tuple)) or len(self.travels)<1 or
            any(isinstance(t,bool) or not isinstance(t,Real) or not np.isfinite(t) or t<=0 for t in self.travels)):
            raise ValueError('Invalid action grid')
        if not 0<self.gamma<=1 or not 0<self.gae_lambda<=1:raise ValueError('Invalid discount')
        for k in ['tracking_seconds_per_segment','engage_seconds','settle_after_push','settle_before_transfer','terminal_hold','reward_time','reward_error','reward_input','success_bonus','failure_penalty','selection_time_weight','logit_time_weight','residual_tolerance']:
            if not np.isfinite(getattr(self,k)) or getattr(self,k)<0:raise ValueError(f'Invalid cost: {k}')

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False), encoding='utf-8');tmp.replace(path)
def load_config(path=None):return TimeConfig(**(json.loads(Path(path).read_text(encoding='utf-8-sig')) if path else {}))

def validate_arm_timing(cfg):
    """Reject simulator/estimator parameter drift instead of silently comparing it."""
    fixed=dict(push_speed=.006,retreat_speed=.010,waypoint_gain=8.,waypoint_tolerance=.0003,settle_after_push=.5,settle_before_transfer=.5,terminal_hold=1.5)
    bad=[key for key,value in fixed.items() if not np.isclose(getattr(cfg,key),value,rtol=0,atol=1e-12)]
    if bad:raise ValueError('Existing arm executor has fixed values for '+', '.join(bad)+'; update and validate the physical executor before using these changes')

def segment_seconds(length,speed,cfg):
    """Travel at capped speed, then proportional deceleration to tolerance."""
    if length<=cfg.waypoint_tolerance:return 0.
    knee=speed/cfg.waypoint_gain
    if knee<=cfg.waypoint_tolerance:return (length-cfg.waypoint_tolerance)/speed+cfg.tracking_seconds_per_segment
    return max(length-knee,0)/speed+np.log(min(length,knee)/cfg.waypoint_tolerance)/cfg.waypoint_gain+cfg.tracking_seconds_per_segment

def geometry_matrix(poly,points,normals,cfg):
    obstacle=poly.buffer(.0049,quad_segs=4)
    goals=points+normals*.0058;starts=[];retreat=[]
    for p,n in zip(points,normals):
        contact=p+n*.0041;found=None
        for distance in [.004,.003,.002,.0015]:
            end=contact+n*distance
            if poly.distance(Point(end))>.00525 and not LineString([contact+n*.0006,end]).intersects(poly.buffer(.004)):
                found=end;break
        starts.append(found if found is not None else contact);retreat.append(distance if found is not None else np.nan)
    vertices=np.asarray(obstacle.buffer(.00025,join_style=2).exterior.coords)[:-1]
    nodes=np.vstack([starts,HOME[None],goals,vertices]);n=len(nodes)
    adjacency=np.full((n,n),np.inf);np.fill_diagonal(adjacency,0)
    for i in range(n):
        if obstacle.contains(Point(nodes[i])):continue
        for j in range(i):
            if np.any(nodes[[i,j],0]<-.064) or np.any(nodes[[i,j],0]>.069) or np.any(abs(nodes[[i,j],1])>.069):continue
            if not LineString([nodes[i],nodes[j]]).intersects(obstacle):
                cost=segment_seconds(float(np.linalg.norm(nodes[i]-nodes[j])),cfg.transit_speed,cfg)
                adjacency[i,j]=adjacency[j,i]=max(cost,1e-9)
    distance=shortest_path(adjacency,directed=False)
    matrix=distance[:13,13:25].copy()
    for i in range(12):
        matrix[i]+=segment_seconds(retreat[i],cfg.retreat_speed,cfg) if np.isfinite(retreat[i]) else np.inf
    return matrix

def prepare(output,shapes,calibration_root,cfg,measure_pairs=0,seed=20261005):
    """Never fits timings from the previous held-out evaluation JSON files."""
    output=Path(output)
    if measure_pairs:validate_arm_timing(cfg)
    if output.exists():raise FileExistsError(f'Refusing to overwrite {output}; choose a fresh time-bank path')
    records={};rng=np.random.default_rng(seed)
    for name in shapes:
        source=Path(calibration_root)/name/'calibration.json'
        cal=json.loads(source.read_text(encoding='utf-8-sig'));cells,poly=m.geometry(name);points,normals=m.modes(poly)
        times=geometry_matrix(poly,points,normals,cfg);measurements=[];ratios=[]
        pairs=[(i,j) for i in range(12) for j in range(12) if i!=j and np.isfinite(times[i,j])]
        pairs.sort(key=lambda pair:times[pair]);chosen=np.linspace(0,len(pairs)-1,min(measure_pairs,len(pairs)),dtype=int) if pairs else []
        if len(chosen):
            with m.install(name):
                class DummyValue:
                    def value(self,q):return np.array([float(np.dot(q,q))])
                for index in chosen:
                    i,j=pairs[index];q=np.r_[rng.uniform(-.006,.006,2),rng.uniform(-.2,.2)]
                    s=m.c.Plant(m.c.Config(plant='arm',speed=cfg.transit_speed,max_time=60),seed)
                    row=dict(source_mode=i,target_mode=j,pose=q.tolist(),predicted_seconds=float(times[i,j]))
                    try:
                        s.reset(q,q[:2]+m.base.rot(q[2])@(points[i]+normals[i]*.0041))
                        m.c.transfer(s,j,i,DummyValue());actual=s.transfers[-1]['time_s']
                        row.update(ok=True,measured_seconds=actual);ratios.append(actual/times[i,j])
                    except RuntimeError as ex:row.update(ok=False,reason=str(ex))
                    finally:s.close()
                    measurements.append(row)
        factor=float(np.clip(np.median(ratios),.25,4)) if ratios else 1.
        reachable=np.isfinite(times);fitted=np.where(reachable,times*factor,0.)
        records[name]=dict(polygon=list(poly.exterior.coords),points=points.tolist(),normals=normals.tolist(),C=cal['C'],valid=cal['valid'],transfer_seconds=fitted.tolist(),reachable=reachable.tolist(),fit_multiplier=factor,source='robot-calibrated geometry' if ratios else 'geometry estimate; not timing-validated',measurements=measurements,response_sha256=digest(source))
        print(f'prepared {name}: reachable={reachable.sum()}/156, robot fits={len(ratios)}, factor={factor:.3f}',flush=True)
    bank=dict(schema='so101_timebank_v1',config=asdict(cfg),calibration_seed=seed,shapes=records,notes='13 x 12 table: rows 0..11 previous mode, row 12 home/initial. Diagonal is re-engagement cost, not zero; valid maintained contact overrides it to zero. No held-out test records used to fit timings.')
    write_json(output,bank);return bank

class TimeBank:
    def __init__(self,path):
        self.path=Path(path);content=self.path.read_bytes()
        self.sha256=hashlib.sha256(content).hexdigest();data=json.loads(content.decode('utf-8-sig'))
        if data['schema']!='so101_timebank_v1':raise ValueError('Unsupported time bank')
        if not isinstance(data.get('shapes'),dict) or not data['shapes']:
            raise ValueError('The time bank must declare at least one shape')
        self.config=TimeConfig(**data['config']);self.names=list(data['shapes']);self.specs={}
        for name,raw in data['shapes'].items():
            spec=dict(raw)
            for key in ['points','normals','C','transfer_seconds']:spec[key]=np.asarray(raw[key],float)
            for key in ['valid','reachable']:
                values=np.asarray(raw[key])
                if values.dtype.kind != 'b':raise ValueError(f'{name}: {key} must contain booleans')
                spec[key]=values
            spec['poly']=Polygon(raw['polygon'])
            if (spec['C'].shape!=(12,3) or spec['points'].shape!=(12,2) or
                spec['normals'].shape!=(12,2) or spec['valid'].shape!=(12,) or
                spec['reachable'].shape!=(13,12) or spec['transfer_seconds'].shape!=(13,12) or
                not spec['valid'].any()):raise ValueError(f'Invalid shape {name}')
            if (any(not np.isfinite(spec[key]).all() for key in ('points','normals','C','transfer_seconds')) or
                np.any(spec['transfer_seconds']<0) or spec['poly'].is_empty or not spec['poly'].is_valid or
                not np.isfinite(spec['poly'].area) or spec['poly'].area<=0):
                raise ValueError(f'Invalid numerical bank for {name}')
            if not (spec['reachable'][12]&spec['valid']).any():
                raise ValueError(f'{name}: no valid contact reachable from the initial/home state')
            self.specs[name]=spec
    def row(self,name,previous,contact):
        s=self.specs[name];i=previous if previous>=0 else 12
        times=s['transfer_seconds'][i].copy();reachable=s['reachable'][i].copy()&s['valid'];reposition=np.ones(12,bool)
        if previous>=0 and contact:times[previous]=0.;reachable[previous]=s['valid'][previous];reposition[previous]=False
        return times,reachable,reposition
    def contact_usable(self,name,q,previous,pusher):
        if previous is None or previous<0:return False
        spec=self.specs[name];local=m.base.rot(-q[2])@(pusher-q[:2]);gap=spec['poly'].distance(Point(local))-.004
        tangent=np.array([-spec['normals'][previous,1],spec['normals'][previous,0]])
        return -.0003<=gap<=.0006 and abs((local-spec['points'][previous])@tangent)<.0015
