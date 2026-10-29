"""Nominal MJX models and verified rear contact regions for the +x primitive."""
from pathlib import Path
import importlib
import os
import hashlib
import sys
import numpy as np
from shapely.geometry import box, LineString, Point
from shapely.ops import unary_union
from shapely import get_coordinates

HYDRAX_ROOT = Path(os.environ.get('HYDRAX_ROOT', 'external_assets/hydrax'))


def activate_hydrax(root=HYDRAX_ROOT):
    root=Path(root).resolve()
    if not (root/'hydrax/algs/mppi.py').is_file():
        raise FileNotFoundError(f'Missing hydrax checkout: {root}')
    for name,module in list(sys.modules.items()):
        if name=='hydrax' or name.startswith('hydrax.'):
            origin=getattr(module,'__file__',None)
            if origin and not Path(origin).resolve().is_relative_to(root):
                raise RuntimeError('Another hydrax is already loaded; start a fresh process')
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    module=importlib.import_module('hydrax')
    if not Path(module.__file__).resolve().is_relative_to(root):raise RuntimeError('Wrong hydrax source')
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (root/'hydrax/algs/mppi.py',root/'hydrax/alg_base.py',root/'hydrax/task_base.py')}


def contact_targets(spec, count=33, radius=.018, gap=.002):
    """First intersections from -x with the inflated true union, not h(n)*n."""
    geometry=unary_union([box(cx-hx,cy-hy,cx+hx,cy+hy) for (cx,cy),(hx,hy) in spec.boxes])
    inflated=geometry.buffer(radius+gap,quad_segs=64)
    lo,hi=geometry.bounds[1],geometry.bounds[3]
    targets=[]
    for y in np.linspace(lo+.002,hi-.002,count):
        coords=get_coordinates(inflated.intersection(LineString([(-.5,y),(.5,y)])))
        if not len(coords):continue
        p=np.array([coords[:,0].min(),y])
        if geometry.distance(Point(p))<radius+gap-1e-5:
            raise ValueError('Preparation target fails circle/shape clearance check')
        targets.append(p)
    if not targets:raise ValueError('No valid +x rear contact targets')
    return np.asarray(targets,np.float32)


def model_xml(spec, mass=.8, iterations=20):
    """Same compound boxes/mass allocation as PhysX; explicit sphere approximation."""
    area=sum(4*hx*hy for _,(hx,hy) in spec.boxes)
    geoms=''.join(f'<geom name="part{i}" type="box" pos="{cx} {cy} 0" size="{hx} {hy} .02" mass="{mass*4*hx*hy/area}"/>'
                  for i,((cx,cy),(hx,hy)) in enumerate(spec.boxes))
    pairs=''.join(f'<pair geom1="part{i}" geom2="{other}" condim="3" friction="{mu} {mu} .005 .0001 .0001" solref=".01 1" solimp=".9 .95 .001"/>'
                  for i in range(len(spec.boxes)) for other,mu in [('floor',1.75),('pusher_geom',2.)])
    return f'''<mujoco model="primitive_x"><size nuserdata="4"/>
    <option timestep=".005" gravity="0 0 -9.81" integrator="implicitfast" solver="Newton" iterations="{iterations}" ls_iterations="4" cone="pyramidal"/>
    <worldbody><geom name="floor" type="plane" size=".35 .35 .01"/>
    <body name="object" pos="0 0 .0205">
    <joint name="obj_x" type="slide" axis="1 0 0"/><joint name="obj_y" type="slide" axis="0 1 0"/>
    <joint name="obj_z" type="slide" axis="0 0 1"/><joint name="obj_yaw" type="hinge" axis="0 0 1" damping=".05"/>{geoms}</body>
    <body name="x_carriage"><inertial pos="0 0 0" mass=".02" diaginertia="1e-5 1e-5 1e-5"/>
    <joint name="pusher_x" type="slide" axis="1 0 0" limited="true" range="-.232 .232"/>
    <body name="pusher" pos="0 0 .0205"><joint name="pusher_y" type="slide" axis="0 1 0" limited="true" range="-.232 .232"/>
    <geom name="pusher_geom" type="sphere" size=".018" mass=".20"/></body></body></worldbody>
    <actuator><velocity joint="pusher_x" kv="100" ctrllimited="true" ctrlrange="-.3 .3" forcelimited="true" forcerange="-30 30"/>
    <velocity joint="pusher_y" kv="100" ctrllimited="true" ctrlrange="-.3 .3" forcelimited="true" forcerange="-30 30"/></actuator>
    <contact>{pairs}</contact></mujoco>'''
