"""Exact constrained-mesh fallback for new concave calibration geometries.

The original compiler remains untouched. Friction, mass, actuators and time
step are unchanged; only previously incomplete geometric decomposition is fixed.
"""
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from shapely import constrained_delaunay_triangles
from shapely.geometry import box
from shapely.geometry.polygon import orient
from shapely.affinity import translate
from shapely.ops import unary_union
from .polygon_fixture import polygon_model as legacy_polygon_model
from .assets import load_geometry
from .model import certificate_xml


def polygon_model(polygon,path,radius=.018,dt=.0005,kv=500):
    try:
        return legacy_polygon_model(polygon,path,radius,dt,kv)
    except ValueError as exc:
        if 'collision decomposition does not exactly cover polygon' not in str(exc):
            raise
    p=orient(translate(polygon,xoff=-polygon.centroid.x,yoff=-polygon.centroid.y),1.)
    pieces=[t for t in constrained_delaunay_triangles(p).geoms if t.area>1e-13]
    changed=True
    while changed:
        changed=False
        for i in range(len(pieces)):
            for j in range(i+1,len(pieces)):
                if pieces[i].boundary.intersection(pieces[j].boundary).length<1e-9:
                    continue
                union=pieces[i].union(pieces[j])
                if union.geom_type=='Polygon' and union.convex_hull.symmetric_difference(union).area<1e-12:
                    pieces[i]=union;pieces.pop(j);changed=True;break
            if changed:break
    if not pieces or unary_union(pieces).symmetric_difference(p).area>1e-10:
        raise ValueError('constrained decomposition failed exact coverage')
    root=ET.fromstring(certificate_xml(load_geometry('assets/native_v1','Square').xml,
                                       free_body=False,physics_dt=dt,actuator_kv=kv))
    option=root.find('option');option.set('cone','elliptic');option.set('iterations','100');option.set('tolerance','1e-10')
    body=root.find('.//body[@name="object"]')
    for geom in list(body.findall('geom')):body.remove(geom)
    asset=root.find('asset');asset.clear();contact=root.find('contact');contact.clear()
    for i,part in enumerate(pieces):
        name=f'piece_{i}'
        if abs(part.area-box(*part.bounds).area)<1e-12:
            l,b,r,t=part.bounds
            ET.SubElement(body,'geom',name=name,type='box',pos=f'{(l+r)/2} {(b+t)/2} 0',
                          size=f'{(r-l)/2} {(t-b)/2} .02',rgba='.5 .7 .8 1')
        else:
            xy=list(part.exterior.coords)[:-1]
            vertices=[(x,y,z) for z in (-.02,.02) for x,y in xy]
            ET.SubElement(asset,'mesh',name=name,vertex=' '.join(str(v) for vertex in vertices for v in vertex))
            ET.SubElement(body,'geom',name=name,type='mesh',mesh=name,rgba='.5 .7 .8 1')
        ET.SubElement(contact,'pair',geom1=name,geom2='floor',condim='3',friction='.5 .5 .005 .0001 .0001',
                      solref='.01 1',solimp='.9 .95 .001')
        ET.SubElement(contact,'pair',geom1=name,geom2='pusher_geom',condim='1',friction='0 0 0 0 0',
                      solref='.01 1',solimp='.9 .95 .001')
    xx=yy=xy=0.
    for ring in [p.exterior,*p.interiors]:
        a=np.asarray(ring.coords);x,y=a[:-1].T;u,v=a[1:].T;c=x*v-u*y
        xx+=np.sum((x*x+x*u+u*u)*c)/12
        yy+=np.sum((y*y+y*v+v*v)*c)/12
        xy+=np.sum((2*x*y+x*v+u*y+2*u*v)*c)/24
    mass=.8;density=mass/p.area;zz=mass*.04**2/12
    inertial=body.find('inertial');inertial.attrib.clear()
    inertial.set('pos','0 0 0');inertial.set('mass',str(mass))
    inertial.set('fullinertia',f'{density*yy+zz} {density*xx+zz} {density*(xx+yy)} {-density*xy} 0 0')
    root.find('.//geom[@name="pusher_geom"]').set('size',str(radius))
    path.write_text(ET.tostring(root,encoding='unicode'))
    return mujoco.MjModel.from_xml_path(str(path)),p
