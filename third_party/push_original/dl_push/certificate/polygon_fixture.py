"""Compile a homogeneous extruded polygon to the planar calibration fixture."""
import itertools
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from shapely.geometry import box
from shapely.geometry.polygon import orient
from shapely.ops import triangulate,unary_union
from shapely.affinity import translate
from .assets import load_geometry
from .model import certificate_xml


def polygon_model(polygon,path,radius=.018,dt=.0005,kv=500):
    if not polygon.is_valid or polygon.geom_type!='Polygon' or polygon.area<=0:
        raise ValueError('a valid connected polygon with positive area is required')
    p=orient(translate(polygon,xoff=-polygon.centroid.x,yoff=-polygon.centroid.y),sign=1)
    root=ET.fromstring(certificate_xml(load_geometry('assets/native_v1','Square').xml,free_body=False,physics_dt=dt,actuator_kv=kv))
    option=root.find('option');option.set('cone','elliptic');option.set('iterations','100');option.set('tolerance','1e-10')
    body=root.find('.//body[@name="object"]')
    for g in list(body.findall('geom')):body.remove(g)
    asset=root.find('asset');asset.clear()
    contact=root.find('contact');contact.clear()
    # Exact axis-aligned slab decomposition, or validated triangle tessellation.
    coords=np.array(p.exterior.coords);edges=np.diff(coords,axis=0)
    rectilinear=np.all(np.isclose(edges[:,0],0)|np.isclose(edges[:,1],0)) and not p.interiors
    pieces=[]
    if rectilinear:
        xs=sorted(set(coords[:,0]))
        for l,r in zip(xs[:-1],xs[1:]):
            cut=p.intersection(box(l,p.bounds[1]-1,r,p.bounds[3]+1))
            for part in ([cut] if cut.geom_type=='Polygon' else getattr(cut,'geoms',[])):
                if part.geom_type=='Polygon' and part.area>1e-12:pieces.append(part)
    elif not p.interiors and p.convex_hull.symmetric_difference(p).area<1e-12:
        # One convex mesh preserves the polygon exactly and avoids artificial
        # coplanar seams / O(vertices) contact pairs for resolved curved hulls.
        pieces=[p]
    else:
        pieces=[t for t in triangulate(p) if p.covers(t)]
        # Merge adjacent triangles whenever their union is convex. Preserve
        # the exact outline, while removing redundant internal collision faces
        # close to short arc facets (which can otherwise contact the ball).
        changed=True
        while changed:
            changed=False
            for i in range(len(pieces)):
                for j in range(i+1,len(pieces)):
                    if pieces[i].boundary.intersection(pieces[j].boundary).length<1e-9:continue
                    union=pieces[i].union(pieces[j])
                    if union.geom_type=='Polygon' and union.convex_hull.symmetric_difference(union).area<1e-12:
                        pieces[i]=union;pieces.pop(j);changed=True;break
                if changed:break
    if not pieces or unary_union(pieces).symmetric_difference(p).area>1e-10:
        raise ValueError('collision decomposition does not exactly cover polygon')
    for i,part in enumerate(pieces):
        name=f'piece_{i}'
        if abs(part.area-box(*part.bounds).area)<1e-12:
            l,b,r,t=part.bounds
            ET.SubElement(body,'geom',name=name,type='box',pos=f'{(l+r)/2} {(b+t)/2} 0',size=f'{(r-l)/2} {(t-b)/2} .02',rgba='.5 .7 .8 1')
        else:
            xy=list(part.exterior.coords)[:-1];vertices=[(x,y,z) for z in (-.02,.02) for x,y in xy]
            ET.SubElement(asset,'mesh',name=name,vertex=' '.join(str(v) for vertex in vertices for v in vertex))
            ET.SubElement(body,'geom',name=name,type='mesh',mesh=name,rgba='.5 .7 .8 1')
        ET.SubElement(contact,'pair',geom1=name,geom2='floor',condim='3',friction='.5 .5 .005 .0001 .0001',solref='.01 1',solimp='.9 .95 .001')
        ET.SubElement(contact,'pair',geom1=name,geom2='pusher_geom',condim='1',friction='0 0 0 0 0',solref='.01 1',solimp='.9 .95 .001')
    xx=yy=xy=0.
    for ring in [p.exterior,*p.interiors]:
        a=np.array(ring.coords);x,y=a[:-1].T;u,v=a[1:].T;c=x*v-u*y
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
