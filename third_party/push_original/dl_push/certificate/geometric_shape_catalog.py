"""32 geometry-diverse training proposals; no mirror pair counted as diversity.

Shapes remain single connected, hole-free polygons. Curves are polygonal
approximations. Flat contact platforms are explicit geometry, not fake normals.
"""
from __future__ import annotations
from pathlib import Path
import json
import math
import numpy as np
from shapely.geometry import Point, LineString, Polygon, box, shape, mapping
from shapely.geometry.polygon import orient
from shapely.affinity import scale, translate
from shapely.ops import unary_union

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'runs/mode_library_curved32_20260912'
INTERMEDIATE=ROOT/'runs/mode_library_balanced32_20260913'
HOOK=ROOT/'runs/full_m_external_shape_20260913/NovelRoundedHook'


def original(name):
    return shape(json.loads((OLD/name/'polygon.json').read_text()))


def modified(name):
    return shape(json.loads((INTERMEDIATE/name/'polygon.json').read_text()))


def bar(a,b,width=.050):
    return LineString([a,b]).buffer(width/2,cap_style=2,join_style=2)


def centred(p):
    p=orient(translate(p,xoff=-p.centroid.x,yoff=-p.centroid.y),1.)
    # Leave room for the 18 mm sphere and >30 mm measured push in the fixture.
    radius=max(abs(v) for v in p.bounds)
    if radius>.138:
        factor=.138/radius
        p=scale(p,xfact=factor,yfact=factor,origin=(0,0))
    if p.geom_type!='Polygon' or not p.is_valid or p.interiors:
        raise ValueError('connected, valid, hole-free shape required')
    return p


def catalog():
    proposals=[]
    def add(name,family,p,prior,description):
        p=centred(p)
        folder=HOOK if prior=='NovelRoundedHook' else OLD/prior
        proposals.append(dict(name=name,family=family,polygon=mapping(p),
                              prior_folder=str(folder),description=description))
    # Six substantially different convex polygonal outlines, not rotations of one rectangle.
    for name,source in [('Square','Square'),('RectWide','RectWide'),('Hexagon','Hexagon')]:
        add(name,'凸多边形',original(source),source,'Unmodified reference outline.')
    add('TruncatedTriangle','凸多边形',modified('Triangle'),'Triangle',
        'Triangle-like outline with an explicit broad flat tip; convex vertex pushes are excluded.')
    add('Trapezoid','凸多边形',original('Trapezoid'),'Trapezoid','Unequal upper/lower widths.')
    add('Parallelogram','凸多边形',original('Parallelogram'),'Parallelogram','Sheared, non-orthogonal side faces.')

    add('Ellipse','曲线与混合凸轮廓',original('Ellipse'),'Ellipse','Unequal ellipse axes; not a rotation-invariant disk.')
    capsule=original('Capsule');l,b,r,t=capsule.bounds
    capsule=capsule.intersection(box(l+.012,b-.01,r-.012,t+.01))
    add('CapsuleFlatEnds','曲线与混合凸轮廓',capsule,'Capsule','Stadium with broad end flats and remaining circular shoulders.')
    add('DShapeFlatCaps','曲线与混合凸轮廓',modified('DShape'),'DShape','D outline with safe horizontal caps and a curved right side.')
    semi=scale(Point(0,0).buffer(.075,resolution=24),xfact=1.3,yfact=1.,origin=(0,0))
    semi=semi.intersection(box(-.063,0,.063,.10))
    add('HalfEllipse','曲线与混合凸轮廓',semi,'Ellipse','Upper half ellipse, broad base, and clipped side contact faces.')
    add('CurvedC','复合曲线与非对称',original('ArcC'),'ArcC',
        'Rounded outside and a circular concave opening; distinct from the rectangular CWide and quarter-annular elbow.')
    add('RoundedTriangle','曲线与混合凸轮廓',modified('RoundedTriangle'),'RoundedTriangle',
        'Three rounded shoulders and a short, safe upper platform.')

    add('L','开口凹槽与弯钩',modified('L'),'L','Right-angle open concavity with 40 mm arms and sphere clearance.')
    add('JHook','开口凹槽与弯钩',shape(json.loads((HOOK/'polygon.json').read_text())),
        'NovelRoundedHook','Previously viewed J-hook; now a development/training candidate, not a blind test.')
    add('CWide','开口凹槽与弯钩',original('C_wide'),'C_wide','Wide rectangular side opening.')
    add('UShape','开口凹槽与弯钩',original('UShape'),'UShape','Long parallel arms with an open central channel.')
    v=unary_union([box(-.073,-.073,.073,-.023),
                   bar((-.050,-.023),(-.100,.098)),bar((.050,-.023),(.100,.098))])
    add('VChannel','开口凹槽与弯钩',v,'UShape','Diverging arms and a broad inner-bottom contact platform.')
    arc=Point(0,0).buffer(.14,resolution=32).difference(Point(0,0).buffer(.05,resolution=32))
    arc=arc.intersection(box(0,0,.115,.115))
    add('QuarterRing','开口凹槽与弯钩',arc,'ArcC','Quarter-annular elbow with inner/outer curvature and explicit outer flats.')

    add('T','分支与骨架',original('T'),'T','Reference T with usable two-face underarm contacts.')
    add('Plus','分支与骨架',original('Plus'),'Plus','Four orthogonal branches.')
    add('H','分支与骨架',original('H'),'H','Two vertical rails linked at the middle.')
    add('IShape','分支与骨架',original('IShape'),'IShape','Two flanges linked by a central web.')
    y=unary_union([box(-.060,-.052,.060,.052),box(-.027,-.125,.027,-.035),
                   bar((-.043,.043),(-.100,.115)),bar((.043,.043),(.100,.115))])
    add('YShape','分支与骨架',y,'Plus','Three branches, with exposed central contact platforms.')
    k=unary_union([box(-.060,-.080,.060,.080),box(-.110,-.140,-.055,.140),
                   bar((.050,.060),(.125,.125)),bar((.050,-.060),(.125,-.125))])
    add('KShape','分支与骨架',k,'H','Long spine plus two diagonal branches; unlike a reflected or scaled T.')

    add('Z','多凹槽与阶梯',original('Z'),'Z','Offset top/bottom bars joined by a web.')
    add('Staircase','多凹槽与阶梯',modified('StepShape'),'StepShape','Three width levels with a broad middle band.')
    f=unary_union([box(-.100,-.110,-.050,.110),box(-.100,.065,.100,.110),
                   box(-.100,-.040,.075,.040)])
    add('FShape','多凹槽与阶梯',f,'AsymT','Two unequal horizontal arms on a long spine.')
    e=unary_union([box(-.095,-.1125,-.045,.1125),
                   box(-.095,.0675,.095,.1125),box(-.095,-.0225,.095,.0225),
                   box(-.095,-.1125,.095,-.0675)])
    add('EShape','多凹槽与阶梯',e,'H','Three arms with two 45 mm open gaps.')
    s=unary_union([box(-.090,.0625,.090,.1075),box(-.090,-.0225,.090,.0225),
                   box(-.090,-.1075,.090,-.0625),box(-.090,-.0225,-.045,.1075),
                   box(.045,-.1075,.090,.0225)])
    add('SShape','多凹槽与阶梯',s,'Z','Oppositely opening slots linked into an S-shaped band.')
    notched=box(-.095,-.100,.095,.100)
    notched=notched.difference(box(-.11,.010,-.030,.060))
    notched=notched.difference(box(.030,-.060,.11,-.010))
    add('DoubleNotch','多凹槽与阶梯',notched,'StepShape','Two opposite, offset 50 mm-wide notches.')

    w=LineString([(-.150,.085),(-.075,-.085),(0,.065),(.075,-.085),(.150,.085)]).buffer(
        .036,resolution=16,join_style=1,cap_style=1)
    add('WShape','复合曲线与非对称',w,'RoundedCross','Four slanted segments, two valleys, and rounded ends.')
    head=box(-.045,-.045,.060,.045).buffer(.030,resolution=24)
    key=unary_union([head,box(.020,-.025,.145,.025),
                     box(.068,-.060,.105,-.025),box(.112,-.060,.150,-.025)])
    add('KeyShape','复合曲线与非对称',key,'RoundedL','Rounded head, offset shaft, and two teeth; narrow faces are not contact candidates.')
    assert len(proposals)==len({x['name'] for x in proposals})==32
    return proposals


def descriptors(p):
    l,b,r,t=p.bounds
    area=p.area;perimeter=p.length
    mirror_x=scale(p,xfact=-1,yfact=1,origin=(0,0))
    mirror_y=scale(p,xfact=1,yfact=-1,origin=(0,0))
    vertices=np.asarray(p.exterior.coords)[:-1]
    edges=np.roll(vertices,-1,axis=0)-vertices
    prev=np.roll(edges,1,axis=0)
    turn=np.arctan2(prev[:,0]*edges[:,1]-prev[:,1]*edges[:,0],np.sum(prev*edges,axis=1))
    lengths=np.linalg.norm(edges,axis=1)
    smooth=(np.abs(turn)>1e-5)&(np.abs(turn)<np.deg2rad(6.1))
    return dict(width_m=r-l,height_m=t-b,area_m2=area,
                aspect_ratio=max(r-l,t-b)/min(r-l,t-b),solidity=area/p.convex_hull.area,
                compactness=4*math.pi*area/perimeter**2,
                approximate_curved_boundary_fraction=float(lengths[smooth].sum()/perimeter),
                sharp_concave_corners=int((turn<np.deg2rad(-15)).sum()),
                reflection_x_overlap=1-p.symmetric_difference(mirror_x).area/(2*area),
                reflection_y_overlap=1-p.symmetric_difference(mirror_y).area/(2*area),
                com_bbox_offset_m=float(np.linalg.norm([(l+r)/2,(b+t)/2])))


def render_catalog(destination,verified=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as Patch
    from matplotlib.font_manager import FontProperties
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    rows=catalog();families=list(dict.fromkeys(x['family'] for x in rows))
    colors=['#2b70a5','#269b95','#de9144','#8960aa','#bc626b','#577641']
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    fig,axes=plt.subplots(4,8,figsize=(19.2,10.6),facecolor='#fafbfc')
    for i,(ax,row) in enumerate(zip(axes.flat,rows)):
        p=shape(row['polygon']);color=colors[families.index(row['family'])]
        ax.add_patch(Patch(np.asarray(p.exterior.coords)*1000,facecolor=color+'35',edgecolor=color,lw=1.8))
        ax.plot(0,0,'+',color='#374151',ms=5,mew=1.)
        ax.set(xlim=(-150,150),ylim=(-150,150),aspect='equal')
        ax.set_title(f'{i+1:02d}  {row["name"]}',fontsize=10,pad=3)
        ax.text(.5,-.04,row['family'],transform=ax.transAxes,ha='center',fontproperties=font,fontsize=8,color=color)
        ax.axis('off')
    title='32个已标定形状：多几何类型推动模式库' if verified else '32个几何多样性候选：不以镜像或轻微尺寸变化凑数'
    subtitle=('同一毫米尺度；十字为质心。32形状×12模式已完成标定；尚未进行新一轮闭环训练验收。'
              if verified else '同一毫米尺度；十字为质心。此图仅为几何候选，不代表模式标定或闭环验收已经通过。')
    fig.suptitle(title,fontproperties=font,fontsize=21,y=.986)
    fig.text(.5,.946,subtitle,
             ha='center',fontproperties=font,fontsize=11,color='#4b5563')
    fig.subplots_adjust(top=.91,bottom=.055,wspace=.12,hspace=.24)
    fig.savefig(destination/('shape_catalog_32.png' if verified else 'geometry_candidates_32.png'),dpi=160,facecolor=fig.get_facecolor())
    plt.close(fig)
    return rows
