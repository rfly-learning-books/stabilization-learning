"""Build, verify and film eight T material-contact calibration modes."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
from .fixture import build_model,Fixture,run
from dl_push.certificate.library import ModeLibrary
from dl_push.certificate.math import rot2

RADIUS=.018


def font(size):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)


def frame(f,row,point,normal,polygon,mode,role,shape='T',faces=None,radius=.018):
    RADIUS=radius
    im=Image.new('RGB',(640,720),(249,250,252));d=ImageDraw.Draw(im)
    scale=1400.;center=np.array([320,350])
    def pixel(x):return tuple(center+scale*np.array([x[0],-x[1]]))
    p,yaw,vel,om,ball=f.state();R=rot2(yaw)
    d.line([(320,65),(320,535)],fill='#dfe4eb');d.line([(35,302),(605,302)],fill='#dfe4eb')
    vertices=[pixel(p+R@np.array(x)) for x in polygon.exterior.coords]
    d.polygon(vertices,fill='#b8dbea',outline='#30556a',width=2)
    for hole in polygon.interiors:
        d.polygon([pixel(p+R@np.array(x)) for x in hole.coords],fill='#f9fafc',outline='#30556a',width=2)
    faces=np.asarray([normal] if faces is None else faces)
    offset=RADIUS if len(faces)==1 else RADIUS/float(faces[0]@normal)
    anchor=point+offset*normal
    q=p+R@point;target=p+R@anchor
    bx,by=pixel(ball);radius=RADIUS*scale
    d.ellipse((bx-radius,by-radius,bx+radius,by+radius),fill='#315dda',outline='#142f8b',width=2)
    tx,ty=pixel(target);d.ellipse((tx-radius,ty-radius,tx+radius,ty+radius),outline='#e38b00',width=2)
    x,y=pixel(q);d.line([(x-6,y),(x+6,y)],fill='#d53038',width=3);d.line([(x,y-6),(x,y+6)],fill='#d53038',width=3)
    if len(faces)>1:
        for face in faces:
            x,y=pixel(p+R@(anchor-RADIUS*face))
            d.ellipse((x-3,y-3,x+3,y+3),fill='#d53038')
    def arrow(a,b,color):
        aa=np.array(pixel(a));bb=np.array(pixel(b));delta=bb-aa
        if np.linalg.norm(delta)<1:return
        unit=delta/np.linalg.norm(delta);perp=np.array([-unit[1],unit[0]])
        d.line([tuple(aa),tuple(bb)],fill=color,width=4)
        d.polygon([tuple(bb),tuple(bb-unit*10+perp*5),tuple(bb-unit*10-perp*5)],fill=color)
    arrow(q,q-.024*(R@normal),'#14805c')
    arrow(p,p+vel*.9,'#c2343b')
    if abs(om)>.005:
        x,y=pixel(p);rr=27
        d.arc((x-rr,y-rr,x+rr,y+rr),30,270,fill='#a25715',width=3)
        d.text((x+30,y-12),'CCW' if om>0 else 'CW',font=font(13),fill='#a25715')
    x,y=pixel(p);d.ellipse((x-3,y-3,x+3,y+3),fill='#222222')
    d.text((18,12),f'{shape}  mode {mode:02d}  {role}',font=font(21),fill='#17324d')
    d.text((18,42),'PLANAR CALIBRATION | ball radius 18 mm',font=font(14),fill='#475569')
    d.rectangle((0,536,640,720),fill='#eef2f7')
    lines=[f't={row[0]:.2f} s   input normal speed=40 mm/s',
      f'body v=({row[4]*1000:+.2f}, {row[5]*1000:+.2f}) mm/s',
      f'omega={row[6]:+.5f} rad/s   yaw={np.degrees(row[3]):+.2f} deg',
      f'contact error={row[10]*1000:.3f} mm   normal error={np.degrees(row[11]):.4f} deg',
      f'normal force={row[12]:.2f} N   contact pairs={int(row[13])}',
      'red cross: target point | gold ring: target ball centre',
      'green: inward normal | red arrow: measured world velocity']
    for j,line in enumerate(lines):d.text((16,548+j*23),line,font=font(16 if j<5 else 13),fill='#17324d')
    return np.asarray(im)

