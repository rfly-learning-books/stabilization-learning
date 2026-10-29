"""CLI: python -m dl_push.certificate.generate --all --modes 12 --output PATH."""
import argparse,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import imageio.v2 as imageio
from PIL import Image
from shapely.geometry import shape as geo_shape
from .shape_catalog import catalog
from .mode_generator import generate_modes,GeneratorConfig
from .mode_render import frame


def worker(name,polygon,n,out,cfg,videos):
    frames={i:[] for i in range(n)}
    def render(mode,fx,row,c,p):
        frames[mode].append(frame(fx,row,c.point,c.normal,p,mode,c.role,name,c.faces,cfg.radius))
    lib,r=generate_modes(polygon,n,out,name,cfg,render if videos else None)
    out=Path(out)
    if videos and all(frames.values()):
        for i,ims in frames.items():
            with imageio.get_writer(str(out/f'mode_{i:02}.mp4'),fps=30,codec='libx264',quality=7,ffmpeg_params=['-threads','1']) as w:
                for im in ims:w.append_data(im)
        cols=4;rows=(n+cols-1)//cols
        with imageio.get_writer(str(out/'modes.mp4'),fps=30,codec='libx264',quality=7,macro_block_size=2,ffmpeg_params=['-threads','1']) as w:
            for j in range(max(map(len,frames.values()))):
                grid=Image.new('RGB',(cols*320,rows*360),'white')
                for i,ims in frames.items():grid.paste(Image.fromarray(ims[min(j,len(ims)-1)]).resize((320,360)),((i%cols)*320,(i//cols)*360))
                w.append_data(np.asarray(grid))
            grid.save(out/'overview.png')
    lines=[f'# {name}: {n} measured modes','',f"Status: {r['status']}; valid modes: {r['valid_modes']}/{n}; elapsed: {r['elapsed_s']:.1f}s.",'',
      'C has shape (3,N). Each column maps scalar commanded inward speed (m/s) to object body-frame COM vx, vy (m/s), omega_z (rad/s). Columns are concatenated, not summed. Invalid columns are retained with valid=false and must not be executed.', '',
      '[Video](modes.mp4) | [C.npy](C.npy) | [Full audit](report.json)','',
      '|mode|point x,y (mm)|c_x|c_y|c_omega (rad/m)|valid|held-out error|video|','|---|---|---|---|---|---|---|---|']
    for m in r['modes']:
        x,y=m['geometry']['point'];c=m['c'];i=m['mode']
        lines.append(f"|{i}|{x*1000:.2f}, {y*1000:.2f}|{c[0]:.6f}|{c[1]:.6f}|{c[2]:.6f}|{m['valid']}|{m['validation_error']:.2%}|[mp4](mode_{i:02}.mp4)|")
    lines+=['','Measured authority:','',json.dumps(r['authority'],ensure_ascii=False,indent=2),'',
      'Screening, two-speed regression, independent speed/yaw validation and demonstrations are stored separately. No Q training. L/Z remain sealed-test geometry; no closed-loop test reward is used. Planar fixture only; these matrices are incompatible with the old controller/actuator protocol.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    return {k:r[k] for k in ('shape','N','status','valid_modes','authority','elapsed_s')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--modes',type=int,default=12)
    p.add_argument('--all',action='store_true');p.add_argument('--shapes',nargs='+',default=['T','Square','RectWide','Plus','H'])
    p.add_argument('--polygon',help='GeoJSON Polygon file in metres; centered internally at homogeneous COM')
    p.add_argument('--name',default='custom');p.add_argument('--workers',type=int,default=4)
    p.add_argument('--candidate-budget',type=int,default=40);p.add_argument('--no-video',action='store_true')
    args=p.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    all_shapes=catalog()
    chosen={args.name:geo_shape(json.loads(Path(args.polygon).read_text()))} if args.polygon else {name:all_shapes[name] for name in (all_shapes if args.all else args.shapes)}
    started=time.perf_counter();reports=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(worker,name,poly,args.modes,out/name,GeneratorConfig(dt=.0001 if name=='Plus' else .0005,candidate_budget=args.candidate_budget),not args.no_video):name for name,poly in chosen.items()}
        for future in as_completed(futures):
            name=futures[future]
            try:r=future.result()
            except Exception as e:r=dict(shape=name,status='ERROR',error=repr(e))
            reports.append(r);print(json.dumps(r,ensure_ascii=False),flush=True)
            (out/'campaign.json').write_text(json.dumps(dict(elapsed_s=time.perf_counter()-started,reports=reports),indent=2,ensure_ascii=False))
    lines=['# 模式生成库：形状 → 候选 → 实测 C','', f'共 {len(chosen)} 个形状，目标每形状 {args.modes} 个模式。','',
      '|形状|状态|有效模式|四向近纯平移|正负旋转|结果|','|---|---|---|---|---|---|']
    for r in sorted(reports,key=lambda r:list(chosen).index(r['shape'])):
        name=r['shape'];a=r.get('authority',{})
        lines.append(f"|{name}|{r['status']}|{r.get('valid_modes','?')}/{args.modes}|{a.get('four_direction_translation',False)}|{bool(a.get('rotation_positive')) and bool(a.get('rotation_negative'))}|[结果]({name}/README.md)|")
    lines+=['','每个目录有 C.npy、library/library.npz、逐候选与逐速度实测记录、有效mask、模式视频和模式合集。未通过约束的候选保留但不可用；不能把所有N列都当作有效。','',
      '四方向验收使用实测近纯平移响应，必须覆盖±X、±Y扇区且正锥张成整个XY平面。任意形状并不一定具有这样的接触能力。Q=I与学习仍暂停。']
    (out/'README.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
