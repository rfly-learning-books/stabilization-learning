"""Figures and clearly labelled state-schematic videos; never simulate/train."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[key]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
OUTPUT=HERE/"checks/full_fixtures_v1"
from run_full_fixtures import read_commit,sha,write_json


def main():
    import cv2
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not (OUTPUT/"COMPLETE.json").exists():
        raise RuntimeError("All fixed fixtures must finish before rendering aggregate results")
    figures=OUTPUT/"media"
    figures.mkdir(exist_ok=True)
    if (figures/"MANIFEST.json").exists():
        raise RuntimeError("Existing media is preserved; do not silently rerender")
    results=json.loads((OUTPUT/"RESULTS.json").read_text(encoding="utf-8"))
    manifest={"source_sha256":sha(__file__),"simulation_steps":0,"training_updates":0,
              "video_description":"2D schematic drawn from saved actual physical states; not camera imagery or a real robot", "items":[]}
    for fi,summary in enumerate(results["fixtures"]):
        fixture=summary["fixture"]; name=fixture["name"]
        folder=OUTPUT/f"{fi+1:02d}_{name}"
        records=[read_commit(p) for p in sorted(folder.glob("step_*.pkl.gz"))]
        rows=[r["row"] for r in records]
        states=np.array([rows[0]["state_before"]]+[r["state_after"] for r in rows])
        times=np.arange(len(states))*.02; goal=np.array(fixture["goal_xy"])
        gaps=np.linalg.norm(states[:,3:5]-states[:,:2],axis=1)-.019
        errors=np.column_stack(((states[:,:2]-goal)/.05,states[:,5:7]/.1,states[:,7]/3,
                                states[:,8:10]/.1,np.maximum(0,.002-gaps)/.02))
        values=np.square(errors).sum(axis=1)
        commands=np.array([r["applied_command"] for r in rows])
        fig,axes=plt.subplots(2,2,figsize=(10,7))
        axes[0,0].plot(times,np.linalg.norm(states[:,:2]-goal,axis=1)); axes[0,0].axhline(.008,c="k",ls="--",lw=.8)
        axes[0,0].set_ylabel("disk goal distance [m]")
        axes[0,1].plot(times,values); axes[0,1].set_ylabel("candidate V = e.T I e")
        axes[1,0].plot(times[1:],commands[:,0],label="vx command"); axes[1,0].plot(times[1:],commands[:,1],label="vy command")
        axes[1,0].set_ylabel("pusher velocity command [m/s]"); axes[1,0].legend()
        axes[1,1].plot(times,gaps); axes[1,1].axhline(.002,c="k",ls="--",lw=.8); axes[1,1].set_ylabel("surface gap [m]")
        for ax in axes.flat: ax.grid(alpha=.3); ax.set_xlabel("physical time [s]"); ax.set_xlim(0,30)
        fig.suptitle(name+f" | identity M | final 1s hold: {summary['success']}")
        fig.tight_layout(); png=figures/f"{fi+1:02d}_{name}.png"; fig.savefig(png,dpi=150); plt.close(fig)
        video=figures/f"{fi+1:02d}_{name}.mp4"
        writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*"mp4v"),25.,(800,800))
        if not writer.isOpened(): raise RuntimeError("Video writer failed")
        def point(xy): return (int(400+xy[0]*4300),int(440-xy[1]*4300))
        def label(frame,text,y,color=(40,40,40)):
            cv2.putText(frame,text,(22,y),cv2.FONT_HERSHEY_SIMPLEX,.57,color,1,cv2.LINE_AA)
        indices=list(range(0,len(states),2))
        if indices[-1]!=len(states)-1: indices.append(len(states)-1)
        for i in indices:
            frame=np.full((800,800,3),250,dtype=np.uint8)
            cv2.rectangle(frame,point([-.065,.070]),point([.070,-.070]),(190,190,190),1)
            cv2.circle(frame,point(goal),int(.008*4300),(225,245,225),-1)
            cv2.drawMarker(frame,point(goal),(45,130,45),cv2.MARKER_CROSS,18,2)
            if i:
                track=np.array([point(p) for p in states[:i+1,:2]],dtype=np.int32)
                cv2.polylines(frame,[track],False,(210,165,100),1)
            cv2.circle(frame,point(states[i,:2]),int(.015*4300),(210,150,70),-1)
            cv2.circle(frame,point(states[i,3:5]),int(.004*4300),(65,100,210),-1)
            label(frame,"ACTUAL STATE SCHEMATIC - NOT CAMERA / NOT REAL ROBOT",28)
            label(frame,name+" | identity M | no learning",58)
            label(frame,f"t = {times[i]:.2f} s / 30 s     V = {values[i]:.5f}",88)
            label(frame,f"goal distance = {np.linalg.norm(states[i,:2]-goal):.5f} m    gap = {gaps[i]:.5f} m",118)
            label(frame,"Blue: passive disk     Orange: velocity-controlled pusher",752)
            if i==len(states)-1:
                label(frame,"FINAL HOLD SUCCESS" if summary["success"] else "FINAL HOLD FAILED / TIMEOUT",782,(40,120,40) if summary["success"] else (30,30,200))
            writer.write(frame)
        writer.release()
        cap=cv2.VideoCapture(str(video)); decoded=0
        while True:
            ok,frame=cap.read()
            if not ok: break
            if frame.shape!=(800,800,3): raise RuntimeError("Unexpected decoded video dimensions")
            decoded+=1
        cap.release()
        if decoded!=len(indices): raise RuntimeError("Video frame decode count mismatch")
        manifest["items"].append({"fixture":name,"png":png.name,"video":video.name,"frames":decoded,
            "png_sha256":sha(png),"video_sha256":sha(video),"physical_steps":len(rows),
            "physical_duration_seconds":float(times[-1]),"success":summary["success"]})
    write_json(figures/"MANIFEST.json",manifest)
    print(json.dumps({"rendered":len(manifest["items"]),"all_videos_decoded":True,"new_physics_steps":0}))


if __name__=="__main__": main()
