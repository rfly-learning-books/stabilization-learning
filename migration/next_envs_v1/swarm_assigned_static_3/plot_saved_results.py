"""Plot immutable saved records only: zero environment imports/steps/solves."""
import os
os.environ["MPLBACKEND"]="Agg"
from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE/"checks/full_v1"
OUT=ROOT/"analysis"; OUT.mkdir(exist_ok=True)
summary=json.loads((ROOT/"SUMMARY.json").read_text(encoding="utf-8"))
names={"ordinary_identity":"Ordinary start", "exact_target":"Already at target", "known_non_target_deadlock":"Safe non-target deadlock"}
colors=["#1565c0","#00897b","#c62828"]
fig,axes=plt.subplots(3,3,figsize=(15,11),layout="constrained")
manifest={"scope":"saved-record plots only", "new_physical_steps":0,"new_QP_calls":0,"sources":{}}
for i,result in enumerate(summary["results"]):
    name=result["fixture"]; file=ROOT/name/"final_snapshot.json"
    manifest["sources"][name]=hashlib.sha256(file.read_bytes()).hexdigest()
    snap=json.loads(file.read_text(encoding="utf-8")); rows=snap["trace"]
    p=np.array([snap["origin"]["p"]]+[r["state_after"] for r in rows]).reshape(-1,3,2)
    q=np.array(snap["q"]).reshape(3,2)[snap["assignment"]]; u=np.array([r["action"] for r in rows]).reshape(-1,3,2)
    times=np.arange(len(p))*.05; v=np.array([rows[0]["V_before"]]+[r["V_after"] for r in rows])
    ax=axes[0,i]
    for j in range(3):
        ax.plot(p[:,j,0],p[:,j,1],color=colors[j],linewidth=2,label=f"Agent {j+1}")
        ax.scatter(*p[0,j],color=colors[j],s=30,marker="o")
        ax.scatter(*q[j],color=colors[j],s=70,marker="x")
        ax.add_patch(plt.Circle(p[-1,j],.1,color=colors[j],alpha=.12))
    ax.set(xlim=(-1.25,1.25),ylim=(-.45,1.3),aspect="equal",xlabel="x (m)",ylabel="y (m)",
        title=names[name]+(" — held success" if result["success"] else " — timeout"))
    if i==0: ax.legend(fontsize=8,loc="lower left")
    label=names[name]; color=colors[i]
    axes[1,0].plot(times,v,label=label,color=color)
    axes[1,1].plot(times,np.linalg.norm(p-q[None],axis=2).max(axis=1),label=label,color=color)
    axes[1,2].step(times[:-1],np.linalg.norm(u,axis=2).max(axis=1),where="post",label=label,color=color)
    axes[2,0].plot(times[:-1],[r["minimum_safe_margin_m"] for r in rows],label=label,color=color)
    axes[2,1].plot(times[1:],[r["delta_V"] for r in rows],label=label,color=color)
    axes[2,2].plot(times,np.r_[0,np.cumsum([r["cost"]+r["failure_tail"] for r in rows])],label=label,color=color)
for ax in axes[1:].flat: ax.set_xlabel("Time (s)"); ax.grid(alpha=.2)
axes[1,0].set(title="Actual fixed-M certificate",ylabel="V = e^T M e",yscale="symlog",ylim=(-1e-13,2))
axes[1,0].set_yscale("symlog",linthresh=1e-12); axes[1,0].legend(fontsize=8)
axes[1,1].set(title="Assigned position error",ylabel="max agent error (m)"); axes[1,1].axhline(.03,color="black",ls="--",lw=.8)
axes[1,2].set(title="Executed interval speed",ylabel="max agent speed (m/s)"); axes[1,2].axhline(.02,color="black",ls="--",lw=.8)
axes[2,0].set(title="Held-segment minimum distance",ylabel="minimum distance - 0.30 m"); axes[2,0].axhline(0,color="black",lw=.8)
axes[2,1].set(title="Actual per-step certificate change",ylabel="V(next) - V(current)"); axes[2,1].axhline(0,color="black",lw=.8)
axes[2,2].set(title="Full-horizon physical cost",ylabel="integral sum squared error (m^2 s)")
fig.suptitle("Three-agent unit-M engineering: safe nonincrease can coexist with failure to reach the target\n3 fixed fixtures, 600 steps each; no learned policy, no statistical success-rate claim",fontsize=13)
fig.savefig(OUT/"saved_trajectories.png",dpi=160); fig.savefig(OUT/"saved_trajectories.svg"); plt.close(fig)
(OUT/"PLOT_MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"png":str(OUT/"saved_trajectories.png"),"svg":str(OUT/"saved_trajectories.svg"),"new_physical_steps":0,"new_QP_calls":0}))
