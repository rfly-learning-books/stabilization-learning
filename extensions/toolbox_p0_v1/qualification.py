"""Run E1 local kernels and S0 same-output timings; records all adverse results."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import time
import shutil
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/"scripts",ROOT/"stabilization-gym",ROOT/"stabilization-learning-control"):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
from stabilization_gym import make
from m_learning_suite.task_runtime import FullHorizonTask
from extensions.toolbox_p0_v1.corrected_kernel import CorrectedTensorKernel
from extensions.fast_m_learning_v1.checks.e1_gradients import check_local_gradients
from stabilization_learning.objectives_v2 import descent_terms


def main(output):
    output=Path(output);torch.set_num_threads(1)
    previous=None
    if (output/"KERNEL_QUALIFICATION.json").exists():
        previous=json.loads((output/"KERNEL_QUALIFICATION.json").read_text(encoding="utf-8"))
        shutil.copy2(output/"KERNEL_QUALIFICATION.json",output/("KERNEL_QUALIFICATION_before_"+str(time.time_ns())+".json"))
    reports={}
    for name in ("coupled_linear","point_mass","pendulum","cartpole","two_link_arm","unicycle_tracking","multirotor_tracking"):
        task=FullHorizonTask(make(name),name)
        kernel=CorrectedTensorKernel(task)
        state,_=task.reset(seed=777)
        # Use a declared interior point without changing any check tolerances.
        if name not in ("unicycle_tracking","multirotor_tracking"):
            state=np.asarray(task.goal)+np.arange(1,len(state)+1)*.001
        ledger=[]
        def charge(kind,count,detail):ledger.append(dict(kind=kind,count=count,detail=detail))
        result=check_local_gradients(task,kernel,torch.as_tensor(state,dtype=torch.float64),
            torch.eye(task.error_dim,dtype=torch.float64),charge=charge)
        reports[name]=dict(result=result,charge_ledger=ledger,dt=task.dt)
    timing=[]
    for name in (() if previous is not None else ("coupled_linear","point_mass")):
        task=FullHorizonTask(make(name),name);kernel=CorrectedTensorKernel(task)
        starts=np.stack([task.reset(seed=800+i)[0] for i in range(24)])
        M=torch.eye(task.error_dim,dtype=torch.float64).expand(24,-1,-1)
        start=time.perf_counter();state=torch.as_tensor(starts);cost=torch.zeros(24,dtype=torch.float64)
        for _ in range(task.max_steps):
            data=kernel.transition(state,M);state=data["state_after"];cost+=data["cost"]
        tensor_wall=time.perf_counter()-start
        start=time.perf_counter();scalar_cost=[];scalar_end=[]
        for initial in starts:
            state_i=torch.as_tensor(initial);c=0.
            for _ in range(task.max_steps):
                e,ctx=task.build_error(state_i)
                action=task.inner_env.calculate_action_from_M(M[0],e,context=ctx)
                c+=task.dt*float(e.square().sum())
                state_i=task.inner_env.torch_step(state_i,action)
            scalar_cost.append(c);scalar_end.append(state_i.numpy())
        scalar_wall=time.perf_counter()-start
        equal=bool(np.allclose(cost.numpy(),scalar_cost,rtol=1e-10,atol=1e-10) and
                   np.allclose(state.numpy(),scalar_end,rtol=1e-10,atol=1e-10))
        timing.append(dict(task=name,batch=24,steps=task.max_steps,same_outputs=equal,
            tensor_seconds=tensor_wall,scalar_seconds=scalar_wall,wall_ratio=scalar_wall/tensor_wall,
            scope="mathematical transition and costs only; not learner throughput or sample efficiency"))
    if previous is not None:
        timing=previous["S0"]
    payload=dict(schema="toolbox_p0_E1_S0_v1",E1=reports,S0=timing,
        E1_passed=all(v["result"]["passed"] for v in reports.values()),S0_output_equal=all(t["same_outputs"] for t in timing))
    (output/"KERNEL_QUALIFICATION.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(dict(E1={k:v["result"]["status"] for k,v in reports.items()},S0=timing),ensure_ascii=False))


if __name__=="__main__":main(sys.argv[1])
