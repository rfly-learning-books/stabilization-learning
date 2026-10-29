"""Bound one newly spawned RGB run; never restart a stopped or failed worker."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time


def supervise(config, output):
    root=Path(__file__).resolve().parents[2]
    output=Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    c=json.loads(Path(config).read_text(encoding='utf-8-sig'))
    deadline=min(time.time()+60*c['maximum_minutes'],c.get('absolute_deadline_unix',float('inf')))
    c['absolute_deadline_unix']=deadline
    name=output.name
    frozen=root/'handoff'/(name+'_CONFIG.json')
    with frozen.open('x',encoding='utf-8') as f: json.dump(c,f,ensure_ascii=False,indent=2,allow_nan=False)
    status=root/'handoff'/(name+'_SUPERVISOR.json')
    args=[sys.executable,str(root/'scripts/run_rgb_eta_return_v1.py'),'train','--config',str(frozen),'--output',str(output)]
    with (root/'handoff'/(name+'_STDOUT.log')).open('x',encoding='utf-8') as out, (root/'handoff'/(name+'_STDERR.log')).open('x',encoding='utf-8') as err:
        child=subprocess.Popen(args,stdout=out,stderr=err,cwd=root,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        def record(**more):
            tmp=status.with_suffix('.tmp')
            tmp.write_text(json.dumps(dict(supervisor_pid=os.getpid(),child_pid=child.pid,command=args,
                deadline_unix=deadline,root=str(output),automatic_resume=False,**more),ensure_ascii=False,indent=2),encoding='utf-8')
            os.replace(tmp,status)
        record(status='running',started_unix=time.time())
        while child.poll() is None and time.time()<deadline:
            try: child.wait(timeout=min(5,max(.01,deadline-time.time())))
            except subprocess.TimeoutExpired: pass
        if child.poll() is None:
            # PID belongs to this exact Popen child and has not been reaped.
            # /T is limited to its Windows venv launcher/worker descendants.
            result=subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True,text=True,timeout=30)
            child.wait(timeout=30)
            record(status='deadline_terminated',finished_unix=time.time(),exit_code=child.returncode,
                termination_result=result.stdout+result.stderr)
        else:
            record(status='exited',finished_unix=time.time(),exit_code=child.returncode)
    return child.returncode
