"""Sequential training/evaluation worker; exactly one child uses the GPU."""
import argparse
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def write(path, payload):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temp.replace(path)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--config',required=True)
    args=ap.parse_args();out=Path(args.run).resolve()
    stamp=time.strftime('%Y%m%d_%H%M%S')
    def execute(argv, phase):
        stdout=out/f'{phase}.{stamp}.stdout.log'
        stderr=out/f'{phase}.{stamp}.stderr.log'
        with stdout.open('xb') as fout, stderr.open('xb') as ferr:
            child=subprocess.Popen(argv,cwd=ROOT,stdout=fout,stderr=ferr,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
            write(out/'active_process.json',dict(phase=phase,pid=child.pid,arguments=argv,
                                                stdout=str(stdout),stderr=str(stderr),started=time.time()))
            code=child.wait()
        write(out/f'{phase}_exit.json',dict(exit_code=code,finished=time.time(),
                                           stdout=str(stdout),stderr=str(stderr)))
        print(json.dumps(dict(phase=phase,exit_code=code,stdout=str(stdout),stderr=str(stderr))),flush=True)
        return code
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        if not (out/'training_complete.json').is_file():
            argv=[sys.executable,'-u',str(ROOT/'examples/train_original_protocol.py'),
                  '--config',str(Path(args.config).resolve()),'--output',str(out),'--resume']
            code=execute(argv,'training')
            if code:return code
        if not (out/'training_complete.json').is_file():return 0
        if (out/'STOP').exists():return 0
        argv=[sys.executable,'-u',str(ROOT/'scripts/evaluate_original_push_migration.py'),'--run',str(out)]
        code=execute(argv,'evaluation')
        return code
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':raise SystemExit(main())
