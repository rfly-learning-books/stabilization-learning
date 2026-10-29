"""Materialize the already selected overnight protocol and engineering gate."""
from pathlib import Path
from copy import deepcopy
import json

ROOT=Path(__file__).resolve().parents[1]
PYTHON=ROOT/'.venv_rgb_gpu_20261027/Scripts/python.exe'
GENPY=ROOT/'.venv_book_generation_20261029/Scripts/python.exe'
CONFIG=ROOT/'configs/book_overnight_v1'
CONFIG.mkdir(parents=True,exist_ok=True)


def save(path,value):
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))==value:return
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2)


formal=json.loads((ROOT/'extensions/book_generation_overnight_v1/seed41.json').read_text(encoding='utf-8'))
smoke=deepcopy(formal)
smoke.update(mode='engineering',image_size=256,steps=4,updates=20,maximum_minutes=15,
    train_prompts=[{'id':'engtr01','text':'A blue bowl on a gray table.'},{'id':'engtr02','text':'A pencil on plain white paper.'}],
    validation_prompts=[{'id':'engval01','text':'A stone on green moss.'}],
    test_prompts=[{'id':'engtest01','family':'engineering_only','text':'A photograph of a plain wooden chair.'}],test_seeds=[60113])
smoke_path=CONFIG/'generation_engineering.json';save(smoke_path,smoke)

base_env=dict(CUDA_VISIBLE_DEVICES='-1',PYTHONHASHSEED='0',PYTHONUTF8='1',PYTHONIOENCODING='utf-8',
    OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
    HF_HOME=str(ROOT/'.cache/book_generation_hf'))
steps=[]


def step(identifier,tasks,scope,module,entry,args,minutes,expected,*,gpu=False,generation=False,depends=(),minimum=60):
    output=ROOT/'runs'/('book_'+identifier+'_seed41_20261029')
    source=list((ROOT/'extensions'/module).glob('*.py'))+[ROOT/'scripts'/entry]
    for argument in args:
        if str(argument).endswith('.json'):
            source.append(Path(argument))
    command=dict(argv=[str(GENPY if generation else PYTHON),'-X','utf8','-u',str(ROOT/'scripts'/entry),*map(str,args),'--output',str(output)],
        cwd=str(ROOT),timeout_seconds=minutes*60,environment={**base_env,'CUDA_VISIBLE_DEVICES':'0' if gpu else '-1'},
        expected_files=[str(output/name) for name in expected])
    steps.append(dict(id=identifier,task_ids=tasks,scope=scope,commands=[command],maximum_seconds=minutes*60+30,
        minimum_start_seconds=minimum,output_dirs=[str(output)],source_files=list(map(str,source)),depends_on=list(depends)))


step('foundations',['E01','E02','E08'],'Source-linked sampled, PI, virtual-output, output-history and physical resonance subexperiments; original full task acceptance remains separate.',
    'book_foundations_overnight_v1','run_book_foundations_overnight_v1.py',['--max-minutes','10'],10,['RESULTS.json','FINAL_STATE.json','REPORT.md'])
step('disk',['E03','E04','E05'],'One-sided quasi-static disk: known dynamics, direct D, identified dynamics and GMM; matched data budgets and independent residual calibration; E05 only disk excitation diagnostics.',
    'book_disk_overnight_v1','run_book_disk_overnight_v1.py',['--seed','41','--max-minutes','30'],30,['FINAL_STATE.json','REPORT.md','test_scores.csv','paired_comparisons.csv'])
step('e09',['E09'],'24576 scalar stochastic disk-error trajectories with multiplicative and persistent additive noise; physical one-sided-domain exits recorded separately.',
    'book_overnight_v1','run_book_overnight_v1.py',['e09','--seed','41'],10,['RESULTS.json','FIGURES_COMPLETE.json','REPORT.md'])
step('e10',['E10'],'Reviewed 95-state exact MDP with fixed drift certificate, reward and constrained policy iteration, plus 10000 independent rollouts per policy and parameter scans.',
    'book_grid_overnight_v1','run_book_grid_overnight_v1.py',['--seed','41','--episodes','10000','--max-minutes','15'],15,['RESULTS.json','FINAL_STATE.json','BOOK_MODEL_COMPARISON.json'])
step('e13',['E13'],'GPU off-policy SAC on a two-dimensional integrator: reward, soft penalty and exact fixed-certificate action filter; equal transitions and independent nominal/disturbed tests.',
    'book_rl_overnight_v1','run_book_rl_overnight_v1.py',['train','--config',str(ROOT/'extensions/book_rl_overnight_v1/seed41.json')],30,['RESULTS.json','FINAL_STATE.json','FIGURES_COMPLETE.json'],gpu=True)
step('e25',['E25'],'SD1.5 and 20 DDIM steps: four local x0 correction methods, held-out prompt families, paired noise, independent SigLIP; human blind scores and distribution metrics remain unfinished.',
    'book_generation_overnight_v1','run_book_generation_overnight_v1.py',['--config',str(ROOT/'extensions/book_generation_overnight_v1/seed41.json')],90,['FINAL_STATE.json','RESULTS.json','FIGURES_COMPLETE.json'],gpu=True,generation=True,minimum=1200)
step('e20',['E20'],'Book calibration and original A/B; new seed41 C PPO training, 512 numerical plus 64 MuJoCo paired episodes; continuous physical hold and exact M-dependent successor V audited.',
    'book_overnight_v1','run_book_overnight_v1.py',['e20','--config',str(ROOT/'extensions/book_overnight_v1/e20_seed41.json')],120,['RESULTS.json','FINAL_STATE.json','FIGURES_COMPLETE.json'],minimum=1800)
step('observers',['E22'],'Corrected v4 full-sequence observer training for motor, cart-pole and known cooperative target pose; separate noise/mismatch/geometry conditions. E23/E24 unknown-scene and coupled-control tasks not implemented.',
    'book_overnight_v1','run_book_overnight_v1.py',['observers','--config',str(ROOT/'extensions/book_overnight_v1/observers_seed41.json')],90,['RESULTS.json','FINAL_STATE.json','FIGURES_COMPLETE.json'],minimum=1200)

plan=dict(schema='book_overnight_queue_v1',deadline_iso='2026-10-29T09:00:00+08:00',minimum_free_gib=20,
    checklist='E:/rfly-books/output/reports/全书实验验证任务清单_20261028/实验任务清单.json',
    after_rgb=dict(run=str(ROOT/'runs/rgb_book_seed41_20261029_0036'),report=str(ROOT/'artifacts/rgb_book_seed41_20261029_0036_report'),
        curves_marker=str(ROOT/'artifacts/rgb_book_seed41_20261029_0036_report/academic_figures/CURVES_COMPLETE.json')),
    source_files=[str(Path(__file__).resolve()),str(ROOT/'scripts/launch_book_after_rgb_v1.ps1'),str(ROOT/'scripts/analysis/plot_rgb_book_v1.py'),
        str(ROOT/'artifacts/book_overnight_asset_mapping_20261029_v1/ASSET_MAPPING.json'),
        str(ROOT/'artifacts/book_generation_engineering_20261029_0116/FINAL_STATE.json'),
        str(ROOT/'artifacts/book_generation_ddim_audit_20261029/AUDIT.json'),
        str(ROOT/'scripts/report_book_overnight_v1.py')],steps=steps,
    task_notes=dict(E00='Queue protocol, provenance and 32-task evidence matrix are shared deliverables; full-book acceptance remains per original checklist.',
        E14='Env/algo/Runner interfaces exercised by the completed runs; a fresh installation on a different machine is not claimed.',
        E16='RGB book AE+M reproduction is a separate evidence package. The chapter 13 B/J/P/G four-channel ablation is not equivalent.',
        E25='Engineering gate contributes no performance evidence. Human blind evaluation is prepared but not performed.',
        E31='Finite MDP comparisons do not supply an adversarial game experiment.'))
save(CONFIG/'seed41_night_frozen.json',plan)
print(json.dumps(dict(plan=str(CONFIG/'seed41_night_frozen.json'),steps=len(steps)),ensure_ascii=False))
