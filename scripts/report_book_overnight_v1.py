"""Build a post-writer index of every completed and incomplete work package."""
from pathlib import Path
import argparse
import hashlib
import html
import json
import time


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def link(path):return str(Path(path).resolve()).replace('\\','/')


def run(root,output,snapshot):
    root=Path(root).resolve();output=Path(output).resolve();inventory=read(snapshot)
    if inventory.get('schema')!='unified_training_process_snapshot' or not inventory.get('query_success') or time.time()-inventory['created_unix']>60:
        raise RuntimeError('A successful fresh external complete process snapshot is required')
    for process in inventory['processes']:
        command=(process.get('CommandLine') or '').replace('\\','/').lower()
        if not command or not (('/chatgpt-meetings/' in command and '/scripts/meetings_mcp_entrypoint.py' in command) or ('/conda' in command and ' hook' in command)):
            raise RuntimeError('Unresolved Python process in post-writer inventory: '+str(process))
    if (root/'WRITER.lock').exists():raise RuntimeError('Queue writer remains')
    protocol=read(root/'protocol.json');final=read(root/'FINAL_STATE.json');matrix=read(root/'TASK_STATUS_FINAL.json')
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    rgb=Path(protocol['plan']['after_rgb']['report'])
    rows=[];figures=[]
    status_by_id={r['id']:r for r in final['steps']}
    for step in protocol['plan']['steps']:
        status=status_by_id.get(step['id'],dict(status='not_run'))
        directories=[Path(p) for p in step.get('output_dirs',[])]
        artifacts=[]
        for directory in directories:
            if not directory.exists():continue
            for name in ('REPORT.md','RESULTS.json','FINAL_STATE.json','protocol.json','FIGURES_COMPLETE.json','test_scores.csv','test_results.csv'):
                path=directory/name
                if path.is_file():artifacts.append(dict(name=name,path=link(path),sha256=digest(path)))
            for path in directory.rglob('*.png'):
                if 'figures' in str(path.parent).lower() or path.name=='noise_and_sampling_MC.png':
                    figures.append(dict(step=step['id'],path=link(path),sha256=digest(path)))
        rows.append(dict(id=step['id'],tasks=step['task_ids'],status=status['status'],scope=step['scope'],
            error=status.get('error'),directories=list(map(link,directories)),artifacts=artifacts))
    inventory_record=dict(schema='book_overnight_deliverable_v1',queue=link(root),rgb_report=link(rgb),status=final['status'],
        rows=rows,task_matrix=matrix,figures=figures,process_snapshot=link(snapshot),
        notes=['Single training seed; episode variation is not a cross-seed confidence interval.',
               'Completed commands do not imply full acceptance of an original E00-E31 package.',
               'RGB whole-card telemetry includes approximately 27 seconds of concurrent generation engineering GPU checks around 01:15-01:17.'])
    (output/'DELIVERY.json').write_text(json.dumps(inventory_record,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 夜间训练与全书实验交付索引','','整组训练截止：北京时间2026-10-29 09:00。各实验的原始结果、负结果、源码冻结与图表独立保存。',
        '',f'RGB书稿复现：[完整报告]({link(rgb/"REPORT.md")}) · [曲线与图片]({link(rgb/"academic_figures/CURVES_COMPLETE.json")})',
        '',f'夜间队列状态：**{final["status"]}**。','','|实验包|状态|本轮范围|结果目录|','|---|---|---|---|']
    for row in rows:
        lines.append(f'|{row["id"]} / {", ".join(row["tasks"])}|{row["status"]}|{row["scope"]}|'+ '、'.join(f'[{Path(p).name}]({p})' for p in row['directories'])+'|')
    lines += ['','## 原始32项验收状态','','各编号仍按原任务定义验收。本轮子实验有结果，不能自动把完整应用、实飞或实物项目标记为完成。',
        '','|编号|原始任务|本轮证据状态|','|---|---|---|']
    for row in matrix:lines.append(f'|{row["id"]}|{row["title"]}|{row["state"]}|')
    lines += ['','## 图表索引','','PNG供预览，已输出的PDF用于排版。完整图表和校验值见DELIVERY.json。']
    for figure in figures:lines.append(f'- {figure["step"]}：[ {Path(figure["path"]).name} ]({figure["path"]})')
    lines += ['','单训练种子。随机回合或测试初态的区间不能解释为多训练种子置信区间。RGB整卡GPU记录包含两次约27秒并行生成工程检查；网络训练轨迹与冻结配置均未改动。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    cards=[]
    for figure in figures:
        source=Path(figure['path']).as_uri()
        cards.append(f'<figure><figcaption>{html.escape(figure["step"]+" — "+Path(figure["path"]).name)}</figcaption><a href="{html.escape(source)}"><img loading="lazy" src="{html.escape(source)}"></a></figure>')
    table=''.join('<tr><td>'+html.escape(r['id'])+'</td><td>'+html.escape(r['status'])+'</td><td>'+html.escape(r['scope'])+'</td></tr>' for r in rows)
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>夜间实验图表</title><style>body{max-width:1200px;margin:40px auto;padding:0 24px;font:16px/1.7 system-ui;color:#182332;background:#f5f7fa}table{border-collapse:collapse;background:white}td,th{border:1px solid #ddd;padding:10px}figure{margin:28px 0;padding:16px;background:white}img{max-width:100%;max-height:1000px}figcaption{font-weight:600;margin-bottom:12px}</style><h1>夜间训练与实验图表</h1><p>单训练种子；真实计算结果。各原始任务的完整验收范围见报告索引。</p><p><a href="'+html.escape((rgb/'REPORT.html').as_uri())+'">RGB书稿复现报告</a></p><table><tr><th>实验</th><th>状态</th><th>本轮范围</th></tr>'+table+'</table>'+''.join(cards)+'</html>'
    (output/'REPORT.html').write_text(page,encoding='utf-8')
    return dict(report=link(output/'REPORT.md'),html=link(output/'REPORT.html'),figures=len(figures),status=final['status'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('root','output','process-snapshot'):parser.add_argument('--'+field,required=True)
    args=parser.parse_args();print(json.dumps(run(args.root,args.output,args.process_snapshot),ensure_ascii=False))
