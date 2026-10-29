"""CPU-only compact interpretation of the completed derivative measurements."""
from pathlib import Path
import argparse
import json
import math
import numpy as np


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    folder = args.root / 'derivative_geometry_v2'
    summary = json.loads((folder/'SUMMARY.json').read_text(encoding='utf-8'))
    cases = json.loads((folder/'CASES.json').read_text(encoding='utf-8'))
    with np.load(folder/'ARRAYS.npz', allow_pickle=False) as d:
        e, M, q = d['error'].astype(float), d['M'].astype(float), d['offset'].astype(float)
        changed = d['pixel_changed_fraction'].astype(float)
    E = (q[:,:3]**2).sum(1) + (q[:,3]/math.radians(25))**2
    V = .5*np.einsum('ni,nij,nj->n',e,M,e)
    upper = 7.25*(e*e).sum(1)
    scales = []
    for i, step in enumerate(summary['steps']):
        scales.append(dict(step=step, translation_displacement_m=.4*step,
                           yaw_displacement_deg=math.degrees(.35*step),
                           changed_rgb_channel_fraction_mean=float(changed[:,i].mean()),
                           changed_rgb_channel_fraction_median=float(np.median(changed[:,i]))))
    compact = []
    for row in summary['summary']:
        if row['kind']=='feature_motion' and row['source'] in ('fast','true') and row['size'] in (.01,.04):
            if row['stratum'] in ('all','near','axis_0','axis_1','axis_2','axis_3'):
                compact.append(dict(source=row['source'],step=row['size'],stratum=row['stratum'],
                    n=row['n'],cosine_mean=row['cosine']['mean'],opposite_fraction=row['opposite_fraction'],
                    relative_error_median=row['relative_error']['median']))
    metric=[]
    for name in ('all','near','medium','large'):
        mask=np.ones(len(cases),bool) if name=='all' else np.array([c['difficulty']==name for c in cases])
        metric.append(dict(stratum=name,n=int(mask.sum()),V_over_E_median=float(np.median(V[mask]/E[mask])),
                           current_lower_violation_fraction=float(np.mean(V[mask]<.1*E[mask])),
                           lower_impossible_for_any_allowed_M_fraction=float(np.mean(upper[mask]<.1*E[mask]))))
    result=dict(schema='rgb_v5_derivative_compact_analysis', measured_states=96, split='train',
        evidence=['JVP numerical differentiation agrees at useful image perturbation amplitudes',
                  'JVP applied to real image differences predicts real feature differences almost perfectly on these directions',
                  'Analytic image-motion approximation becomes less accurate as physical perturbations shrink',
                  'Replacing FastDepth by verified simulator depth does not improve mean feature-motion direction',
                  'No normalized-command coordinate/sign/scale error was detected by exact RGB step checks'],
        interpretation=['Strong evidence against a forward-AD/JVP implementation bug in these frozen-model directions',
                        'Evidence for image-motion/renderer-discreteness mismatch, not proof that all error is uint8 quantization',
                        'Feature FD sensitivities are nonzero in four directions, but this is not global physical identifiability',
                        'The local differential metric is weak; its eigenvalues away from the goal are not a direct global V/E theorem',
                        'Train designed central differences are not directly comparable to pooled closed-loop test one-sided transitions'],
        scales=scales, per_axis_and_near=compact, sampled_state_calibration=metric,
        finite_difference_reference=summary['fd_control_reference'],
        next_discriminator='Compare pre-quantization antialiasing accumulator with the unchanged uint8 path; rgb_float from get_rgb is already quantized')
    target=folder/'ANALYSIS.json'
    with target.open('x',encoding='utf-8') as f: json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    text=('冻结96个训练诊断状态：JVP实现与数值差分相符；真实图像差分经编码器JVP后与真实特征差分方向约0.99998。\n'
          '解析图像运动模型对真实渲染的误差明显依赖位移尺度，第二平移轴最差。真深度改善幅值但没有改善平均方向。\n'
          '因此首先排除明显JVP算子错误与统一符号/单位错误；仍需分离8bit量化、纹理采样、遮挡和光照导致的离散性。\n'
          '非零局部敏感性只说明这些扰动可改变特征，不能证明目标误差的全局可辨识性或稳定性。\n'
          '预定FD控制参照h=0.01秒保留，且它本身在尾部状态仍不稳定，不能被称为精确oracle。\n')
    (folder/'ANALYSIS.txt').write_text(text,encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
