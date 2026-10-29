"""Independent double-precision finite differences for both new gradient paths."""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from derivative_critic_core import DerivativeCritic, derivative_coefficients, physical_scales
from goal_fixed_core import ROOT, make_model, GoalMetric, policy
from lac_vs.vision import image_tensor


def main():
    torch.set_num_threads(2)
    out=ROOT/'runs/derivative_critic_smoke_20261019_v1'
    cfg=json.loads((out/'protocol.json').read_text(encoding='utf-8'))['config']
    saved=torch.load(out/'best.pt',map_location='cpu')
    model=make_model(cfg,initialize=False).double()
    model.load_state_dict(saved['learner']['ae'])
    critics=torch.nn.ModuleList([DerivativeCritic()]).double()
    critics.load_state_dict(saved['learner']['critics']); critics.requires_grad_(False)
    metric=GoalMetric().double(); metric.load_state_dict(saved['learner']['metric'])
    shard=out/'replay'/saved['replay_rows'][0]['file']
    with np.load(shard) as a:
        rgb=a['rgb'][0]; goal=a['goal_rgb']; offset=a['offset'][0]
    images=image_tensor(cfg,[rgb,goal],'cpu').double()
    r=torch.from_numpy(offset[None]).double()
    velocities=torch.tensor([[.12,-.08,.04,.07]],dtype=torch.double)
    label=(derivative_coefficients(r)*velocities).sum(-1)
    def feature_loss():
        z=model.encode(images)
        return (critics[0](z[:1],z[1:],velocities)-label).square().sum()
    def finite_difference(parameter,loss_function):
        analytic=torch.autograd.grad(loss_function(),parameter)[0]
        index=np.unravel_index(int(analytic.abs().reshape(-1).argmax()),parameter.shape)
        expected=float(analytic[index]); old=float(parameter[index]); eps=1e-5
        with torch.no_grad():
            parameter[index]=old+eps; plus=float(loss_function())
            parameter[index]=old-eps; minus=float(loss_function()); parameter[index]=old
        numerical=(plus-minus)/(2*eps)
        return dict(analytic=expected,finite_difference=numerical,
                    relative_error=abs(expected-numerical)/max(abs(expected),abs(numerical),1e-10),
                    passed=abs(expected-numerical)<1e-7+1e-4*abs(numerical))
    encoder=finite_difference(model.base.encoder[-1].weight,feature_loss)
    with torch.no_grad():
        z=model.encode(images); current=z[:1]; goal_z=z[1:]; e=(current-goal_z)/4
    with np.load(out/'evaluation/fixed_test/learned_P0_0000.npz') as a:
        J=torch.from_numpy(a['J_reference'][None]).double()
    def control_loss():
        u=policy(e,J,metric(goal_z)[1])[0]
        return critics[0](current,goal_z,u*physical_scales(u)).sum()
    actor=finite_difference(metric.net[-1].weight,control_loss)
    videos=0; frames=0; errors=[]
    for path in (out/'evaluation').glob('**/*.mp4'):
        cap=cv2.VideoCapture(str(path)); count=0
        while True:
            ok,_=cap.read()
            if not ok:break
            count+=1
        cap.release(); row=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        if count!=row['steps']+1:errors.append(str(path))
        videos+=1; frames+=count
    result=dict(passed=encoder['passed'] and actor['passed'] and not errors,
                encoder_supervised_gradient=encoder,M_through_controller_derivative_gradient=actor,
                decoded_videos=videos,decoded_frames=frames,video_errors=errors,
                safe_stop_resume_completed=(out/'complete.json').exists())
    (out/'extra_gradient_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))
    if not result['passed']:raise RuntimeError('Independent gradient check failed')


if __name__=='__main__':main()
