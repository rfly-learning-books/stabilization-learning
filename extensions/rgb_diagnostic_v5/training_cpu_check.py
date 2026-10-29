"""Small CPU arithmetic checks; no optimizer step, rollout, or GPU allocation."""
import json
from pathlib import Path
import torch

from .common import PROJECT, write_json
from .training_ablation import CalibrationRepairLearner, gradient_flags, precision_floor_from_task
from stabilization_learning.joint_v2.visual import TinyRGBBackbone, GoalConditionedMetric, VisualBatch
from stabilization_learning.joint_v3.features import AcceleratedRGBFeatures
from stabilization_learning.joint_v3.visual import AcceleratedVisualLearner
from stabilization_gym.tasks_v2.visual import VisualObservation


def run(output):
    torch.set_num_threads(2)
    torch.manual_seed(8127)
    features = AcceleratedRGBFeatures(TinyRGBBackbone((3, 16, 16), 16, 16), 16)
    metric = GoalConditionedMetric(16, 16)
    base = AcceleratedVisualLearner(features, metric)
    image, goal = torch.rand(4, 3, 16, 16), torch.rand(4, 3, 16, 16)
    offset = torch.tensor([[.03,0,0,0],[.04,0,0,0],[.08,0,0,0],[.20,0,0,0]])
    E = base.adapter.physical_error(offset)
    features.calibrate(image, goal, E)
    # Force a nonzero lower-bound violation to exercise the changed denominator.
    features.scale.mul_(10.)
    tangents = torch.randn(4, 4, 3, 16, 16)*.02
    following = image.roll(1, -1)
    action = torch.full((4,4), .1)
    batch = VisualBatch(VisualObservation(image, goal, tangents), following, offset, offset*.98,
                        action, torch.full((4,), .1), torch.tensor([True,True,False,False]),
                        previous_action=torch.zeros(4,4), behavior_version=0)
    learner = CalibrationRepairLearner(base, calibration_mode="original", precision_floor=.0004)
    rows=[]
    for block in ("feature","metric"):
        with gradient_flags(features, metric, block):
            learner.calibration_mode="original"
            original=learner.losses(batch,feature_graph=block=="feature")
            learner.calibration_mode="cal_precision"
            revised=learner.losses(batch,feature_graph=block=="feature")
            unchanged={}
            for key in original:
                if key not in {"calibration","total"}:
                    unchanged[key]=float((original[key]-revised[key]).abs().max())
                    assert torch.equal(original[key],revised[key]), key
            assert revised["calibration"] > original["calibration"]
            expected=learner.config.calibration_weight*(revised["calibration"]-original["calibration"])
            assert torch.allclose(revised["total"]-original["total"],expected,atol=1e-7,rtol=1e-5)
            zero_terms={}
            if block=="metric":
                for key in ("interaction","reconstruction","geometry"):
                    zero_terms[key]=not revised[key].requires_grad
                    assert zero_terms[key]
            rows.append(dict(block=block,unchanged_max_difference=unchanged,
                             original_cal=float(original["calibration"].detach()),
                             revised_cal=float(revised["calibration"].detach()),metric_terms_zero=zero_terms))
    floor=precision_floor_from_task(dict(collection=dict(position_tolerance_m=.02,yaw_tolerance_deg=1.),
                                        cost=dict(yaw_scale_deg=25.)))
    assert floor == .0004
    alternating=[(b,i) for i in range(600) for b in ("feature","metric")]
    split=[(b,i) for b in ("feature","metric") for i in range(600)]
    for block in ("feature","metric"):
        assert [i for b,i in alternating if b==block]==[i for b,i in split if b==block]==list(range(600))
    assert not base.feature_optimizer.state and not base.metric_optimizer.state
    assert not learner.feature_optimizer.state and not learner.metric_optimizer.state
    result=dict(passed=True,scope="CPU synthetic arithmetic only; not controller performance",
                optimizer_steps=0,device="cpu",tests=rows,precision_floor=floor,
                schedules=dict(each_module_same_order=True,feature_updates=600,metric_updates=600),
                only_calibration_and_total_changed=True)
    write_json(output,result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--output",required=True)
    run(Path(p.parse_args().output))
