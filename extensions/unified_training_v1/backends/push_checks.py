"""Bounded numerical checks for the new discrete-SAC Push backend."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from extensions.unified_training_v1.backends.push import PushBackend, semi_markov_target, _module_hash


def run(output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    backend = PushBackend(dict(num_envs=2, initial_steps=4, collection_steps=2,
                               batch_size=8, validation_repeats=1, test_repeats=1), out,
                          deadline=time.time()+180)
    started = time.perf_counter()
    initialization = backend.initialize()
    collection = backend.collect(initial=True)
    prefit = backend.pretrain_evaluator(2)
    before = _module_hash(backend.actor)
    epoch = backend.train_epoch(0, 0, 2, 2)
    changed = _module_hash(backend.actor) != before
    sample = backend._batch()["data"]
    parameter = backend.actor.head[-1].weight
    backend.actor.zero_grad(set_to_none=True)
    probability, _, valid, cand = backend._distribution(sample, training=True)
    ramp = torch.linspace(-1., 1., probability.shape[-1])
    loss = (probability*ramp).sum(-1)[valid].mean()
    analytic, = torch.autograd.grad(loss, parameter)
    flat = int(analytic.abs().argmax())
    i, j = divmod(flat, parameter.shape[1])
    base = float(parameter[i, j].detach())
    eps = 1e-3
    results = []
    for sign in (1., -1.):
        with torch.no_grad():
            parameter[i, j] = base+sign*eps
        pi, _, mask, _ = backend._distribution(sample, training=True)
        results.append(float((pi*ramp).sum(-1)[mask].mean().detach()))
    with torch.no_grad():
        parameter[i, j] = base
    numeric = (results[0]-results[1])/(2*eps)
    actual = float(analytic[i, j])
    relative = abs(numeric-actual)/max(abs(numeric), abs(actual), 1e-6)
    # V(next) must use the actual next M rather than reusing the current M.
    with torch.no_grad():
        Dn = backend.model.coordinates(cand["pred"])
        _, Mn = backend.actor.conditioned(Dn, cand["condition"].detach())
        independent = torch.einsum("bki,bkij,bkj->bk", Dn, Mn, Dn)
    successor_difference = float((independent-cand["V_next"].detach()).abs().max())
    unavailable = {key:value.clone() for key, value in sample.items()}
    unavailable["reachable"].fill_(False)
    invalid_pi, invalid_logp, invalid_valid, _ = backend._distribution(unavailable)
    reward = torch.tensor([1., 1., 1.])
    value = torch.tensor([10., 10., 10.])
    target = semi_markov_target(reward, value, torch.tensor([True, False, False]),
        torch.tensor([False, True, False]), torch.tensor([2., 2., 1.]), .5, 1.)
    expected = torch.tensor([1., 3.5, 6.])
    checkpoint = out/"snapshot.pt"
    backend.save(checkpoint)
    x = backend._critic_update()
    first_hash = _module_hash(backend.q1)
    backend.load(checkpoint)
    y = backend._critic_update()
    second_hash = _module_hash(backend.q1)
    checkpoint_equivalent = first_hash == second_hash and x == y
    rejected = False
    try:
        backend.save(checkpoint)
    except FileExistsError:
        rejected = True
    tests_rejected = False
    try:
        backend.evaluate("test")
    except RuntimeError:
        tests_rejected = True
    validation = backend.validate()
    checks = dict(actor_changed=changed, prefit_kept_actor=prefit["actor_unchanged"],
        actor_shape_encoder_gradient=epoch["actor_statistics"]["shape_encoder_gradient_norm"] > 0,
        actor_gradient_central_difference=relative < .02,
        full_successor_V=successor_difference < 1e-5,
        invalid_mask_no_sampling=not bool(invalid_valid.any()) and bool(torch.isfinite(invalid_pi).all())
            and bool(torch.isfinite(invalid_logp).all()) and bool(torch.allclose(invalid_pi.sum(-1), torch.ones(len(sample["q"])))),
        termination_truncation_and_time=bool(torch.allclose(target, expected)),
        checkpoint_next_update_exact=checkpoint_equivalent, immutable_checkpoint=rejected,
        no_test_before_freeze=tests_rejected, independent_V_arithmetic=validation["numpy_V_max_difference"] < 1e-5)
    result = dict(passed=all(checks.values()), checks=checks, wall_seconds=time.perf_counter()-started,
        analytic_gradient=actual, numeric_gradient=numeric, relative_gradient_error=relative,
        full_successor_difference=successor_difference, collection=collection, prefit=prefit, epoch=epoch,
        validation={key: value for key, value in validation.items() if key != "episodes"},
        initialization=initialization, scope="software and numerical check, not a performance experiment")
    (out/"CHECKS.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("passed", "checks", "wall_seconds", "relative_gradient_error")}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.output)
    raise SystemExit(0 if result["passed"] else 1)
