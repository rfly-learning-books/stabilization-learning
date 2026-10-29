"""Preserved Push evaluation with the encoding-safe environment dependency."""
import numpy as np
import torch

from .environment import JointPushEnv, vendor_modules, _deployment_logits
from .numerical import require_finite


def evaluate(actor, model, names, seeds, costs, guard=lambda: None):
    """Evaluate the same greedy candidate policy and complete successor V."""
    rows = []
    reached = vendor_modules()["reached"]
    for name in names:
        for seed in seeds:
            env = JointPushEnv(model.bank, [name], int(seed), costs)
            env.reset(shape=name)
            initial, trace = env.plant.q.copy(), []
            while not env.plant.done:
                guard()
                data = model.batch([env.observation()])
                require_finite(data, "evaluation candidate input")
                cand = model.candidates(actor, data)
                require_finite(cand, "evaluation candidate output")
                index = int(_deployment_logits(cand)[0].detach().argmax())
                physical = bool(cand["physical"][0].any())
                before, q = float(cand["V"][0].detach()), env.plant.q.copy()
                error = cand["D"][0].detach().cpu().numpy().astype(float)
                matrix = cand["M"][0].detach().cpu().numpy().astype(float)
                result = env.step(index) if physical or reached(env.plant.spec, env.plant.q) else env.abort()
                obs, reward, term, trunc, info = result
                require_finite((obs, reward, info), "evaluation transition")
                with torch.no_grad():
                    condition = actor.encoder(model.C_encoded[[obs["shape"]]],
                        model.valid[[obs["shape"]]], ell=model.cfg.length_scale)
                    error_next = model.coordinates(torch.as_tensor(env.plant.q[None],
                        device=model.device, dtype=torch.float32))
                    _, matrix_next = actor.conditioned(error_next, condition)
                    following = actor.value(error_next, condition)
                    require_finite((condition, error_next, matrix_next, following), "evaluation successor V")
                en, mn = error_next[0].cpu().numpy().astype(float), matrix_next[0].cpu().numpy().astype(float)
                numpy_v, numpy_vn = float(error @ matrix @ error), float(en @ mn @ en)
                arithmetic = max(abs(numpy_v-before), abs(numpy_vn-float(following[0])))
                if not np.isclose(numpy_v, before, atol=1e-6, rtol=2e-5) or not np.isclose(numpy_vn, float(following[0]), atol=1e-6, rtol=2e-5):
                    raise RuntimeError("Complete successor V arithmetic mismatch")
                trace.append(dict(q=q.tolist(), q_next=env.plant.q.tolist(), V=before,
                    V_next=float(following[0]), error=error.tolist(), error_next=en.tolist(),
                    M=matrix.tolist(), M_next=mn.tolist(), numpy_V_max_difference=arithmetic,
                    action=index, reward=reward, **info))
            rows.append(dict(shape=name, seed=int(seed), initial=initial.tolist(), success=info["success"],
                reason=info["reason"], seconds=env.plant.elapsed, reward=env.total_reward,
                error_integral=sum(t["integrated_error"] for t in trace), switches=env.plant.switches,
                input_square_integral=sum(t["input_square_integral"] for t in trace), trace=trace))
    require_finite(rows, "evaluation episodes")
    return dict(episodes=rows, success_rate=float(np.mean([r["success"] for r in rows])),
        mean_combined_cost=-float(np.mean([r["reward"] for r in rows])),
        mean_seconds=float(np.mean([r["seconds"] for r in rows])),
        mean_error=float(np.mean([r["error_integral"] for r in rows])),
        numpy_V_max_difference=max(t["numpy_V_max_difference"] for r in rows for t in r["trace"]))
