"""Scientific capability declarations travel with each new-run protocol."""


def describe(job):
    backend, cfg = job["backend"], job.get("config", {})
    common = dict(backend=backend, task=cfg.get("task", backend), stability_guarantee="none",
                  test_selection=False, numeric_guards=True, historical_source_mutation=False)
    if backend == "state_model_gradient":
        common.update(metric="state-dependent; both endpoints recomputed", model="declared physical state",
                      horizon="full configured horizon with final hold", evidence="local baseline analysis and sampled trajectories")
    elif backend == "observer_model_gradient":
        common.update(metric="estimate-dependent; both endpoints recomputed", model="per-episode scenario geometry",
                      success="no truth/estimate domain or chart failure in complete history AND final hold",
                      certificate="noise allowance is declared empirical slack, not a proven disturbance bound")
    elif backend == "episode_ppo":
        common.update(metric="episode-fixed", reference="discrete-consistent moving references by default",
                      certificate="actual successor; denominator-only norm floor")
        if cfg.get("task") == "visual_servo":
            common.update(model="generic RGB entry, not reconstructed indoor RGB",
                          P1_certificate="image-error candidate only; actuator memory is not part of V")
        elif cfg.get("task") == "push_any_shape":
            common.update(model="generic PushAnyShape; no SO101 transfer time claim")
    elif backend == "dc_motor_gradient":
        common.update(metric="state-dependent; both endpoints recomputed with gradient",
                      controller="same analytic voltage in training and evaluation; no historical CLF screen substitution")
    elif backend == "push_discrete_sac":
        common.update(model="SO101-calibrated response/time reduction; not joint MuJoCo or hardware",
                      objective="declared physical-time discounted reward and entropy", metric="state-dependent full successor")
    elif backend == "rgb_legacy":
        common.update(model="reconstructed RGB; P0 training and declared P0/P1 evaluation",
                      metric="goal-conditioned and fixed within episode",
                      P1_certificate="image-error candidate only; delay and actual velocity memory are outside V",
                      deployment="4-DOF velocity outer loop; no real-time or hardware guarantee")
    return common

