"""Recognize module relocation without weakening checkpoint checks."""
MODULE_MOVES = {
    "stabilization_learning.envs.classic.": "stabilization_gym.envs.classic_control.classic.",
    "stabilization_learning.envs.robotics.": "stabilization_gym.envs.robotics.robotics.",
    "stabilization_learning.envs.push_any_shape.": "stabilization_gym.envs.tasks.push_any_shape.",
    "stabilization_learning.envs.visual_servo.": "stabilization_gym.envs.tasks.visual_servo.",
    "stabilization_learning.envs.base.": "stabilization_gym.envs.base.",
}


def canonical_protocol(value):
    if isinstance(value, dict):
        result = {key: canonical_protocol(item) for key, item in value.items()}
        # v0.1 switched these two physical spaces from float32 Box metadata to
        # exact float64 mixed/ball spaces after the first smoke checkpoints.
        # Only those known representational constants are canonicalized; the
        # actual physical guards/controllers were unchanged by that switch.
        task = result.get("class", result.get("physical_environment"))
        if task in ("stabilization_gym.envs.tasks.visual_servo.VisualServoEnv",
                    "stabilization_gym.envs.tasks.push_any_shape.PushAnyShapeEnv"):
            previous_constants = {0.4000000059604645: .4, 0.3499999940395355: .35,
                                  0.6000000238418579: .6}
            previous_constants.update({-key: -item for key, item in list(previous_constants.items())})
            for key in ("action_low", "action_high"):
                if key in result:
                    result[key] = [previous_constants.get(item, item) for item in result[key]]
        return result
    if isinstance(value, list):
        return [canonical_protocol(item) for item in value]
    if isinstance(value, tuple):
        return tuple(canonical_protocol(item) for item in value)
    if isinstance(value, str):
        for old, new in MODULE_MOVES.items():
            if value.startswith(old):
                return new + value[len(old):]
    return value
