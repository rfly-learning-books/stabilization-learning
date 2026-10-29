"""Project entry points; run with conda run --no-capture-output -n hydrax."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import load_config
from .runtime import ROOT, REFERENCE, activate_reference, read_json, write_json


def doctor():
    activate_reference()
    import importlib.metadata
    import torch
    import jax
    import mujoco
    import mani_skill
    from starter.priors import verify_checkpoints
    if not torch.cuda.is_available():
        raise RuntimeError("GPU PhysX requires a working NVIDIA GPU")
    if not Path(mani_skill.__file__).resolve().is_relative_to(REFERENCE):
        raise RuntimeError("Wrong ManiSkill import")
    return dict(python=sys.version, executable=sys.executable,
                torch=torch.__version__, jax=jax.__version__, mujoco=mujoco.__version__,
                sapien=importlib.metadata.version("sapien"),
                gpu=torch.cuda.get_device_name(), mjx_backend=jax.default_backend(),
                reference=str(REFERENCE), mani_skill_source=mani_skill.__file__,
                prior_hashes=verify_checkpoints())


def pipeline(config, output):
    from .collection import collect
    from .training import fit
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config.to_dict())
    stages, data_roots, checkpoint = [], [], None
    try:
        for iteration in range(config.training.iterations):
            dataset = output / f"data_{iteration:03d}"
            behavior = config.training.initial_collection if iteration == 0 else config.training.later_collection
            collected = collect(config, dataset, checkpoint=checkpoint, behavior=behavior)
            data_roots.append(dataset)
            fitted = fit(config, data_roots, output / f"fit_{iteration:03d}", initial_checkpoint=checkpoint)
            checkpoint = Path(fitted["best_checkpoint"])
            stages.append(dict(iteration=iteration, collection=collected, fit=fitted))
            write_json(output / "pipeline.json", dict(status="running", stages=stages))
        evaluated = collect(config, output / "evaluation", split="validation",
                            checkpoint=checkpoint, behavior="clf")
        write_json(output / "pipeline.json", dict(status="complete", stages=stages,
                                                   checkpoint=str(checkpoint), evaluation=evaluated))
        return dict(checkpoint=str(checkpoint), evaluation=evaluated)
    except Exception as exc:
        write_json(output / "pipeline.json", dict(status="failed", stages=stages, error=repr(exc)))
        raise


def parser():
    p = argparse.ArgumentParser(description="DL-Push training in the hydrax conda environment")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check runtime, reference imports and prior hashes")
    prepare = sub.add_parser("prepare", help="Build versioned native/custom box-shape visual MJX assets")
    prepare.add_argument("--config", default=str(ROOT / "configs/default.json"))
    for command in ("collect", "train", "evaluate", "run"):
        q = sub.add_parser(command)
        q.add_argument("--config", default=str(ROOT / "configs/default.json"))
        q.add_argument("--output", required=True, help="New output directory; existing output is not overwritten")
        if command == "train":
            q.add_argument("--data", nargs="+", required=True)
            q.add_argument("--resume", help="Observer/optimizer checkpoint, writes into a new output directory")
        elif command in {"collect", "evaluate"}:
            q.add_argument("--checkpoint")
            q.add_argument("--split", choices=["train", "validation", "test"],
                           default="train" if command == "collect" else "validation")
            q.add_argument("--nominal", action="store_true", help="CLF uses uncorrected MJX with the same candidate rules")
            if command == "collect":
                q.add_argument("--behavior", choices=["prior", "clf", "phase"], default="prior")
    inspect = sub.add_parser("inspect", help="Summarize planned denominators and recorded episodes")
    inspect.add_argument("path")
    return p


def main(argv=None):
    # The certificate controller has its own contract-oriented command set.
    # Keep the historical commands below available for reproducibility while
    # allowing ``dl-push build-model ...`` to be the short primary entry.
    forwarded = list(sys.argv[1:] if argv is None else argv)
    certificate_commands = {
        "build-model", "calibrate", "select-modes", "build-transfer",
        "search-q", "train-encoder", "sealed-test",
    }
    if forwarded and forwarded[0] in certificate_commands:
        from .certificate.cli import main as certificate_main
        return certificate_main(forwarded)
    args = parser().parse_args(argv)
    if args.command == "doctor":
        result = doctor()
    elif args.command == "inspect":
        path = Path(args.path)
        candidates = [path / name for name in ("summary.json", "report.json", "pipeline.json")]
        found = next((p for p in candidates if p.exists()), None)
        if found is None:
            raise FileNotFoundError("No completed report; inspect collection.json or error JSON files")
        result = read_json(found)
    else:
        config = load_config(args.config)
        if args.command == "prepare":
            from .geometry import prepare_shape, shape_specs
            specs = shape_specs(config.path(config.shape_catalog) if config.shape_catalog else None)
            names = list(dict.fromkeys(config.train_shapes + config.validation_shapes + config.test_shapes))
            result = {}
            for name in names:
                geometry = prepare_shape(specs[name], config.path(config.asset_root))
                result[name] = dict(path=str(geometry.path), fingerprint=geometry.fingerprint,
                                    parts=geometry.metadata["reconstruction"]["part_count"])
                print(f"prepared {name}", flush=True)
        elif args.command == "train":
            from .training import fit
            result = fit(config, args.data, args.output, args.resume)
        elif args.command == "run":
            result = pipeline(config, args.output)
        else:
            from .collection import collect
            if args.command == "evaluate" and not args.checkpoint and not args.nominal:
                raise ValueError("Evaluation needs --checkpoint, or --nominal for the baseline")
            result = collect(config, args.output, args.split, args.checkpoint,
                             args.behavior if args.command == "collect" else "clf",
                             correction=not args.nominal)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
