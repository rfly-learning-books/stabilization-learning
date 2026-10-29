"""Copy only the original protocol's required assets, with byte-level provenance.

Never modifies the historical experiments. Existing destination bytes must match.
"""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
OLD = Path('D:/书稿论文')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    records = []

    def copy(src, dst):
        expected = sha(src)
        if dst.exists() and sha(dst) != expected:
            raise RuntimeError(f'Refusing to overwrite different bytes: {dst}')
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
        assert sha(dst) == expected
        records.append(dict(source=str(src), destination=str(dst.relative_to(ROOT)),
                            bytes=src.stat().st_size, sha256=expected))

    source = OLD / 'push_any_shape_compact_package/scripts/so101_vendor/dl_push'
    for p in sorted(source.rglob('*.py')):
        copy(p, ROOT / 'third_party/push_original/dl_push' / p.relative_to(source))
    bank = OLD / '_依赖/DL-Push/runs/mode_library_geometric32_20260913'
    reference = OLD / '_依赖/DL-Push/runs/full_m_geometric_20260913'
    manifest = json.loads((reference / 'manifest.json').read_text(encoding='utf-8'))
    for name in manifest['run_config']['train_shapes']:
        for rel in ('polygon.json', 'report.json', 'C.npy', 'library/library.json', 'library/library.npz'):
            copy(bank / name / rel, ROOT / 'assets/push_geometric32' / name / rel)
    for rel in ('manifest.json', 'initial.pt', 'best.pt', 'last.pt', 'history.jsonl',
                'final_best/report.json', 'final_best/trajectory.npz'):
        copy(reference / rel, ROOT / 'assets/original_push_reference' / rel)
    for name, expected in manifest['source_sha256'].items():
        assert sha(ROOT / 'third_party/push_original/dl_push/certificate' / name) == expected

    # Retain the exact visual protocol and its source snapshots without replacing
    # missing renderer/AE code with an invented implementation.
    vs = OLD / 'lac_ae_certificate_vs'
    run = vs / 'runs/ae_derivative_critic_seed41_20261019_v1'
    protocol = json.loads((run / 'protocol.json').read_text(encoding='utf-8'))
    for p in sorted((run / 'provenance').rglob('*.py')):
        copy(p, ROOT / 'assets/original_visual_reference/provenance' / p.relative_to(run / 'provenance'))
    for rel in ('protocol.json', 'resolved_config.yaml', 'scenes.json', 'evaluation_starts.json',
                'training_anchors.json', 'analysis/RESULTS.json'):
        if (run / rel).is_file():
            copy(run / rel, ROOT / 'assets/original_visual_reference' / rel)
    missing = []
    legacy_root = Path(protocol['config']['paths']['legacy_root'])
    for key, expected in protocol['dependencies']['read_only'].items():
        if key.startswith('legacy/'):
            p = legacy_root / 'src' / key.split('/', 1)[1]
            if not p.is_file() or sha(p) != expected:
                missing.append(dict(asset=key, original_path=str(p), expected_sha256=expected,
                                    status='missing' if not p.is_file() else 'hash_mismatch'))
    p = Path(protocol['config']['paths']['initial_autoencoder'])
    if not p.is_file():
        missing.append(dict(asset='initial_autoencoder', original_path=str(p), status='missing',
                            expected_sha256=protocol['dependencies']['read_only']['initial_autoencoder']))
    report = dict(schema=1, purpose='original-protocol migration, not fixed-M baseline',
                  copied_files=records, copied_bytes=sum(x['bytes'] for x in records),
                  push_sources_exact=True, visual_missing_dependencies=missing,
                  visual_closed_loop_ready=not missing)
    target = ROOT / 'migration/original_protocol_assets.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='copied_files'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
