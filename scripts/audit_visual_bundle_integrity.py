"""Independently verify the copied numerical migration data and dependencies."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/visual_original_offline_migration_20261022_v1'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    manifests = [('visual_initialization_bundle.json', 'entries', 'assets/original_visual_initialization'),
                 ('visual_source_bundle.json', 'source_records', 'third_party/visual_original')]
    checks = []
    for filename, key, directory in manifests:
        manifest = json.loads((ROOT / 'migration' / filename).read_text(encoding='utf-8'))
        base = (ROOT / directory).resolve()
        for row in manifest[key]:
            path = (base / row['relative']).resolve()
            if not path.is_relative_to(base):
                raise ValueError('Manifest path escapes bundle')
            actual = sha(path)
            checks.append(dict(relative=path.relative_to(ROOT).as_posix(), sha256=actual,
                               matches=actual == row['sha256'], bytes=path.stat().st_size))
    recovery = json.loads((ROOT / 'migration/visual_recovery_candidates/recovery_report.json').read_text(encoding='utf-8'))
    for name in recovery['found']:
        path = ROOT / 'assets/recovered_visual_legacy/src' / name
        actual = sha(path)
        checks.append(dict(relative=path.relative_to(ROOT).as_posix(), sha256=actual,
                           matches=actual == recovery['expected'][name], bytes=path.stat().st_size))
    report = dict(created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  completed=True, passed=all(x['matches'] for x in checks), files=len(checks),
                  bytes=sum(x['bytes'] for x in checks), records=checks, script_sha256=sha(Path(__file__)))
    (OUT / 'ASSET_AUDIT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}))
    if not report['passed']:
        raise AssertionError('Migration asset mismatch')


if __name__ == '__main__':
    main()
