"""Read-only SHA verification of the original lock and every round2 phase lock."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    locks = [ROOT / 'results/part_a/execution_lock.json']
    locks += sorted(p for p in (ROOT / 'results/part_a_round2').rglob('*.json')
                    if 'lock' in p.stem or 'registration' in p.stem)
    required = {'final_execution_lock.json', 'diagnostic_lock.json',
                'aircraft_registration.json', 'tenshot_registration.json'}
    assert required.issubset({p.name for p in locks}), 'Missing required registration/lock'
    actual, checks, errors = {}, [], []
    for lock in locks:
        payload = json.loads(lock.read_text(encoding='utf-8'))
        files = payload.get('files', payload.get('provenance'))
        assert isinstance(files, dict) and files, f'No hash map in {lock}'
        for name, expected in files.items():
            relative = name.removeprefix('source:').replace('\\', '/')
            target = ROOT / relative
            if relative not in actual:
                actual[relative] = (hashlib.sha256(target.read_bytes()).hexdigest()
                                    if target.is_file() else None)
            matched = actual[relative] == expected
            row = {'lock': lock.relative_to(ROOT).as_posix(), 'file': relative,
                   'expected_sha256': expected, 'actual_sha256': actual[relative],
                   'matched': matched}
            checks.append(row)
            if not matched:
                errors.append(row)
    result = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'passed': not errors, 'n_locks': len(locks), 'n_checks': len(checks),
              'n_unique_files': len(actual), 'errors': errors, 'checks': checks,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output = ROOT / 'reports/part_a_round2/lock_verification.json'
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in
                      ['passed', 'n_locks', 'n_checks', 'n_unique_files', 'errors']}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
