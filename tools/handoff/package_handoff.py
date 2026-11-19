"""Build or verify the light Member A handoff archive (standard library only).

Default inventories the complete local source repository without writing.
--build creates a NEW ZIP. --verify ZIP checks an existing ZIP without extraction.
Rebuilding requires the complete original asset chain; the light ZIP omits caches.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import zipfile

from verify_predictions import ROOT, selected_runs, sha256, output_destination, verify as verify_predictions

# Reuse the already-reviewed standard-library archive verifier, unchanged.
_spec = importlib.util.spec_from_file_location('historical_packager',ROOT/'scripts/package_member_a_deliverables.py')
_legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_legacy)
verify_archive = _legacy.verify_archive
safe_relative = _legacy.safe_relative
PDF = 'output/pdf/HyCal_Member_A_Research_Progress_CN.pdf'
OUTPUT = 'output/handoff/HyCal_Member_A_Code_and_Docs.zip'
REQUIRED_DOCS = ('README_CN.md','docs/handoff/REPRODUCE_CN.md','docs/handoff/CODE_MAP_CN.md',
                 'docs/handoff/METHODS_CN.md','docs/handoff/RESULTS_CN.md','docs/handoff/MATERIALS_CN.md')


def source_locks(root):
    audit=json.loads((root/'reports/part_a_round2/lock_verification.json').read_text(encoding='utf-8'))
    names=sorted({safe_relative(r['lock']) for r in audit['checks']})
    if not audit['passed'] or len(names) != audit['n_locks']:
        raise ValueError('Inconsistent historical lock inventory')
    hashes,checks={},[]
    for name in names:
        lock=json.loads((root/name).read_text(encoding='utf-8'))
        references=lock.get('files',lock.get('provenance'))
        if not isinstance(references,dict) or not references:
            raise ValueError(f'No hash references in {name}')
        for relative,expected in references.items():
            relative=safe_relative(relative.removeprefix('source:').replace('\\','/'))
            if relative not in hashes:
                hashes[relative]=sha256(root/relative)
            if hashes[relative] != expected:
                raise ValueError(f'Historical hash mismatch: {name}: {relative}')
            checks.append({'lock':name,'file':relative,'sha256':expected})
    return {'passed':True,'n_locks':len(names),'n_checks':len(checks),
            'n_unique_files':len(hashes),'checks':checks},hashes


def collect(root):
    root=root.resolve()
    paths=set()
    def add(path):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Missing/linked file: {path}')
        if not path.resolve().is_relative_to(root):
            raise ValueError(f'File outside project: {path}')
        paths.add(path.relative_to(root).as_posix())
    for tree in ('src','scripts','tests','configs','data/manifests','docs','tools/handoff',
                 'reports/part_a','reports/part_a_round2','reports/member_a_brief'):
        for path in sorted((root/tree).rglob('*')):
            if path.is_file() and not any(p in ('__pycache__','.pytest_cache') for p in path.parts) and path.suffix not in ('.pyc','.zip'):
                add(path)
    for relative in ('README.md','.gitattributes',PDF,*REQUIRED_DOCS):
        add(root/relative)
    for run in selected_runs(root):
        for key in ('summary','prediction','record'):
            add(run[key])
    for tree in ('results/part_a','results/part_a_round2'):
        for path in (root/tree).rglob('*.json'):
            if 'archive_before_exact_covariance' in path.parts:
                continue
            if any(token in path.stem for token in ('lock','registration','selection','verification')):
                add(path)
    lock_check,dependencies=source_locks(root)
    # Cache tensors remain in the separately supplied full archive. All other
    # lock dependencies are retained byte-for-byte in this light package.
    for relative in dependencies:
        if not relative.startswith('artifacts/'):
            add(root/relative)
    entries=[{'archive_path':rel,'source_relative_path':rel,'size_bytes':(root/rel).stat().st_size,
              'sha256':sha256(root/rel)} for rel in sorted(paths)]
    if len({r['archive_path'].casefold() for r in entries}) != len(entries):
        raise ValueError('Case-insensitive filename collision')
    return {'schema_version':1,'created_utc':datetime.now(timezone.utc).isoformat(),
            'research_snapshot_commit':'dc8d6a8fd2a158b517b42c715e0ae8a6894b4a2e',
            'handoff_baseline_commit':'c0558b3',
            'scope':'Code, Chinese handoff documents and 336 CSVs for four main methods at five endpoints',
            'omitted_assets':['raw images','model weights','upstream source','virtual environment',
                              'feature tensors','non-main-method prediction CSVs','historical process logs'],
            'historical_hashes_checked_at_original_source':lock_check,
            'historical_dependencies_omitted_from_light_package':[p for p in sorted(dependencies) if p not in paths],
            'reproduction_scope':'Portable main-prediction verification is included. Historical full runners require the full asset chain.',
            'payload_file_count':len(entries),'payload_size_bytes':sum(r['size_bytes'] for r in entries),
            'manifest_self_hash':'Not included to avoid self-reference; payload hashes are integrity checks, not signatures.',
            'files':entries}


def build_archive(root, output, manifest):
    output=output_destination(output,root)
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for row in manifest['files']:
            info=zipfile.ZipInfo(safe_relative(row['archive_path']),date_time=(2026,11,19,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o100644 << 16
            with (root/safe_relative(row['source_relative_path'])).open('rb') as source:
                with archive.open(info,'w',force_zip64=True) as target:
                    shutil.copyfileobj(source,target,1024*1024)
        archive.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return verify_archive(output,root.resolve())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--build',action='store_true')
    mode.add_argument('--verify',type=Path,metavar='ZIP')
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.verify:
        print(json.dumps(verify_archive(args.verify.resolve()),indent=2));return
    root=args.root.resolve()
    manifest=collect(root)
    prediction_check=verify_predictions(root)
    summary={'mode':'build' if args.build else 'dry-run','payload_files':manifest['payload_file_count'],
             'payload_bytes':manifest['payload_size_bytes'],
             'prediction_files_verified':prediction_check['prediction_files'],
             'historical_locks_verified':manifest['historical_hashes_checked_at_original_source']['n_locks'],
             'omitted_historical_dependencies':manifest['historical_dependencies_omitted_from_light_package']}
    if args.build:
        output=(args.output or root/OUTPUT).resolve()
        summary['archive']=str(output)
        summary['verification']=build_archive(root,output,manifest)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    try:
        main()
    except (ValueError,KeyError,OSError,TypeError) as exc:
        print(f'FAIL: {exc}',file=sys.stderr)
        sys.exit(1)
