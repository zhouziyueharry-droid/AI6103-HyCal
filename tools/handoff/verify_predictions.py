"""Recompute the delivered main results without third-party libraries or model runs.

Default: read-only; prints 20 aggregate rows. --output-dir writes only to a new
directory. Historical runners, locks and analysis reports are never rewritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ('H0', 'H2', 'GDA', 'Propagation')
PRIMARY = (0, 1, 42, 1993)
SEEDS = PRIMARY + tuple(range(2, 18))
# dataset, phase, shot, number of classes, number of query images, support seeds
ENDPOINTS = (
    ('flowers102', 'test', 5, 102, 2463, SEEDS),
    ('dtd', 'test', 5, 47, 1692, SEEDS),
    ('aircraft', 'test', 5, 100, 3333, SEEDS),
    ('joint_flowers_dtd', 'test', 5, 149, 4155, SEEDS),
    ('flowers102', 'test_10shot', 10, 102, 2463, PRIMARY),
)
PROTECTED = {'src','scripts','tests','configs','docs','reports','results','data',
             'artifacts','tools','external','.git','.venv','logs'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha256(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def selected_runs(root):
    """Yield exactly the frozen four main families; never rank by test accuracy."""
    base = root / 'results/part_a_round2'
    selections = {}
    for relative in ('selection.json', 'propagation/selection.json'):
        for item in read_json(base / relative)['selected']:
            config = item['config']
            if config['family'] in FAMILIES:
                require(config['family'] not in selections, 'Duplicate selected family')
                selections[config['family']] = config
    require(set(selections) == set(FAMILIES), 'Missing frozen main-method selection')
    for dataset, phase, shot, classes, count, seeds in ENDPOINTS:
        for family in FAMILIES:
            folder = base / ('propagation' if family == 'Propagation' and shot == 5 else '') / dataset / phase
            summary = folder / 'summary.json'
            rows = [r for r in read_json(summary) if r['config']['family'] == family]
            require(len(rows) == len(seeds) and {r['seed'] for r in rows} == set(seeds),
                    f'{summary}: wrong/duplicate/missing seeds for {family}')
            for row in sorted(rows, key=lambda r: seeds.index(r['seed'])):
                require(row['dataset'] == dataset and row['phase'] == phase and row['shot'] == shot,
                        f'{summary}: wrong endpoint')
                require(row['config'] == selections[family], f'{summary}: changed frozen config')
                tag = row['tag']
                require(isinstance(tag, str) and tag and all(c.isalnum() or c in '_-.' for c in tag),
                        'Invalid result tag')
                yield {'dataset':dataset,'phase':phase,'shot':shot,'classes':classes,
                       'count':count,'family':family,'row':row,'summary':summary,
                       'prediction':folder / f'{tag}_predictions.csv',
                       'record':folder / f'{tag}.json'}


def prediction_metrics(path, classes, expected_n):
    """Return hand-auditable percentage metrics and the complete query identity map."""
    counts, hits = [0] * classes, [0] * classes
    identities = {}
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames == ['sample_id','true_label','pred_label','correct'],
                f'{path}: unexpected CSV schema')
        for row in reader:
            require(None not in row, f'{path}: extra CSV columns')
            sample = row['sample_id']
            require(sample and sample not in identities, f'{path}: empty/duplicate sample ID {sample}')
            true, pred, correct = (int(row[k]) for k in ('true_label','pred_label','correct'))
            require(0 <= true < classes and 0 <= pred < classes, f'{path}: invalid class label')
            require(correct in (0,1) and correct == int(true == pred), f'{path}: wrong correct field')
            identities[sample] = true
            counts[true] += 1
            hits[true] += correct
    require(len(identities) == expected_n and all(counts), f'{path}: wrong count or missing class')
    per_class = [100 * h / n for h, n in zip(hits, counts)]
    result = {'n':len(identities),'top1':100 * sum(hits) / len(identities),
              'macro':statistics.mean(per_class),'per_class':per_class,'class_counts':counts}
    if classes == 149:
        result['equal_domain_macro'] = (statistics.mean(per_class[:102]) + statistics.mean(per_class[102:])) / 2
    return result, identities


def close(actual, expected, label):
    require(isinstance(expected, (int,float)) and not isinstance(expected,bool)
            and math.isfinite(expected) and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-8),
            f'{label}: recomputed={actual}, recorded={expected}')


def compare_record(actual, expected, label):
    require(actual['n'] == expected['n'] and actual['class_counts'] == expected['class_counts'],
            f'{label}: sample/class counts differ')
    for key in ('macro','top1'):
        close(actual[key], expected[key], f'{label}/{key}')
    require(len(actual['per_class']) == len(expected['per_class']), f'{label}: class metric length')
    for i,(a,b) in enumerate(zip(actual['per_class'],expected['per_class'])):
        close(a,b,f'{label}/class {i}')


def verify(root):
    root = root.resolve()
    groups, references, inputs = {}, {}, {}
    total, files = 0, 0
    def record(path):
        relative = path.relative_to(root).as_posix()
        if relative not in inputs:
            inputs[relative] = sha256(path)
    for run in selected_runs(root):
        actual, identities = prediction_metrics(run['prediction'], run['classes'], run['count'])
        compare_record(actual, run['row'], str(run['prediction']))
        individual = read_json(run['record'])
        require(individual == run['row'], f"Individual JSON differs from summary: {run['record']}")
        key = (run['dataset'],run['phase'],run['shot'])
        if key in references:
            require(identities == references[key], f'Query identities/labels differ across methods/seeds: {key}')
        else:
            references[key] = identities
        groups.setdefault((*key,run['family']),[]).append((run['row']['seed'],actual))
        for path in (run['prediction'],run['record'],run['summary']):
            record(path)
        total += actual['n']
        files += 1
    aggregates = []
    for (dataset,phase,shot,family), values in groups.items():
        metric = 'equal_domain_macro' if dataset == 'joint_flowers_dtd' else 'macro'
        agg = {'dataset':dataset,'phase':phase,'shot':shot,'family':family,
               'n_seeds':len(values),'metric':metric,'seeds':[s for s,_ in values]}
        for key in ('macro','top1') + (('equal_domain_macro',) if metric == 'equal_domain_macro' else ()):
            series = [m[key] for _,m in values]
            agg[f'{key}_mean'] = statistics.mean(series)
            agg[f'{key}_sd'] = statistics.stdev(series)
        table_name = 'summary_10shot.csv' if shot == 10 else ('summary_aircraft.csv' if dataset == 'aircraft' else 'summary.csv')
        table = root / 'reports/part_a_round2/analysis' / table_name
        with table.open(encoding='utf-8-sig',newline='') as f:
            matches = [r for r in csv.DictReader(f) if r['dataset'] == dataset and r['phase'] == phase
                       and int(r['shot']) == shot and r['family'] == family
                       and r['seed_group'] == ('primary4' if shot == 10 else 'all20')]
        require(len(matches) == 1, f'{table}: missing/duplicate aggregate {dataset}/{family}')
        expected = matches[0]
        require(int(expected['n_seeds']) == len(values), f'{table}: incorrect n_seeds')
        for key,value in agg.items():
            if key.endswith(('_mean','_sd')):
                close(value,float(expected[key]),f'{table}/{dataset}/{family}/{key}')
        record(table)
        aggregates.append(agg)
    for relative in ('selection.json','propagation/selection.json'):
        record(root / 'results/part_a_round2' / relative)
    return {'passed':True,'scope':'Four frozen main methods; five endpoints; CSV metrics only, no new scoring',
            'prediction_files':files,'prediction_records':total,'aggregate_groups':len(aggregates),
            'scope_note':'Does not rerun models, certify every ablation, or establish independent unseen-test validation.',
            'aggregates':aggregates,'input_sha256':inputs}


def output_destination(destination, root):
    """Resolve a new output location without entering a frozen research tree."""
    destination, root = destination.resolve(), root.resolve()
    require(destination != root, 'Output cannot be the project root')
    if destination.is_relative_to(root):
        require(destination.relative_to(root).parts[0].casefold() not in PROTECTED, 'Output is inside a protected research tree')
    if destination.exists():
        raise FileExistsError(f'Refusing to replace existing output: {destination}')
    return destination


def save_outputs(result, destination, root):
    destination = output_destination(destination, root)
    destination.mkdir(parents=True,exist_ok=False)
    (destination/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if result['aggregates']:
        keys = list(dict.fromkeys(k for row in result['aggregates'] for k in row))
        with (destination/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(result['aggregates'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT,help='Extracted project root (default: script location)')
    parser.add_argument('--output-dir',type=Path,help='Optional NEW directory; never overwrite historical reports')
    args=parser.parse_args()
    result=verify(args.root)
    if args.output_dir:
        save_outputs(result,args.output_dir,args.root)
    print(f"PASS: {result['prediction_files']} CSVs / {result['prediction_records']} prediction records / {result['aggregate_groups']} groups")
    print('Dataset              shot family       n   primary mean +/- SD    top1 mean')
    for row in result['aggregates']:
        metric=row['metric']
        print(f"{row['dataset']:20s} {row['shot']:4d} {row['family']:11s} {row['n_seeds']:2d}  {row[metric+'_mean']:8.4f} +/- {row[metric+'_sd']:.4f}  {row['top1_mean']:8.4f}")
    print('Primary: macro; joint endpoint: equal-domain macro. SD is sample SD across support seeds. All values are percentages.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError,KeyError,OSError,TypeError) as exc:
        print(f'FAIL: {exc}',file=sys.stderr)
        sys.exit(1)
