from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def _selected(rows, threshold, *, adaptive, run_all=False):
    p = np.array([row['p'] for row in rows])
    q = np.array([row['noev'] for row in rows])
    if run_all:
        return np.ones(len(rows), dtype=bool)
    return (q >= .90) | ((q >= .02) & (p >= threshold)) if adaptive else p >= threshold

def _losses(rows, threshold, guarantee, adaptive, run_all=False):
    selected = _selected(rows, threshold, adaptive=adaptive, run_all=run_all)
    values = []
    for change in sorted({row['change'] for row in rows}):
        labels = np.array([row['label'] and row['change'] == change for row in rows])
        missed = int(np.sum(labels & ~selected))
        values.append(float(bool(missed) and missed == int(labels.sum())) if guarantee == 'change'
                      else missed / max(1, int(labels.sum())))
    return np.array(values)

def risk_curve(rows: list[dict], alpha: float, guarantee: str = 'scenario', *,
               adaptive: bool = False) -> dict:
    """Independently evaluate exact-grid CRC; change risk uses all changes as denominator."""
    if not rows or guarantee not in {'change', 'scenario'} or not 0 <= alpha <= 1:
        raise ValueError('nonempty rows, valid guarantee and alpha in [0,1] required')
    if any(type(row['label']) is not bool or not all(np.isfinite(row[k]) and 0 <= row[k] <= 1
                                                   for k in ('p', 'noev')) for row in rows):
        raise ValueError('labels must be boolean and probabilities finite in [0,1]')
    if len({(r['change'], r['scenario']) for r in rows}) != len(rows):
        raise ValueError('duplicate change/scenario pair')
    n = len({row['change'] for row in rows})
    grid = sorted({0., 1., *(row['p'] for row in rows)})
    risks = [_losses(rows, value, guarantee, adaptive).mean() for value in grid]
    if np.any(np.diff(risks) < -1e-12):
        raise ValueError('policy loss is not monotone in lambda')
    valid = [i for i, risk in enumerate(risks) if (n * risk + 1) / (n + 1) <= alpha]
    threshold, certified = (grid[valid[-1]], True) if valid else (0., False)
    losses = _losses(rows, threshold, guarantee, adaptive, not certified)
    selected = _selected(rows, threshold, adaptive=adaptive, run_all=not certified)
    regressing = len({row['change'] for row in rows if row['label']})
    misses = _losses(rows, threshold, 'change', adaptive, not certified).sum()
    cheap = sum(.02 <= row['noev'] < .90 for row in rows)
    return {'n': n, 'alpha': alpha, 'guarantee': guarantee, 'lambda_hat': threshold,
            'certified': certified, 'run_all': not certified,
            'status': 'certified' if certified else 'no feasible finite-sample CRC threshold; run all',
            'risk_metric': 'unconditional_missed_regressing_changes/all_changes' if guarantee == 'change'
            else 'mean_per_change_fraction_of_regressed_scenarios_missed',
            'empirical_risk': float(losses.mean()), 'risk_bound': float((losses.sum() + 1) / (n + 1)),
            'conditional_change_miss_rate': float(misses / regressing) if regressing else None,
            'expensive_fraction': float(selected.mean()),
            'cheap_fraction': cheap / len(rows) if adaptive else 1.,
            'risk_curve': [{'lambda': float(value), 'risk': float(_losses(
                rows, value, guarantee, adaptive).mean())} for value in np.linspace(0, 1, 41)]}

def _load(compiled, probs):
    candidates = [r for r in _rows(compiled) if r['question'] == 'regressed' and r['split'] != 'train']
    examples = {r['id']: r for r in candidates}
    if len(examples) != len(candidates):
        raise ValueError('duplicate compiled regression ID')
    meta = json.loads((probs.parent / 'run_meta.json').read_text())
    fit = set(meta['calibration_partition']['fit_change_ids'])
    risk = set(meta['calibration_partition']['risk_change_ids'])
    if fit & risk or not risk:
        raise ValueError('invalid independent calibration partition')
    scores = {}
    for split in ('calibration', 'test', 'ood'):
        for row in _rows(probs / f'{split}.jsonl'):
            if row['question'] != 'regressed':
                continue
            source = examples.get(row['id'])
            options = dict(zip('ABCDEF', row['options'], strict=False))
            expected = dict(zip('ABCDEF', source['options'])) if source and isinstance(source['options'], list) else (
                source['options'] if source else {})
            temperature = row.get('temperature')
            if (not source or row['id'] in scores or options != expected or row['label'] != source['label']
                    or row['split'] != source['split'] or row['revision'] != meta['revision']
                    or type(temperature) not in (float, int) or not .05 <= temperature <= 20):
                raise ValueError('scaled probability identity/label/revision/temperature mismatch')
            letters = list(options)
            logits = np.asarray(row['logits'][:len(letters)], dtype=float) / temperature
            weights = np.exp(logits - logits.max())
            if not np.allclose(weights / weights.sum(), [row['probs'][k] for k in letters], atol=1e-12, rtol=1e-12):
                raise ValueError('scaled probabilities disagree with logits/temperature')
            scores[row['id']] = {**row, 'p': row['probs'][next(k for k, v in options.items() if v == 'yes')]}
    if set(scores) != set(examples):
        raise ValueError('probability coverage differs from compiled held-out examples')
    rows = []
    for key, source in examples.items():
        if source['variant'] != 'ev' or source['change_id'] in fit:
            continue
        row, noev = scores[key], scores[key.removesuffix(':ev') + ':noev']
        rows.append({'id': key, 'change': source['change_id'], 'scenario': source['scenario_id'],
                     'env': source['env'], 'split': source['split'], 'label': source['options'][
                         'ABCDEF'.index(source['label'])] == 'yes', 'p': row['p'], 'noev': noev['p']})
    actual_risk = {row['change'] for row in rows if row['split'] == 'calibration'}
    if not risk <= actual_risk or any(int(c[1:]) <= 160 for c in actual_risk - risk):
        raise ValueError('risk IDs differ from frozen partition or unapproved original calibration IDs')
    return rows, scores, meta, sorted(fit)

def _ledger(path, rows):
    result = {'decisions': 0, 'reason_counts': {}, 'probability_comparisons': [], 'families': {}}
    if path is None:
        return result
    with sqlite3.connect(f'{path.resolve().as_uri()}?mode=ro', uri=True) as connection:
        cards = {change: json.loads(card) for change, card in connection.execute('select change_id,card_json from changes')}
        result['families'] = {change: [v['family'] for v in card['families']] for change, card in cards.items()}
        reasons = Counter()
        lookup = {(r['change'], r['scenario']): r for r in rows}
        for change, scenario, env, tier, model, raw, reason in connection.execute(
                'select d.change_id,d.scenario_id,c.env,d.tier_used,d.model_version,d.probs_json,d.reason '
                'from decisions d join changes c using(change_id) order by d.created_at,d.decision_id'):
            result['decisions'] += 1
            reasons[f'{env}:{reason}'] += 1
            source = lookup.get((change, scenario))
            if source and len(result['probability_comparisons']) < 20:
                probabilities = json.loads(raw)
                actual = probabilities.get('regressed', probabilities)
                actual = actual.get('yes', actual.get('p_yes')) if isinstance(actual, dict) else actual
                comparable = tier.lower() in {'t2', 'glimmer'} or '+lora@' in model
                variant = 'noev' if tier.startswith('T2-noev') else 'ev'
                expected_p = source['noev'] if variant == 'noev' else source['p']
                result['probability_comparisons'].append({'change': change, 'scenario': scenario,
                    'tier': tier, 'model': model, 'source': f'fine-tuned temperature-scaled {variant}' if comparable
                    else 'T1/other routing; not a Glimmer mismatch', 'ledger_p': actual,
                    'file_p': expected_p, 'absolute_difference': abs(actual - expected_p)
                    if comparable and type(actual) in (int, float) else None})
        result['reason_counts'] = dict(reasons)
    return result

def diagnose(compiled: Path, probs: Path, out: Path, ledger: Path | None = None, *,
             pooled_heldout: bool = False, guarantee: str = 'scenario') -> dict:
    """Write independent CRC diagnostics without modifying scores, labels or the ledger."""
    rows, scores, meta, fit = _load(compiled, probs)
    ledger_result = _ledger(ledger, rows)
    calibration = [r for r in rows if r['split'] == 'calibration']
    pooled = [r for r in rows if r['split'] in {'calibration', 'test'}]
    selected = pooled if pooled_heldout else calibration
    positives = np.array([r['p'] for r in calibration if r['label']])
    negatives = np.array([r['p'] for r in calibration if not r['label']])
    auroc = float(np.mean((positives[:, None] > negatives) + .5 * (positives[:, None] == negatives))) if (
        positives.size and negatives.size) else None
    curves = {env: [risk_curve([r for r in selected if env == 'pooled' or r['env'] == env], a,
                               guarantee, adaptive=pooled_heldout) for a in (.05, .10, .20)]
              for env in sorted({r['env'] for r in selected}) + ['pooled']}
    hard = []
    for change in sorted({r['change'] for r in calibration}):
        group = [r for r in calibration if r['change'] == change]
        loss = float(_losses(group, .1, guarantee, pooled_heldout).mean())
        hard.append({'change_id': change, 'loss': loss,
                     'families': ledger_result['families'].get(change),
                     'regressed_scenarios': [{'scenario': r['scenario'], 'p_regress': r['p']}
                                             for r in group if r['label']]})
    result = {'schema': 'brier-policy-diagnostics-v1', 'guarantee': guarantee,
              'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               [compiled, probs.parent / 'run_meta.json', *sorted(probs.glob('*.jsonl'))]},
              'model': sorted({row['model'] for row in scores.values()}), 'revision': meta['revision'],
              'variant': 'ev', 'temperature_applied': True, 'probability_fit_ids_excluded': fit,
              'risk_changes': len({r['change'] for r in calibration}),
              'independent_pooled_heldout_changes': len({r['change'] for r in pooled}),
              'pooling_used': pooled_heldout, 'curves': curves, 'ledger': ledger_result,
              'sanity': {'regressed_mean_p_yes': float(positives.mean()) if positives.size else None,
                         'nonregressed_mean_p_yes': float(negatives.mean()) if negatives.size else None,
                         'auroc': auroc, 'letter_mapping_flipped': bool(positives.size and negatives.size
                                                                     and positives.mean() < negatives.mean())},
              'confident_misses_at_lambda_0_1': sorted(hard, key=lambda r: (-r['loss'], r['change_id']))}
    result['findings'] = ['sample size: fewer than 20 independent calibration changes'] if len({
        r['change'] for r in selected}) < 20 else []
    result['findings'] += ['semantic yes probability is lower on regressed pairs'] if result['sanity']['letter_mapping_flipped'] else []
    result['findings'] += [f"probability mismatch: {r['change']}:{r['scenario']}" for r in ledger_result['probability_comparisons']
                           if r['absolute_difference'] is not None and r['absolute_difference'] > 1e-6]
    out.mkdir(parents=True, exist_ok=True)
    if pooled_heldout and ledger is not None:
        binary = shutil.which('brier') or str(Path(__file__).resolve().parents[2] / 'target/debug/brier')
        rust_path = out / 'rust-policy.json'
        if rust_path.exists():
            raise FileExistsError(rust_path)
        subprocess.run([binary, 'replay', '--ledger', str(ledger), '--compiled', str(compiled),
                        '--probs', str(probs), '--alphas', '0.05,0.10,0.20', '--guarantee', guarantee,
                        '--out', str(rust_path)], check=True, capture_output=True, text=True)
        rust = {float(k): v for k, v in json.loads(rust_path.read_text())['guide3']['alphas'].items()}
        for row in curves['pooled']:
            actual = rust[row['alpha']]['production_policy']['lambda']
            row['rust_lambda_hat'], row['rust_absolute_difference'] = actual, abs(actual - row['lambda_hat'])
            if row['rust_absolute_difference'] > 1e-9:
                result['findings'].append(f"Rust/Python lambda mismatch at alpha={row['alpha']}")
    path = out / 'policy_diag.json'
    with path.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result

def _main():
    parser = argparse.ArgumentParser(description='Independent measured policy CRC diagnostics')
    for name in ('compiled', 'probs', 'out'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--ledger', type=Path)
    parser.add_argument('--pooled-heldout', action='store_true')
    parser.add_argument('--guarantee', choices=('change', 'scenario'), default='scenario')
    args = parser.parse_args()
    print(json.dumps(diagnose(args.compiled, args.probs, args.out, args.ledger,
                             pooled_heldout=args.pooled_heldout, guarantee=args.guarantee), indent=2))

if __name__ == '__main__':
    _main()
