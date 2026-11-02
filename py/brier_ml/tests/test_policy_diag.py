import json
import math
import sqlite3

import pytest
from brier_ml.policy_diag import diagnose, risk_curve


def _pairs(n):
    return [{'change': f'c{i:03}', 'scenario': 'pusht-s01', 'label': i % 2 == 0,
             'p': .8 if i % 2 == 0 else .1, 'noev': .5} for i in range(n)]


def test_exact_grid_known_perfect_separation_and_small_sample_safe_fallback():
    report = risk_curve(_pairs(30), .05)
    assert report['lambda_hat'] == .8
    assert report['expensive_fraction'] == .5
    assert report['certified'] and not report['run_all']
    assert len(report['risk_curve']) == 41
    small = risk_curve(_pairs(15), .05)
    assert small['lambda_hat'] == 0
    assert small['expensive_fraction'] == 1
    assert not small['certified'] and small['run_all']


def test_change_loss_detects_one_selected_regression_and_distinguishes_denominators():
    rows = _pairs(30)
    for row in rows:
        rows.append({**row, 'scenario': 'pusht-s02', 'p': .05})
        if len(rows) == 60:
            break
    change = risk_curve(rows, .05, 'change')
    scenario = risk_curve(rows, .05, 'scenario')
    assert change['lambda_hat'] == .8 > scenario['lambda_hat'] == .05
    sparse = [{'change': f'c{i}', 'scenario': 's', 'label': i == 0, 'p': .01, 'noev': .5}
              for i in range(40)]
    result = risk_curve(sparse, .05, 'change')
    assert result['empirical_risk'] == .025
    assert result['conditional_change_miss_rate'] == 1
    assert result['risk_metric'] == 'unconditional_missed_regressing_changes/all_changes'


def test_adaptive_low_high_boundaries_and_uncertified_run_all_bypasses_low_mask():
    rows = _pairs(30)
    for row in rows:
        row['noev'] = .90 if row['label'] else .019
    result = risk_curve(rows, .05, 'change', adaptive=True)
    assert result['lambda_hat'] == 1 and result['expensive_fraction'] == .5
    assert result['cheap_fraction'] == 0
    for row in rows:
        row['noev'] = .019
    result = risk_curve(rows, .05, 'change', adaptive=True)
    assert not result['certified'] and result['run_all']
    assert result['expensive_fraction'] == 1 and result['empirical_risk'] == 0
    assert result['cheap_fraction'] == 0


@pytest.mark.parametrize('alter', [lambda r: r.update(p=math.nan), lambda r: r.update(label=1),
                                   lambda r: r.update(noev=1.01)])
def test_invalid_probabilities_and_labels_fail_loudly(alter):
    rows = _pairs(30)
    alter(rows[0])
    with pytest.raises(ValueError, match='boolean|finite'):
        risk_curve(rows, .05)
    with pytest.raises(ValueError, match='duplicate'):
        risk_curve(_pairs(30) * 2, .05)


def _inputs(tmp_path):
    model = tmp_path / 'model'
    probs = model / 'probs_temp'
    probs.mkdir(parents=True)
    meta = {'revision': 'a' * 40, 'calibration_partition': {
        'fit_change_ids': ['c001'], 'risk_change_ids': [f'c{i:03}' for i in range(2, 32)]}}
    (model / 'run_meta.json').write_text(json.dumps(meta))
    compiled, split_rows = [], {'calibration': [], 'test': [], 'ood': []}
    for i in range(1, 34):
        split = 'calibration' if i < 32 else 'test' if i == 32 else 'ood'
        for variant in ('ev', 'noev'):
            p = .8 if i % 2 == 0 else .1
            row = {'id': f'c{i:03}:pusht-s01:regressed:{variant}', 'change_id': f'c{i:03}',
                   'pair_id': f'c{i:03}:pusht-s01', 'scenario_id': 'pusht-s01', 'env': 'pusht',
                   'split': split, 'question': 'regressed', 'variant': variant,
                   'label': 'B' if i % 2 == 0 else 'A', 'options': ['no', 'yes']}
            compiled.append(row)
            split_rows[split].append({**row, 'logits': [math.log(1 - p), math.log(p)],
                                      'probs': {'A': 1 - p, 'B': p}, 'model': 'FT',
                                      'revision': meta['revision'], 'temperature': 1.})
    for split, rows in split_rows.items():
        (probs / f'{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    source = tmp_path / 'compiled.jsonl'
    source.write_text(''.join(json.dumps(r) + '\n' for r in compiled))
    return source, probs


def test_diagnose_excludes_fit_ood_pools_changes_and_semantic_yes(tmp_path):
    compiled, probs = _inputs(tmp_path)
    before = compiled.read_bytes()
    report = diagnose(compiled, probs, tmp_path / 'out', pooled_heldout=True, guarantee='change')
    assert report['risk_changes'] == 30 and report['independent_pooled_heldout_changes'] == 31
    assert report['curves']['pooled'][0]['n'] == 31
    assert report['sanity']['regressed_mean_p_yes'] == pytest.approx(.8)
    assert report['sanity']['nonregressed_mean_p_yes'] == pytest.approx(.1)
    assert report['sanity']['auroc'] == 1 and not report['sanity']['letter_mapping_flipped']
    assert report['ledger']['decisions'] == 0
    assert compiled.read_bytes() == before
    assert json.loads((tmp_path / 'out/policy_diag.json').read_text()) == report
    with pytest.raises(FileExistsError):
        diagnose(compiled, probs, tmp_path / 'out')


def test_diagnose_refuses_wrong_source_or_probability_metadata(tmp_path):
    compiled, probs = _inputs(tmp_path)
    path = probs / 'test.jsonl'
    original = path.read_text()
    row = json.loads(original.splitlines()[0])
    row['temperature'] = None
    path.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='temperature'):
        diagnose(compiled, probs, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()
    row['temperature'] = 2
    path.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='logits/temperature'):
        diagnose(compiled, probs, tmp_path / 'out')


def test_ledger_sources_separate_t1_routing_and_count_actual_reasons(tmp_path):
    compiled, probs = _inputs(tmp_path)
    ledger = tmp_path / 'ledger.sqlite'
    with sqlite3.connect(ledger) as db:
        db.execute('create table changes(change_id,env,card_json)')
        db.execute('create table decisions(change_id,scenario_id,tier_used,model_version,probs_json,reason,created_at,decision_id)')
        db.execute('insert into changes values(?,?,?)', ('c002', 'pusht', json.dumps({'families': [{'family': 'obs_noise'}]})))
        for i, (tier, model, p) in enumerate([('T1', 't1@digest', 0.), ('T2', 'base+lora@digest', .8)]):
            db.execute('insert into decisions values(?,?,?,?,?,?,?,?)',
                       ('c002', 'pusht-s01', tier, model, json.dumps({'regressed': {'yes': p}}),
                        'policy_above_lambda', 0, i))
    before = ledger.read_bytes()
    report = diagnose(compiled, probs, tmp_path / 'out', ledger)
    result = report['ledger']
    assert result['decisions'] == 2 and result['reason_counts'] == {'pusht:policy_above_lambda': 2}
    assert result['probability_comparisons'][0]['absolute_difference'] is None
    assert result['probability_comparisons'][1]['absolute_difference'] == 0
    assert ledger.read_bytes() == before


def test_pooled_diagnostics_invokes_actual_existing_replay_route_and_compares(tmp_path, monkeypatch):
    from brier_ml import policy_diag
    compiled, probs = _inputs(tmp_path)
    ledger = tmp_path / 'ledger.sqlite'
    with sqlite3.connect(ledger) as db:
        db.execute('create table changes(change_id,env,card_json)')
        db.execute('create table decisions(change_id,scenario_id,tier_used,model_version,probs_json,reason,created_at,decision_id)')
    def run(command, **kwargs):
        assert command[:2] == ['/local/brier', 'replay']
        assert command[command.index('--guarantee') + 1] == 'change'
        assert command[command.index('--probs') + 1] == str(probs)
        path = command[command.index('--out') + 1]
        with open(path, 'w') as stream:
            json.dump({'guide3': {'alphas': {a: {'production_policy': {'lambda': .8}}
                                              for a in ('0.05', '0.10', '0.20')}}}, stream)
    monkeypatch.setattr(policy_diag.shutil, 'which', lambda _: '/local/brier')
    monkeypatch.setattr(policy_diag.subprocess, 'run', run)
    report = diagnose(compiled, probs, tmp_path / 'out', ledger, pooled_heldout=True, guarantee='change')
    assert all(r['rust_absolute_difference'] == 0 for r in report['curves']['pooled'])
    assert report['findings'] == []


@pytest.mark.parametrize('new_id,valid', [('c161', True), ('c100', False)])
def test_additional_calibration_accepts_only_global_extension_ids_and_preserves_original_risk(
    tmp_path, new_id, valid,
):
    compiled, probs = _inputs(tmp_path)
    originals = [json.loads(line) for line in compiled.read_text().splitlines()]
    score_rows = [json.loads(line) for line in (probs / 'test.jsonl').read_text().splitlines()]
    for source, scored in zip(originals[-4:-2], score_rows, strict=True):
        source = {k: v.replace('c032', new_id) if isinstance(v, str) else v for k, v in source.items()}
        scored = {k: v.replace('c032', new_id) if isinstance(v, str) else v for k, v in scored.items()}
        source['split'] = scored['split'] = 'calibration'
        with compiled.open('a') as stream:
            stream.write(json.dumps(source) + '\n')
        with (probs / 'calibration.jsonl').open('a') as stream:
            stream.write(json.dumps(scored) + '\n')
    if valid:
        report = diagnose(compiled, probs, tmp_path / 'out', pooled_heldout=True)
        assert report['risk_changes'] == 31 and report['independent_pooled_heldout_changes'] == 32
    else:
        with pytest.raises(ValueError, match='unapproved original'):
            diagnose(compiled, probs, tmp_path / 'out')
