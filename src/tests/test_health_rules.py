from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.health_rules import evaluate, rules_table
from src.health_stats import dataset_stats
from src.health_time import load_timestamps, time_gaps


def clean_stats():
    return pd.DataFrame([dict(frame_id=f'{i:06}', n_points=100, invalid_ratio=0.0,
        range_p95=50, range_max=70, ratio_within_2m=0.0, intensity_mean=0.2,
        intensity_zero_ratio=0.0, intensity_sat_ratio=0.0, z_p50=-1.7,
        empty_az_bins=0, az_max_sector_gap_deg=0, az_min_bin_ratio=0.8,
        points_hash=str(i)) for i in range(6)])


def test_clean_and_low_points():
    stats = clean_stats()
    stats.loc[0, 'n_points'] = 49
    result = evaluate(stats).set_index('frame_id')
    assert result.loc['000000', 'flags'] == 'LOW_POINTS'
    assert result.loc['000000', 'status'] == 'ERROR'
    assert result.loc['000000', 'action'] == 'drop'
    assert 'n_points=49' in result.loc['000000', 'reasons']
    assert result.loc['000001', ['status', 'flags', 'action']].tolist() == ['OK', '', 'use']
    stats.loc[0, 'n_points'] = 50
    assert evaluate(stats).iloc[0].status == 'OK'


def test_duplicates_and_times():
    stats = clean_stats()
    stats.loc[1, 'points_hash'] = '0'
    gaps = pd.DataFrame({'frame_id': stats.frame_id, 'dt_s': [None, .1, .2, -.1, .1, .1],
                         'dt_ratio': [None, 1, 2, -1, 1, 1]})
    result = evaluate(stats, gaps)
    assert result.iloc[0]['flags'] == ''
    assert result.iloc[1]['flags'] == 'DUPLICATE_FRAME'
    assert result.iloc[2]['flags'] == 'TIME_GAP'
    assert result.iloc[2].action == 'review/label'
    assert result.iloc[3]['flags'] == 'TIME_NONMONOTONIC'
    assert result.iloc[3].action == 'drop'
    assert 'TIME_' not in ';'.join(evaluate(stats)['flags'])


@pytest.mark.parametrize('key,value,code,status', [
    ('invalid_ratio', .001, 'INVALID_POINTS', 'WARN'),
    ('invalid_ratio', .02, 'INVALID_POINTS', 'ERROR'),
    ('n_points', 151, 'HIGH_POINTS', 'WARN'),
    ('az_max_sector_gap_deg', 30, 'SECTOR_GAP', 'ERROR'),
    ('empty_az_bins', 3, 'SECTOR_GAP', 'ERROR'),
    ('range_p95', 24, 'RANGE_ANOMALY', 'WARN'),
    ('range_p95', 76, 'RANGE_ANOMALY', 'WARN'),
    ('range_max', 201, 'RANGE_ANOMALY', 'WARN'),
    ('ratio_within_2m', .06, 'NEAR_BLOCKAGE', 'WARN'),
    ('intensity_zero_ratio', .91, 'INTENSITY_ANOMALY', 'WARN'),
    ('intensity_sat_ratio', .21, 'INTENSITY_ANOMALY', 'WARN'),
    ('intensity_mean', .3, 'INTENSITY_ANOMALY', 'WARN'),
    ('z_p50', -1.3, 'Z_SHIFT', 'WARN'),
    ('az_min_bin_ratio', .3, 'SECTOR_SPARSE', 'WARN'),
])
def test_rules(key, value, code, status):
    stats = clean_stats()
    stats.loc[0, key] = value
    result = evaluate(stats).iloc[0]
    assert code in result['flags'].split(';')
    assert result.status == status
    assert f'{key}={value:g}' in result.reasons


def test_limited_fov_and_relative_gap():
    stats = clean_stats()
    stats['empty_az_bins'] = 27
    stats['az_max_sector_gap_deg'] = 270
    stats['az_min_bin_ratio'] = 0
    assert (evaluate(stats).status == 'OK').all()
    stats.loc[0, 'empty_az_bins'] = 30
    assert evaluate(stats).iloc[0]['flags'] == 'SECTOR_GAP'


def test_synthetic_planted_faults():
    # Inspection: all frames have 22-23 invalid points (ratios .000960-.000997).
    # 000003 has dt=.2 s (ratio=2) and sparse sector ratio=.311964,
    # versus .472767-.498338 in other frames; no fully empty azimuth bins.
    root = Path(__file__).resolve().parents[2] / 'data' / 'synthetic'
    stats = dataset_stats(root)
    result = evaluate(stats, time_gaps(load_timestamps(root, stats.frame_id))).set_index('frame_id')
    for fid in stats.frame_id:
        assert 'INVALID_POINTS' in result.loc[fid, 'flags'].split(';')
    assert 'TIME_GAP' in result.loc['000003', 'flags'].split(';')
    assert 'SECTOR_SPARSE' in result.loc['000003', 'flags'].split(';')
    assert (result.status != 'OK').sum() >= 2


def test_kitti_no_false_error_alarms():
    root = Path(__file__).resolve().parents[2] / 'data' / 'kitti_mini'
    assert (evaluate(dataset_stats(root)).status == 'ERROR').mean() < .25


def test_metadata_empty_and_input_preservation():
    stats = clean_stats()
    original = stats.copy(deep=True)
    assert evaluate(stats).equals(evaluate(stats))
    pd.testing.assert_frame_equal(stats, original)
    assert evaluate(stats.iloc[:0]).empty
    table = rules_table()
    assert table.columns.tolist() == ['code', 'severity', 'description', 'threshold_text']
    assert table.code.is_unique

