"""Dataset-relative health alerts with measured reasons and triage actions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

INVALID_ERROR_RATIO = 0.01
LOW_POINTS_FACTOR = 0.5
HIGH_POINTS_FACTOR = 1.5
SECTOR_GAP_DEG = 30.0
EMPTY_AZ_BIN_EXCESS = 2
RANGE_LOW_FACTOR = 0.5
RANGE_HIGH_FACTOR = 1.5
RANGE_MAX_M = 200.0
NEAR_RATIO_FLOOR = 0.05
NEAR_MEDIAN_FACTOR = 3.0
INTENSITY_ZERO_MAX = 0.9
INTENSITY_SAT_MAX = 0.2
INTENSITY_MAD_FLOOR = 0.01
MAD_SIGMA_FACTOR = 1.4826
INTENSITY_SIGMAS = 3.0
Z_SHIFT_M = 0.3
TIME_GAP_RATIO = 1.5
SECTOR_MIN_DENSITY_RATIO = 0.35


@dataclass(frozen=True)
class Rule:
    code: str
    severity: str
    description: str
    threshold_text: str
    check: Callable[[pd.Series, dict], str | None]


def _compare(row, key, op, threshold, label=None):
    value = row.get(key, float('nan'))
    hit = value > threshold if op == '>' else value < threshold if op == '<' else value >= threshold if op == '>=' else value <= threshold
    return f'{key}={value:g} {op} {label or format(threshold, "g")}' if hit else None


def _relative(row, ctx, key, op, factor):
    median = ctx['medians'].get(key, float('nan'))
    return _compare(row, key, op, factor * median, f'{factor:g}*median({median:g})')


def _sector(row, ctx):
    # A persistent blind sector is the dataset FOV, not a per-frame dropout.
    median_gap = ctx['medians'].get('az_max_sector_gap_deg', float('nan'))
    absolute = _compare(row, 'az_max_sector_gap_deg', '>=', SECTOR_GAP_DEG) if median_gap < SECTOR_GAP_DEG else None
    median = ctx['medians'].get('empty_az_bins', float('nan'))
    return absolute or _compare(row, 'empty_az_bins', '>', median + EMPTY_AZ_BIN_EXCESS, f'median({median:g})+{EMPTY_AZ_BIN_EXCESS}')


def _intensity(row, ctx):
    reason = (_compare(row, 'intensity_zero_ratio', '>', INTENSITY_ZERO_MAX)
              or _compare(row, 'intensity_sat_ratio', '>', INTENSITY_SAT_MAX))
    median = ctx['medians'].get('intensity_mean', float('nan'))
    value = row.get('intensity_mean', float('nan'))
    limit = INTENSITY_SIGMAS * ctx['intensity_sigma']
    if not reason and abs(value - median) > limit:
        reason = f'intensity_mean={value:g}; abs(value-median({median:g})) > {limit:g} (3*MAD-based sigma)'
    return reason


def _z_shift(row, ctx):
    value = row.get('z_p50', float('nan'))
    median = ctx['medians'].get('z_p50', float('nan'))
    return f'z_p50={value:g}; abs(value-median({median:g})) > {Z_SHIFT_M:g} m' if abs(value - median) > Z_SHIFT_M else None


def _duplicate(row, ctx):
    hashed = row.get('points_hash')
    if pd.isna(hashed):
        return None
    if hashed in ctx['hashes']:
        return f'points_hash={hashed} equals earlier frame {ctx["hashes"][hashed]}'
    ctx['hashes'][hashed] = row['frame_id']
    return None


RULES = [
    Rule('INVALID_POINTS', 'error if > 1%, otherwise warn', 'Non-finite point values', 'invalid_ratio > 0; error above 0.01', lambda r, c: _compare(r, 'invalid_ratio', '>', 0)),
    Rule('LOW_POINTS', 'error', 'Too few points', 'n_points < 0.5*median', lambda r, c: _relative(r, c, 'n_points', '<', LOW_POINTS_FACTOR)),
    Rule('HIGH_POINTS', 'warn', 'Too many points', 'n_points > 1.5*median', lambda r, c: _relative(r, c, 'n_points', '>', HIGH_POINTS_FACTOR)),
    Rule('SECTOR_GAP', 'error', 'Missing angular coverage', 'gap >= 30 deg for full-coverage datasets, or empty bins > median+2; relative clause handles limited FOV', _sector),
    Rule('RANGE_ANOMALY', 'warn', 'Unusual point range', 'p95 outside [0.5, 1.5]*median or max > 200 m', lambda r, c: _relative(r, c, 'range_p95', '<', RANGE_LOW_FACTOR) or _relative(r, c, 'range_p95', '>', RANGE_HIGH_FACTOR) or _compare(r, 'range_max', '>', RANGE_MAX_M)),
    Rule('NEAR_BLOCKAGE', 'warn', 'Excess near-sensor returns', 'ratio_within_2m > max(0.05, 3*median)', lambda r, c: _compare(r, 'ratio_within_2m', '>', max(NEAR_RATIO_FLOOR, NEAR_MEDIAN_FACTOR*c['medians'].get('ratio_within_2m', 0)))),
    Rule('INTENSITY_ANOMALY', 'warn', 'Unusual reflectance', 'zero > 0.9, saturation > 0.2, or mean deviation > 3*1.4826*max(MAD, 0.01)', _intensity),
    Rule('Z_SHIFT', 'warn', 'Ground or mounting height drift', 'abs(z_p50-median) > 0.3 m', _z_shift),
    Rule('DUPLICATE_FRAME', 'error', 'Repeated point cloud', 'hash equals an earlier frame', _duplicate),
    Rule('TIME_GAP', 'warn', 'Possible dropped frame', 'dt_ratio > 1.5', lambda r, c: _compare(r, 'dt_ratio', '>', TIME_GAP_RATIO)),
    Rule('TIME_NONMONOTONIC', 'error', 'Non-increasing timestamp', 'dt_s <= 0', lambda r, c: _compare(r, 'dt_s', '<=', 0)),
    Rule('SECTOR_SPARSE', 'warn', 'Partial angular dropout; a sector has less than 35% of mean return density', 'az_min_bin_ratio < 0.35 for full-coverage datasets', lambda r, c: _compare(r, 'az_min_bin_ratio', '<', SECTOR_MIN_DENSITY_RATIO) if c['medians'].get('empty_az_bins', 0) == 0 else None),
]


def evaluate(stats_df: pd.DataFrame, gaps_df: pd.DataFrame | None = None,
             reference_stats: pd.DataFrame | None = None) -> pd.DataFrame:
    """Evaluate sorted frame IDs without modifying the input tables."""
    reference = stats_df if reference_stats is None else reference_stats
    medians = reference.median(numeric_only=True).to_dict()
    means = reference.get('intensity_mean', pd.Series(dtype=float))
    mad = (means - means.median()).abs().median()
    ctx = {'medians': medians, 'hashes': {},
           'intensity_sigma': MAD_SIGMA_FACTOR * max(INTENSITY_MAD_FLOOR, mad)}
    rows = stats_df.drop(columns=['dt_s', 'dt_ratio'], errors='ignore').copy()
    if gaps_df is not None:
        rows = rows.merge(gaps_df[['frame_id', 'dt_s', 'dt_ratio']], on='frame_id', how='left', validate='one_to_one')
    output = []
    for _, row in rows.sort_values('frame_id', kind='stable').iterrows():
        flags, reasons, severities = [], [], []
        for rule in RULES:
            reason = rule.check(row, ctx)
            if reason is not None:
                flags.append(rule.code)
                reasons.append(f'{rule.code}: {reason}')
                severity = rule.severity
                if rule.code == 'INVALID_POINTS':
                    severity = 'error' if row['invalid_ratio'] > INVALID_ERROR_RATIO else 'warn'
                severities.append(severity)
        status = 'ERROR' if 'error' in severities else 'WARN' if flags else 'OK'
        output.append([row['frame_id'], status, len(flags), ';'.join(flags), ' | '.join(reasons),
                       {'OK': 'use', 'WARN': 'review/label', 'ERROR': 'drop'}[status]])
    return pd.DataFrame(output, columns=['frame_id', 'status', 'n_flags', 'flags', 'reasons', 'action'])


def rules_table() -> pd.DataFrame:
    return pd.DataFrame([{key: getattr(rule, key) for key in ('code', 'severity', 'description', 'threshold_text')} for rule in RULES])
