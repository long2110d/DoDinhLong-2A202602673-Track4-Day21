from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_submission_check():
    result = subprocess.run(
        [sys.executable, 'tools/check_submission.py'], cwd=ROOT,
        capture_output=True, encoding='utf-8',
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'SẴN SÀNG NỘP' in result.stdout


def test_report_images_exist():
    report = ROOT / 'report' / 'REPORT.md'
    text = report.read_text(encoding='utf-8')
    images = re.findall(r'!\[[^\]]*\]\((\.\./results/[^)]+)\)', text)
    assert images, 'The report must reference result images'
    for image in images:
        assert (report.parent / image).is_file(), image
