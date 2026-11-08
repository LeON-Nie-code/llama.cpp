import json
import os
from pathlib import Path

root = Path(os.environ['ROLL_BENCH_RESULTS_DIR'])
rows = json.loads((root / 'model-results.json').read_text())
assert len(rows) == 24
idle = json.loads((root / 'idle-preflight.json').read_text())
assert len(idle) == 10
assert all(not s['processes'] and s['gpu_util'] == '0 %' and int(s['memory_used'].split()[0]) <= 100 for s in idle)
samples = []
observed = 0
for row in rows:
    tag = f"{row['scenario']}-{row['variant']}-{row['trial']}"
    before = json.loads((root / (tag + '-gpu-before.json')).read_text())
    assert not before['processes'] and int(before['memory_used'].split()[0]) <= 100
    monitor = json.loads((root / (tag + '-gpu-monitor.json')).read_text())
    assert not monitor['errors'] and not monitor['foreign_process_detected']
    assert monitor['samples'] and all(not s['foreign_pids'] for s in monitor['samples'])
    assert all(p['pid'] == monitor['benchmark_pid'] for s in monitor['samples'] for p in s['processes'])
    observed += any(p['pid'] == monitor['benchmark_pid'] for s in monitor['samples'] for p in s['processes'])
    samples.extend(monitor['samples'])
assert observed == 24, 'Verify PID visibility in every process.'
report = {'idle_preflight_samples': len(idle), 'benchmark_processes': len(rows),
          'timed_processes': sum(r['trial'] >= 0 for r in rows),
          'runtime_monitor_samples': len(samples), 'other_gpu_process_detected': False,
          'benchmark_pid_observed_in_all_processes': True,
          'sample_interval_seconds': 1,
          'temperature_samples': sorted({s['temperature'] for s in samples}),
          'host_load_1m_range': [min(s['host_load_average'][0] for s in samples), max(s['host_load_average'][0] for s in samples)],
          'start_utc': idle[0]['utc'], 'end_utc': samples[-1]['utc']}
(root / 'isolation-summary.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
