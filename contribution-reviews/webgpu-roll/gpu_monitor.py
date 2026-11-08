"""Read-only GPU isolation checks for local benchmark runs."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import xml.etree.ElementTree as ET


def snapshot(allowed_pid=None):
    gpu = ET.fromstring(subprocess.check_output(['nvidia-smi', '-q', '-x'], timeout=10)).find('gpu')
    assert gpu is not None and gpu.findtext('product_name') == 'NVIDIA L4'
    processes = [{'pid': int(p.findtext('pid')), 'type': p.findtext('type'),
                  'memory': p.findtext('used_memory')}
                 for p in gpu.findall('processes/process_info')]
    return {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'monotonic': time.monotonic(),
            'gpu_util': gpu.findtext('utilization/gpu_util'),
            'memory_used': gpu.findtext('fb_memory_usage/used'),
            'memory_free': gpu.findtext('fb_memory_usage/free'),
            'sm_clock': gpu.findtext('clocks/sm_clock'),
            'temperature': gpu.findtext('temperature/gpu_temp'),
            'processes': processes,
            'foreign_pids': [p['pid'] for p in processes if p['pid'] != allowed_pid],
            'host_load_average': list(os.getloadavg())}


def confirm_idle(output, seconds=10):
    samples = []
    for _ in range(seconds):
        s = snapshot()
        samples.append(s)
        Path(output).write_text(json.dumps(samples, indent=2))
        if s['processes'] or int(s['gpu_util'].split()[0]) != 0 or int(s['memory_used'].split()[0]) > 100:
            raise RuntimeError('GPU is not idle; no benchmark started.')
        time.sleep(1)
    print('L4 idle preflight passed: no GPU processes and 0% utilization for 10 samples.', flush=True)


class Monitor:
    def __init__(self, pid, output):
        self.pid = pid
        self.output = Path(output)
        self.samples = []
        self.errors = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop_event.is_set():
            try:
                self.samples.append(snapshot(self.pid))
            except Exception as e:
                self.errors.append(str(e))
            self.stop_event.wait(1)

    def start(self):
        self.thread.start()

    def finish(self):
        self.stop_event.set()
        self.thread.join()
        self.samples.append(snapshot(self.pid))
        data = {'benchmark_pid': self.pid, 'samples': self.samples, 'errors': self.errors,
                'foreign_process_detected': any(s['foreign_pids'] for s in self.samples)}
        self.output.write_text(json.dumps(data, indent=2))
        if data['foreign_process_detected'] or self.errors:
            raise RuntimeError('GPU isolation check failed; retain diagnostics but reject this measurement.')
        return {'samples': len(self.samples), 'foreign_process_detected': False}
