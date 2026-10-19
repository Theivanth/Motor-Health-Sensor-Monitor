#!/usr/bin/env python3
"""Standalone synthetic motor sensor streamer. Python standard library only.

Outputs CSV compatible with the Machine Sensor Failure Predictor monitor.py.
Each tick represents ONE SIMULATED MINUTE, regardless of --interval speed.
For training/demo only. Not real equipment telemetry.
"""
import argparse
import csv
import json
import math
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

FIELDS = ['timestamp', 'machine_id', 'temperature_c', 'vibration_mm_s',
          'current_a', 'rpm', 'failure_event']


def reading(tick, machine_idx, seed, failure_every, forced_failure_tick=None):
    rng = random.Random(seed + 1009 * machine_idx + tick)
    if forced_failure_tick is not None:
        # Build a visible 85-minute degradation ramp leading to one scheduled
        # failure event, while leaving other motors on the periodic schedule.
        until_failure = max(0, forced_failure_tick - tick)
        failure = int(tick == forced_failure_tick)
        degradation = max(0.0, 1.0 - until_failure / 85.0) ** 1.5
    else:
        # Offset failures between machines so the demonstration is less synchronized.
        offset = (machine_idx * 47) % max(1, failure_every // 3)
        until_failure = (failure_every - ((tick + offset) % failure_every)) % failure_every
        if tick + offset == 0:
            until_failure = failure_every
        failure = int(tick + offset > 0 and (tick + offset) % failure_every == 0)
        degradation = max(0.0, 1.0 - until_failure / 85.0) ** 1.5
    phase = tick / 50.0
    return {
        'temperature_c': round(49 + 1.8*math.sin(phase) + 16*degradation + rng.gauss(0, .8), 3),
        'vibration_mm_s': round(max(.1, 1.4 + .15*math.sin(phase*1.2) + 5.2*degradation + rng.gauss(0, .22)), 3),
        'current_a': round(10.1 + .35*math.sin(phase*.7) + 3.8*degradation + rng.gauss(0, .18), 3),
        'rpm': round(1470 + 10*math.sin(phase) - 125*degradation + rng.gauss(0, 5), 3),
        'failure_event': failure,
    }


def main():
    parser = argparse.ArgumentParser(description='Standalone synthetic motor sensor streamer (no pip install needed)')
    parser.add_argument('--output', default='data/live_stream.csv', help='CSV output for monitor.py')
    parser.add_argument('--interval', type=float, default=.2, help='Wall-clock seconds per simulated minute (0=fast)')
    parser.add_argument('--count', type=int, default=600, help='Number of ticks; 0=until Ctrl+C')
    parser.add_argument('--machines', type=int, default=2)
    parser.add_argument('--failure-every', type=int, default=420, help='Simulated minutes between failures per machine')
    parser.add_argument('--failure-machine', default='MOTOR_002', help='Motor to give an early, controlled failure (empty disables)')
    parser.add_argument('--failure-after', type=int, default=120, help='Ticks before the selected motor fails; degradation begins 85 ticks earlier')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quiet', action='store_true', help='Suppress per-row JSON output')
    args = parser.parse_args()
    if args.interval < 0 or args.count < 0 or args.machines < 1 or args.failure_every < 100 or args.failure_after < 1:
        parser.error('Require interval>=0, count>=0, machines>=1, failure-every>=100, failure-after>=1')
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    start = datetime.now().replace(second=0, microsecond=0)
    print(f'Synthetic sensor stream -> {dest.resolve()} | Ctrl+C to stop', file=sys.stderr)
    try:
        with dest.open('w', newline='', encoding='utf-8', buffering=1) as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            tick = 0
            while args.count == 0 or tick < args.count:
                timestamp = (start + timedelta(minutes=tick)).isoformat()
                for machine_idx in range(args.machines):
                    machine_id = f'MOTOR_{machine_idx+1:03d}'
                    forced_tick = args.failure_after if args.failure_machine == machine_id else None
                    row = {'timestamp': timestamp, 'machine_id': machine_id,
                           **reading(tick, machine_idx, args.seed, args.failure_every, forced_tick)}
                    writer.writerow(row)
                    if not args.quiet:
                        print(json.dumps(row), flush=True)
                tick += 1
                if args.interval:
                    time.sleep(args.interval)
    except KeyboardInterrupt:
        print('\nStopped by user.', file=sys.stderr)
    print(f'CSV saved: {dest.resolve()}', file=sys.stderr)


if __name__ == '__main__':
    main()
