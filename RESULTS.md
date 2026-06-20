# Benchmark Results and Analysis

## Results

| Suite | Agent | Prompt tokens processed | Cross-session recall | Compactions |
| --- | --- | ---: | ---: | ---: |
| Standard | Baseline | 14,069 | 0.000 | 0 |
| Standard | Advanced | 22,877 | 1.000 | 0 |
| Long-context stress | Baseline | 22,046 | 0.000 | 0 |
| Long-context stress | Advanced | 14,376 | 1.000 | 4 |

These figures come from the deterministic offline benchmark and can be reproduced with
`python src/benchmark.py`.

## Analysis

The baseline retains messages only inside one thread. A recall question in a new thread
therefore has no access to earlier facts. The advanced agent writes stable, structured facts
to `User.md`, so it can recall the latest name, location, profession, preferences, and
corrections across sessions.

Persistent memory has overhead. In the standard benchmark, the advanced agent repeatedly
loads the profile in addition to recent messages, so it processes more prompt tokens than the
baseline. Compact memory is not automatically cheaper for short conversations.

In the stress benchmark, the baseline repeatedly carries the full growing history. The
advanced agent compacts older messages four times and keeps only a bounded summary plus recent
messages. This reduces processed prompt tokens by about 35% while preserving full recall.

The main risks are incorrect extraction, stale facts, and unbounded profile growth. This
implementation uses structured fields, replaces an older value when a confident correction
arrives, ignores recall questions as profile updates, sanitizes user paths, and keeps compact
summaries bounded. Production systems should additionally attach timestamps and provenance,
apply memory decay, and require stronger confidence checks for sensitive facts.
