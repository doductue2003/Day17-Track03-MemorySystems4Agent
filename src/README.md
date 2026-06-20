# Memory Systems Lab

This folder contains the completed baseline and advanced memory agents.

- `BaselineAgent` keeps memory only within one thread.
- `AdvancedAgent` adds persistent `User.md` profiles and compact memory.
- `benchmark.py` runs standard and long-context stress suites offline.
- `live_demo.py` performs a small real-provider cross-session smoke test.
- Providers: `openai`, `custom`, `gemini`, `anthropic`, `ollama`, `openrouter`.

Run the deterministic verification:

```powershell
python -m pytest -q src\test_agents.py
python src\benchmark.py
```

For a live model, copy `.env.example` to `.env`, add the selected provider key, install
`requirements.txt`, and run `python src\live_demo.py`. Do not commit `.env`.
