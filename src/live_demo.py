from __future__ import annotations

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


def main() -> None:
    config = load_config()
    baseline = BaselineAgent(config)
    advanced = AdvancedAgent(config)
    if baseline.langchain_agent is None or advanced.langchain_agent is None:
        raise SystemExit(
            "Live model is not configured. Copy .env.example to .env, add your API key, "
            "then install requirements.txt."
        )

    user_id = "live-demo-user"
    fact = "Mình tên là An và thích câu trả lời ngắn gọn."
    question = "Sang phiên mới, bạn nhớ tên mình là gì?"

    print("[1/4] Baseline learning call...", flush=True)
    baseline.reply(user_id, "learn", fact)
    print("[2/4] Advanced learning call...", flush=True)
    advanced.reply(user_id, "learn", fact)
    print("[3/4] Baseline recall call...", flush=True)
    baseline_answer = baseline.reply(user_id, "recall", question)["response"]
    print("[4/4] Advanced recall call...", flush=True)
    advanced_answer = advanced.reply(user_id, "recall", question)["response"]

    print("Baseline (new thread):", baseline_answer)
    print("Advanced (new thread):", advanced_answer)
    print("Advanced memory:", advanced.profile_store.path_for(user_id))


if __name__ == "__main__":
    main()
