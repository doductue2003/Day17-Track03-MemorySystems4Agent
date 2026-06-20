from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Student TODO: read JSON conversations from disk."""

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def recall_points(answer: str, expected: list[str]) -> float:
    """Student TODO: return 0 / 0.5 / 1 depending on how many expected facts appear."""

    if not expected:
        return 1.0
    normalized = answer.casefold()
    found = sum(item.casefold() in normalized for item in expected)
    return found / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Student TODO: add a lightweight quality score for offline mode."""

    recall = recall_points(answer, expected)
    if not answer.strip():
        return 0.0
    clarity = 1.0 if len(answer) <= 800 else 0.5
    return round(0.8 * recall + 0.2 * clarity, 3)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Student TODO: evaluate one agent over many conversations.

    Pseudocode:
    1. Feed all turns to the agent.
    2. Track `agent tokens only`.
    3. Track `prompt tokens processed`.
    4. Ask recall questions in a fresh thread.
    5. Compute average recall and quality.
    6. Record memory file growth and compaction count.
    """

    user_ids = {str(item["user_id"]) for item in conversations}
    before_size = sum(getattr(agent, "memory_file_size", lambda _user: 0)(user) for user in user_ids)
    thread_ids: list[str] = []
    recalls: list[float] = []
    qualities: list[float] = []

    for conversation in conversations:
        user_id = str(conversation["user_id"])
        thread_id = f"{conversation['id']}-turns"
        thread_ids.append(thread_id)
        for turn in conversation["turns"]:
            agent.reply(user_id, thread_id, str(turn))

        for index, recall in enumerate(conversation.get("recall_questions", []), start=1):
            recall_thread = f"{conversation['id']}-recall-{index}"
            thread_ids.append(recall_thread)
            result = agent.reply(user_id, recall_thread, str(recall["question"]))
            answer = str(result["response"])
            expected = [str(value) for value in recall.get("expected_contains", [])]
            recalls.append(recall_points(answer, expected))
            qualities.append(heuristic_quality(answer, expected))

    after_size = sum(getattr(agent, "memory_file_size", lambda _user: 0)(user) for user in user_ids)
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=sum(agent.token_usage(thread) for thread in thread_ids),
        prompt_tokens_processed=sum(agent.prompt_token_usage(thread) for thread in thread_ids),
        recall_score=round(sum(recalls) / len(recalls), 3) if recalls else 0.0,
        response_quality=round(sum(qualities) / len(qualities), 3) if qualities else 0.0,
        memory_growth_bytes=max(0, after_size - before_size),
        compactions=sum(agent.compaction_count(thread) for thread in thread_ids),
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Student TODO: print a markdown table or tabulated output."""

    headers = (
        "Agent", "Agent tokens only", "Prompt tokens processed", "Cross-session recall",
        "Response quality", "Memory growth (bytes)", "Compactions",
    )
    values = [headers]
    values.extend(
        (
            row.agent_name,
            str(row.agent_tokens_only),
            str(row.prompt_tokens_processed),
            f"{row.recall_score:.3f}",
            f"{row.response_quality:.3f}",
            str(row.memory_growth_bytes),
            str(row.compactions),
        )
        for row in rows
    )
    widths = [max(len(item[index]) for item in values) for index in range(len(headers))]
    render = lambda row: "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |"
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([render(values[0]), separator, *(render(row) for row in values[1:])])


def main() -> None:
    """Student TODO: run both benchmark suites.

    Required benchmark sections:
    - Standard benchmark from `data/conversations.json`
    - Long-context stress benchmark from `data/advanced_long_context.json`

    Compare:
    - Baseline
    - Advanced

    Keep the same output columns as the solved lab:
    - Agent tokens only
    - Prompt tokens processed
    - Cross-session recall
    - Response quality
    - Memory growth (bytes)
    - Compactions
    """

    config = load_config(Path(__file__).resolve().parent.parent)
    suites = (
        ("Standard Benchmark", config.data_dir / "conversations.json"),
        ("Long-Context Stress Benchmark", config.data_dir / "advanced_long_context.json"),
    )

    with TemporaryDirectory(prefix="memory-lab-") as temp_dir:
        for suite_index, (title, path) in enumerate(suites):
            conversations = load_conversations(path)
            suite_state = Path(temp_dir) / str(suite_index)
            suite_config = replace(config, state_dir=suite_state)
            baseline = BaselineAgent(suite_config, force_offline=True)
            advanced = AdvancedAgent(suite_config, force_offline=True)
            rows = [
                run_agent_benchmark("Baseline", baseline, conversations, suite_config),
                run_agent_benchmark("Advanced", advanced, conversations, suite_config),
            ]
            print(f"\n## {title}\n")
            print(format_rows(rows))


if __name__ == "__main__":
    main()
