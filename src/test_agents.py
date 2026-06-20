from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import UserProfileStore
from memory_store import extract_profile_updates


def make_config(tmp_path: Path):
    """Student TODO: build an isolated config for tests."""

    # Hint:
    # - point `state_dir` into tmp_path
    # - reduce compact threshold so compaction happens quickly in tests
    config = load_config(Path(__file__).resolve().parent.parent)
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return replace(
        config,
        state_dir=state_dir,
        compact_threshold_tokens=80,
        compact_keep_messages=2,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Student TODO: verify `User.md` can be created, updated, and edited."""

    store = UserProfileStore(tmp_path / "profiles")
    path = store.write_text("dung/ct", "# User Profile\n\n- name: DũngCT")
    assert path.exists()
    assert "DũngCT" in store.read_text("dung/ct")
    assert store.edit_text("dung/ct", "DũngCT", "DũngCT Updated")
    assert "DũngCT Updated" in store.read_text("dung/ct")
    assert store.file_size("dung/ct") > 0


def test_compact_trigger(tmp_path: Path) -> None:
    """Student TODO: verify long threads trigger compaction."""

    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    for index in range(8):
        agent.reply("user", "long-thread", f"Turn {index}: " + "nội dung dài " * 20)
    assert agent.compaction_count("long-thread") > 0
    assert len(agent.compact_memory.context("long-thread")["messages"]) <= 2


def test_cross_session_recall(tmp_path: Path) -> None:
    """Student TODO: verify advanced remembers across sessions and baseline does not."""

    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    fact = "Mình tên là DũngCT và đồ uống yêu thích là cà phê sữa đá."
    baseline.reply("dungct", "session-1", fact)
    advanced.reply("dungct", "session-1", fact)

    question = "Mình tên gì và đồ uống yêu thích là gì?"
    baseline_answer = baseline.reply("dungct", "session-2", question)["response"]
    advanced_answer = advanced.reply("dungct", "session-2", question)["response"]
    assert "DũngCT" not in baseline_answer
    assert "DũngCT" in advanced_answer
    assert "cà phê sữa đá" in advanced_answer


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Student TODO: compare prompt load of baseline vs advanced on a long thread."""

    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    for index in range(20):
        message = f"Turn {index}: " + "context benchmark rất dài " * 18
        baseline.reply("user", "stress", message)
        advanced.reply("user", "stress", message)
    assert advanced.compaction_count("stress") > 0
    assert advanced.prompt_token_usage("stress") < baseline.prompt_token_usage("stress")


def test_questions_are_not_persisted_as_facts() -> None:
    assert extract_profile_updates("Mình tên gì và đang ở đâu?") == {}
    assert extract_profile_updates("Nhắc lại giúp mình nuôi con gì.") == {}


def test_new_correction_replaces_old_profile_value(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    agent.reply("user", "one", "Mình ở Đà Nẵng.")
    agent.reply("user", "two", "Giờ mình đang ở Huế chứ không còn ở Đà Nẵng.")
    facts = agent.profile_store.facts("user")
    assert facts["location"] == "Huế"
