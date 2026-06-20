from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Student TODO: implement Agent A.

    Requirements:
    - Within-session memory only
    - No persistent `User.md`
    - Should forget long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        # TODO: optionally initialize a real LangChain/LangGraph agent when dependencies exist.
        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: return the agent response and token accounting.

        Pseudocode:
        - If a live agent exists, call the live path.
        - Otherwise use a deterministic offline path.
        """

        if self.langchain_agent is None:
            return self._reply_offline(thread_id, message)

        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        session.prompt_tokens_processed += estimate_tokens(
            " ".join(item["content"] for item in session.messages)
        )
        result = self.langchain_agent.invoke(session.messages)
        answer = result.content if hasattr(result, "content") else str(result)
        session.messages.append({"role": "assistant", "content": answer})
        output_tokens = estimate_tokens(answer)
        session.token_usage += output_tokens
        return {"response": answer, "tokens": output_tokens}

    def token_usage(self, thread_id: str) -> int:
        # TODO: return cumulative agent token count for one thread.
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        # TODO: estimate how much prompt context this baseline kept processing.
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: implement a simple offline behavior.

        Suggested behavior:
        - Store the new user message in the session
        - Generate a short deterministic reply
        - Update token counts
        - Never remember facts across different thread ids
        """

        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        session.prompt_tokens_processed += estimate_tokens(
            " ".join(item["content"] for item in session.messages)
        )

        facts: dict[str, str] = {}
        for item in session.messages:
            if item["role"] == "user":
                facts.update(extract_profile_updates(item["content"]))

        if "?" in message:
            if facts:
                answer = "Thông tin trong phiên này: " + "; ".join(
                    f"{key}: {value}" for key, value in sorted(facts.items())
                )
            else:
                answer = "Mình không có thông tin đó trong phiên hiện tại."
        else:
            answer = "Mình đã ghi nhận thông tin trong phiên hiện tại."

        session.messages.append({"role": "assistant", "content": answer})
        output_tokens = estimate_tokens(answer)
        session.token_usage += output_tokens
        return {"response": answer, "tokens": output_tokens}

    def _maybe_build_langchain_agent(self):
        """Student TODO: optionally wire `create_agent` + `InMemorySaver` here.

        Use `build_chat_model(self.config.model)` so the baseline can run with any supported provider.
        """

        if self.force_offline:
            return None
        config = self.config.model
        if config.provider != "ollama" and not config.api_key:
            return None
        try:
            return build_chat_model(config)
        except (ImportError, ValueError):
            return None
