from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Student TODO: implement Agent B / Advanced Agent.

    Required memory layers:
    1. within-session memory
    2. persistent `User.md`
    3. compact memory for long threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}

        # TODO: optionally initialize a real LangChain/LangGraph agent.
        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: route between offline mode and live mode."""

        if self.langchain_agent is None:
            return self._reply_offline(user_id, thread_id, message)

        updates = extract_profile_updates(message)
        if updates:
            self.profile_store.upsert_facts(user_id, updates)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens

        context = self.compact_memory.context(thread_id)
        system = (
            "Use this persistent user profile when relevant:\n"
            + self.profile_store.read_text(user_id)
            + "\nCompact summary:\n"
            + str(context["summary"])
        )
        model_messages = [{"role": "system", "content": system}, *context["messages"]]
        result = self.langchain_agent.invoke(model_messages)
        answer = result.content if hasattr(result, "content") else str(result)
        self.compact_memory.append(thread_id, "assistant", answer)
        output_tokens = estimate_tokens(answer)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + output_tokens
        return {
            "response": answer,
            "tokens": output_tokens,
            "memory_path": str(self.profile_store.path_for(user_id)),
        }

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: implement the deterministic advanced path.

        Pseudocode:
        1. Extract stable profile facts from the incoming message.
        2. Persist those facts into `User.md`.
        3. Append the message into compact memory.
        4. Estimate prompt-context load from `User.md` + summary + recent messages.
        5. Generate a response that can answer long-term recall questions.
        6. Append the assistant reply and update token counters.
        """

        updates = extract_profile_updates(message)
        if updates:
            self.profile_store.upsert_facts(user_id, updates)

        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens

        answer = self._offline_response(user_id, thread_id, message)
        self.compact_memory.append(thread_id, "assistant", answer)
        output_tokens = estimate_tokens(answer)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + output_tokens
        return {
            "response": answer,
            "tokens": output_tokens,
            "memory_path": str(self.profile_store.path_for(user_id)),
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Student TODO: estimate the context carried into one turn.

        Hint:
        - Include `User.md`
        - Include compact summary text
        - Include recent kept messages
        """

        context = self.compact_memory.context(thread_id)
        recent = " ".join(item["content"] for item in context["messages"])
        carried = " ".join(
            (self.profile_store.read_text(user_id), str(context["summary"]), recent)
        )
        return estimate_tokens(carried)

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Student TODO: return a deterministic answer using persisted memory.

        Make sure the advanced agent can answer questions like:
        - "Mình tên gì?"
        - "Hiện tại mình làm nghề gì?"
        - "Nhắc lại style trả lời mình thích"
        - questions in the long stress dataset
        """

        facts = self.profile_store.facts(user_id)
        lower = message.casefold()
        recall_markers = ("nhắc lại", "tóm tắt", "là gì", "tên gì", "đâu mới", "là ai")
        is_recall = "?" in message or any(marker in lower for marker in recall_markers)
        if not is_recall:
            return "Mình đã cập nhật memory bền vững và ngữ cảnh của phiên này."
        if not facts:
            return "Mình chưa có đủ thông tin bền vững để trả lời."
        labels = {
            "name": "tên",
            "location": "nơi ở hiện tại",
            "profession": "nghề nghiệp hiện tại",
            "favorite_drink": "đồ uống yêu thích",
            "favorite_food": "món ăn yêu thích",
            "pet": "thú cưng",
            "interests": "mối quan tâm",
            "response_style": "style trả lời",
        }
        return "Thông tin mình nhớ: " + "; ".join(
            f"{labels.get(key, key)}: {value}" for key, value in sorted(facts.items())
        )

    def _maybe_build_langchain_agent(self):
        """Student TODO: wire a live agent with tools and compact middleware.

        High-level design:
        - `build_chat_model(self.config.model)` for the selected provider
        - `InMemorySaver` for short-term thread state
        - tool to read `User.md`
        - tool to write/edit `User.md`
        - dynamic prompt that injects profile memory
        - summarization middleware for long threads
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
