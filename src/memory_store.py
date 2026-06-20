from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re


def estimate_tokens(text: str) -> int:
    """Student TODO: implement a simple token estimator.

    Example idea:
    - Strip whitespace
    - Return 0 for empty text
    - Approximate tokens from character count, e.g. len(text) / 4
    """

    cleaned = text.strip()
    if not cleaned:
        return 0
    # A stable heuristic is more useful for this offline comparison than a
    # provider-specific tokenizer.
    return max(1, (len(cleaned) + 3) // 4)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`.

    Student TODO:
    - Map each user id to one markdown file
    - Support read / write / edit operations
    - Optionally expose helpers like `facts()` or `upsert_fact()`
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", user_id.strip()).strip(".-")
        if not slug:
            raise ValueError("user_id must contain at least one safe character")
        return self.root_dir / slug / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return "# User Profile\n"
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content.rstrip() + "\n", encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        facts: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            match = re.match(r"^- ([a-z_]+):\s*(.+)$", line.strip())
            if match:
                facts[match.group(1)] = match.group(2).strip()
        return facts

    def upsert_facts(self, user_id: str, updates: dict[str, str]) -> Path:
        facts = self.facts(user_id)
        facts.update({key: value.strip() for key, value in updates.items() if value.strip()})
        lines = ["# User Profile", ""]
        lines.extend(f"- {key}: {value}" for key, value in sorted(facts.items()))
        return self.write_text(user_id, "\n".join(lines))


def extract_profile_updates(message: str) -> dict[str, str]:
    """Student TODO: convert raw user text into stable profile facts.

    Example facts you may want to extract:
    - name
    - location
    - profession
    - preferences / response style
    - favorite food / drink

    Pseudocode:
    1. Build a few regex patterns.
    2. Skip obvious question-only turns.
    3. Return only the facts that are confidently present in the message.
    """

    text = " ".join(message.strip().split())
    lower = text.casefold()
    recall_phrases = ("tên gì", "con gì", "là gì", "đâu mới")
    starts_as_recall = bool(
        re.match(r"^(?:sang thread mới rồi,\s*)?(?:nhắc lại|tóm tắt)", lower)
    )
    if not text or "?" in text or starts_as_recall or any(marker in lower for marker in recall_phrases):
        return {}

    updates: dict[str, str] = {}

    def capture(key: str, patterns: list[str]) -> None:
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).strip(" .,;:!?")
                words = set(value.casefold().split())
                if value and not words.intersection({"gì", "nào", "không"}):
                    updates[key] = value
                    return

    capture("name", [r"(?:mình|tôi)\s+tên\s+(?:là\s+)?([^,.;]+)"])

    # Strong current-location statements override older facts. Travel and
    # explicitly negated locations are deliberately ignored.
    capture("location", [
        r"nơi ở hiện tại\s+(?:là|ở)\s+([^,.;]+)",
        r"(?:thực ra|giờ|hiện tại|từ tuần này)\s+(?:mình\s+)?(?:đang\s+)?(?:làm việc\s+)?ở\s+([^,.;]+?)(?:\s+(?:vài|để|chứ|và)\b|$)",
        r"(?:mình|tôi)\s+(?:hiện\s+)?ở\s+([^,.;]+?)(?:\s+và\b|$)",
    ])

    capture("profession", [
        r"nghề nghiệp hiện tại\s+(?:vẫn\s+)?là\s+([^,.;]+)",
        r"giờ\s+(?:mình\s+)?chuyển sang\s+([^,.;]+)",
        r"đang\s+làm\s+([^,.;]+?)(?:\s+cho\b|$)",
        r"(?:mình|tôi)\s+(?:hiện\s+)?(?:đang\s+)?làm\s+([^,.;]+?)(?:\s+cho\b|$)",
        r"nghề nghiệp thì vẫn là\s+([^,.;]+)",
    ])

    capture("favorite_drink", [r"đồ uống yêu thích\s+(?:của mình\s+)?là\s+([^,.;]+)"])
    capture("favorite_food", [r"món ăn yêu thích\s+(?:của mình\s+)?là\s+([^,.;]+)"])
    capture("pet", [r"(?:mình|tôi)\s+nuôi\s+(?:một\s+)?(?:bé\s+)?([^,.;]+)"])

    interests: list[str] = []
    if re.search(r"\bPython\b", text, re.IGNORECASE):
        interests.append("Python")
    if re.search(r"\bAI(?:\s+ứng dụng|\s+agent)?\b", text, re.IGNORECASE):
        interests.append("AI")
    if interests and any(marker in lower for marker in ("mình thích", "quan tâm", "mối quan tâm")):
        updates["interests"] = ", ".join(interests)

    style_parts: list[str] = []
    if "ngắn gọn" in lower or "bullet ngắn" in lower:
        style_parts.append("ngắn gọn")
    if "3 bullet" in lower:
        style_parts.append("3 bullet")
    elif "bullet" in lower:
        style_parts.append("bullet")
    if "ví dụ thực" in lower:
        style_parts.append("có ví dụ thực chiến")
    if "trade-off" in lower:
        style_parts.append("nhấn mạnh trade-off")
    if style_parts and any(marker in lower for marker in ("trả lời", "style", "giải thích")):
        updates["response_style"] = ", ".join(dict.fromkeys(style_parts))
    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Student TODO: create a compact summary of older messages.

    This can be heuristic text concatenation first.
    Later, you can replace it with an LLM-based summary if desired.
    """

    if not messages:
        return ""
    items = []
    for message in messages[-max_items:]:
        role = message.get("role", "unknown")
        content = " ".join(message.get("content", "").split())
        if len(content) > 240:
            content = content[:237].rstrip() + "..."
        items.append(f"{role}: {content}")
    return " | ".join(items)


@dataclass
class CompactMemoryManager:
    """Student TODO: implement compact memory for long threads.

    Goal:
    - Keep recent messages in full
    - When the thread grows too large, move older content into a summary
    - Track how many compactions happened for benchmarking
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        # TODO:
        # 1. create thread state if missing
        # 2. append the new message
        # 3. trigger compaction if needed
        thread = self.state.setdefault(
            thread_id, {"messages": [], "summary": "", "compactions": 0}
        )
        messages = thread["messages"]
        assert isinstance(messages, list)
        messages.append({"role": role, "content": content})

        summary = str(thread["summary"])
        total_text = summary + " " + " ".join(str(item.get("content", "")) for item in messages)
        if estimate_tokens(total_text) <= self.threshold_tokens or len(messages) <= self.keep_messages:
            return

        older = messages[:-self.keep_messages]
        recent = messages[-self.keep_messages:]
        new_summary = summarize_messages(older)
        merged = " | ".join(part for part in (summary, new_summary) if part)
        # Keep summary cost bounded across repeated compactions.
        thread["summary"] = merged[-1200:]
        thread["messages"] = recent
        thread["compactions"] = int(thread["compactions"]) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        thread = self.state.setdefault(
            thread_id, {"messages": [], "summary": "", "compactions": 0}
        )
        return {
            "messages": [dict(item) for item in thread["messages"]],
            "summary": str(thread["summary"]),
            "compactions": int(thread["compactions"]),
        }

    def compaction_count(self, thread_id: str) -> int:
        return int(self.context(thread_id)["compactions"])
