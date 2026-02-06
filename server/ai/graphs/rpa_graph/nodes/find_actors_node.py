import re
from typing import List
from ai.graphs.rpa_graph.state import ActorItem, GraphState

SYSTEM_ACTOR_KEYWORDS = (
    "system",
    "software",
    "application",
    "platform",
    "service",
    "backend",
    "server",
)

SAFE_SYSTEM_ACTORS = {
    "system administrator",
    "system admin",
    "systems analyst",
    "system analyst",
}


def _is_system_actor(actor_name: str) -> bool:
    """Return True if the actor is system-related and should be filtered out."""
    name_lower = actor_name.lower().strip()
    if name_lower in SAFE_SYSTEM_ACTORS:
        return False
    return any(kw in name_lower for kw in SYSTEM_ACTOR_KEYWORDS)


def _find_actors_regex(sentences: List[str]) -> List[ActorItem]:
    """Extract actors from user stories using regex pattern 'As a/an/the [actor]'."""
    pattern = r"As\s+(?:a|an|the)\s+([^,]+)"
    actor_occurrences = {}

    for i, sent in enumerate(sentences):
        match = re.search(pattern, sent, re.IGNORECASE)
        if match:
            actor = match.group(1).strip().lower()
            if _is_system_actor(actor):
                continue
            if actor not in actor_occurrences:
                actor_occurrences[actor] = []
            actor_occurrences[actor].append(i)

    return [
        ActorItem(actor=actor, sentence_idx=sent_indices)
        for actor, sent_indices in actor_occurrences.items()
    ]


def find_actors_node(state: GraphState):
    """Extract actors using regex pattern from user stories.
    System-related actors (e.g. 'As a system', 'system operator') are filtered out.
    """

    sentences = state.get("sentences") or []
    raw_actors = _find_actors_regex(sentences)

    # DEBUG: in danh sách actor raw
    print("\n==== find_actors_node ====")
    for a in raw_actors:
        print(f"actor={a.actor}, sentence_idx={a.sentence_idx}")

    return {"raw_actors": raw_actors}
