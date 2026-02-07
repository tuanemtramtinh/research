from typing import Any, List
from ai.graphs.rpa_graph.state import UseCase

# Diagram node/link types for frontend (GoJS)
GROUP_KEY = -99


def _to_diagram_data(
    use_cases: List[UseCase],
    actors: List[str] | None = None,
    system_name: str = "System",
) -> dict[str, Any]:
    """
    Transform use_cases and actors to diagram format for frontend.
    Returns { nodes, links } compatible with DiagramWrapper (NodeData[], LinkData[]).
    """
    # Collect unique actors from use_cases if not provided
    if actors is None:
        actors = []
        for uc in use_cases:
            for a in uc.participating_actors or []:
                if a and a.strip() and a not in actors:
                    actors.append(a.strip())

    # Build name -> key maps (case-insensitive for usecase matching)
    actor_to_key: dict[str, int] = {a: i + 1 for i, a in enumerate(actors)}
    usecase_to_key: dict[str, int] = {}
    for i, uc in enumerate(use_cases):
        usecase_to_key[uc.name.lower()] = len(actors) + i + 1

    nodes: List[dict[str, Any]] = []
    links: List[dict[str, Any]] = []

    # 1. Group node (system boundary)
    nodes.append(
        {
            "key": GROUP_KEY,
            "label": system_name,
            "isGroup": True,
        }
    )

    # 2. Actor nodes
    for actor, key in actor_to_key.items():
        nodes.append(
            {
                "key": key,
                "category": "Actor",
                "label": actor,
            }
        )

    # 3. Usecase nodes (inside group)
    for uc in use_cases:
        key = usecase_to_key.get(uc.name.lower())
        if key is not None:
            nodes.append(
                {
                    "key": key,
                    "category": "Usecase",
                    "label": uc.name,
                    "group": GROUP_KEY,
                }
            )

    # 4. Links: actor -> usecase
    link_key = -1
    for uc in use_cases:
        uc_key = usecase_to_key.get(uc.name.lower())
        if uc_key is None:
            continue
        for actor in uc.participating_actors or []:
            actor_key = actor_to_key.get(actor)
            if actor_key is not None:
                links.append({"key": link_key, "from": actor_key, "to": uc_key})
                link_key -= 1

    # 5. Links: usecase -> usecase (include/extend)
    for uc in use_cases:
        src_key = usecase_to_key.get(uc.name.lower())
        if src_key is None:
            continue
        for rel in uc.relationships or []:
            tgt_key = usecase_to_key.get((rel.target_use_case or "").lower())
            if tgt_key is not None:
                text = f"<<{rel.type}>>" if rel.type else "<<include>>"
                links.append(
                    {
                        "key": link_key,
                        "from": src_key,
                        "to": tgt_key,
                        "text": text,
                    }
                )
                link_key -= 1

    return {"nodes": nodes, "links": links}
