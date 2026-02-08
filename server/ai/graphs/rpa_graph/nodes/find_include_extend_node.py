from ai.graphs.rpa_graph.state import GraphState, UseCaseRelationshipResponse


def find_include_extend_node(state: GraphState):
    """
    Tìm tất cả quan hệ «include» và «extend» giữa các Use Case (một bước LLM).
    Input: use_cases từ name_usecases_node (đã có name, description, user_stories).
    Output: include_extend_relationships (list dict) cho merge_relationships_node.
    """
    llm = state.get("llm")
    use_cases = state.get("use_cases") or []

    if not use_cases or not llm:
        return {}

    # Build context: name, description, goals/actions từ user_stories
    lines = []
    uc_names = []
    for uc in use_cases:
        uc_names.append(uc.name)
        parts = [f"- {uc.name}"]
        if uc.description:
            parts.append(f"  Description: {uc.description}")
        if uc.user_stories:
            goals = [s.action for s in uc.user_stories[:3]]
            parts.append(f"  Goals/actions: {', '.join(goals)}")
        lines.append("\n".join(parts))

    usecases_text = "\n\n".join(lines)
    uc_names_lower = [n.lower() for n in uc_names]

    system_prompt = """You are a UML Use Case expert. Identify «include» and «extend» relationships between the given use cases.

DEFINITIONS:
1. «include» (mandatory): Use case A ALWAYS requires use case B to complete.
   - Example: "checkout order" includes "validate cart"
   - Indicators: "must", "requires", "needs to", "first", "then"

2. «extend» (optional): Use case A MAY add behavior to use case B under some conditions.
   - Example: "apply discount" extends "checkout order"
   - Indicators: "optionally", "can also", "if", "when", "may"

RULES:
- Only output relationships clearly implied by descriptions or goals/actions. Do NOT invent.
- source_use_case and target_use_case MUST be exactly one of the use case names from the list (same spelling, use lowercase).
- A use case cannot include/extend itself.
- Return an empty list if there are no clear relationships.
- Output use case names in lowercase."""

    human_prompt = f"""Use cases (from clustered goals, already named):

{usecases_text}

Use case names (use EXACTLY as source/target, lowercase): {uc_names_lower}

Identify all «include» and «extend» relationships. Return source_use_case, relationship_type ("include" or "extend"), target_use_case, and brief reasoning."""

    structured_llm = llm.with_structured_output(UseCaseRelationshipResponse)
    response = structured_llm.invoke(
        [("system", system_prompt), ("human", human_prompt)]
    )

    # Output list of dicts (format cho merge_relationships_node)
    include_extend_relationships = []
    for rel in response.relationships:
        source = rel.source_use_case.lower().strip()
        target = rel.target_use_case.lower().strip()
        if source in uc_names_lower and target in uc_names_lower and source != target:
            include_extend_relationships.append(
                {
                    "source_use_case": source,
                    "type": rel.relationship_type,
                    "target_use_case": target,
                }
            )

    # print("\n--- FIND INCLUDE/EXTEND ---")
    # for rel in include_extend_relationships:
    #     print(f"  {rel['source_use_case']} --{rel['type']}--> {rel['target_use_case']}")

    return {"include_extend_relationships": include_extend_relationships}
