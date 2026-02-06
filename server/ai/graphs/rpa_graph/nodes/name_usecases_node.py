from typing import List
from ai.graphs.rpa_graph.state import (
    GraphState,
    UseCase,
    UseCaseNamingResponse,
    UserStoryItem,
)


def name_usecases_node(state: GraphState):
    """
    Takes clustered user stories and asks LLM to generate a UseCase name for each cluster.
    Returns a list of UseCase objects.
    """

    llm = state.get("llm")
    clusters = state.get("user_story_clusters")

    if not clusters:
        return {}

    # Build prompt with cluster information (using detailed structure)
    clusters_text = ""
    for cluster in clusters:
        cluster_id = cluster["cluster_id"]
        stories = cluster["user_stories"]  # Contains detailed dicts
        clusters_text += f"\n### Cluster {cluster_id}:\n"

        # Extract unique actors
        actors = list(set(story["actor"] for story in stories))

        clusters_text += f"  Actors: {', '.join(actors)}\n"
        clusters_text += "  Actions:\n"
        for story in stories:
            clusters_text += f"    - {story['goal']}\n"
        clusters_text += "  Sample sentences:\n"
        for story in stories[:3]:  # Show max 3 examples
            clusters_text += f'    - "{story["original_sentence"]}"\n'

    prompt = f"""You are a senior software analyst specializing in Use Case modeling for UML diagrams.

Given the following clusters of related user story actions, generate a precise UseCase name for each cluster.

**Naming Guidelines:**
1. Use verb-noun format (1-3 words preferred):
   - For login/register/password actions → "Authenticate" or "Authentication"
   - For CRUD operations on a resource → "Manage [Resource]" (e.g., "Manage Products")
   - For viewing/reading data → "View [Data]" or "Browse [Data]"
   - For creating reports → "Generate [Report Type]"
   
2. **Common patterns:**
   - login, register, logout, reset password → "Authenticate" 
   - add, edit, delete, update items → "Manage [Items]"
   - search, filter, browse → "Search [Items]" or "Browse [Items]"
   - view reports, download data → "View Reports" or "Export Data"
   - configure settings → "Configure System"

3. **Avoid:**
   - Long names (more than 4 words)
   - Generic names like "Handle User Actions"
   - Names that describe WHO does it instead of WHAT is done

4. If actions are mixed, identify the DOMINANT theme and name accordingly

{clusters_text}

For each cluster, provide:
- cluster_id: The cluster number (as shown above)
- usecase_name: A concise name following the patterns above (1-4 words)
- description: One sentence describing the use case scope
"""

    structured_llm = llm.with_structured_output(UseCaseNamingResponse)
    response = structured_llm.invoke(prompt)

    # Convert to UseCase objects
    cluster_lookup = {c["cluster_id"]: c["user_stories"] for c in clusters}
    use_cases: List[UseCase] = []

    for idx, naming in enumerate(response.usecases):
        stories = cluster_lookup.get(naming.cluster_id, [])

        # Extract unique actors
        unique_actors = list(set(story["actor"] for story in stories))

        # Convert story dicts to UserStoryItem objects (action = goal phrase)
        # Đồng thời loại bỏ duplicate theo (actor, câu gốc) trong cùng Use Case.
        # Nghĩa là nếu cùng actor và cùng original_sentence nhưng có nhiều goal khác nhau,
        # ta chỉ giữ lại một bản ghi để tránh lặp dòng trong output.
        unique_story_items: List[UserStoryItem] = []
        seen_story_keys = set()
        for story in stories:
            key = (story["actor"], story["original_sentence"].strip())
            if key in seen_story_keys:
                continue
            seen_story_keys.add(key)
            unique_story_items.append(
                UserStoryItem(
                    actor=story["actor"],
                    action=story["goal"],
                    original_sentence=story["original_sentence"],
                    sentence_idx=story["sentence_idx"],
                )
            )

        user_story_items = unique_story_items

        # Create UseCase object
        use_case = UseCase(
            id=idx + 1,
            name=naming.usecase_name,
            description=naming.description,
            participating_actors=unique_actors,
            user_stories=user_story_items,
            relationships=[],  # Will be filled later if needed
        )
        use_cases.append(use_case)

    # DEBUG: in toàn bộ use_cases trước khi in summary cuối
    print("\n==== name_usecases_node (use_cases) ====")
    for uc in use_cases:
        print(f"UC-{uc.id}: {uc.name}")
        print(f"  Actors: {uc.participating_actors}")
        for s in uc.user_stories:
            print(f"  - [{s.actor}] {s.action} (sent={s.sentence_idx})")
            print(f'    "{s.original_sentence}"')

    # # Print output
    # print("\n" + "=" * 60)
    # print("GENERATED USE CASES")
    # print("=" * 60)

    # for uc in use_cases:
    #     print(f"\n📌 UC-{uc.id}: [{uc.name}]")
    #     print(f"   Description: {uc.description}")
    #     print(f"   Actors: {', '.join(uc.participating_actors)}")
    #     print(f"   User Stories ({len(uc.user_stories)}):")
    #     for story in uc.user_stories:
    #         print(f"     • [{story.actor}] {story.action}")
    #         print(f'       └─ "{story.original_sentence}"')

    return {"use_cases": use_cases}
