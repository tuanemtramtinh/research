import math
from typing import List
from ai.graphs.rpa_graph.state import GraphState, RefineClusteringResponse


def refine_clustering_node(state: GraphState):
    """
    Refine K-Means clusters using LLM: split mixed clusters, merge similar ones,
    move misplaced items. Output format is unchanged for name_usecases_node.
    """
    llm = state.get("llm")
    clusters = state.get("user_story_clusters")

    if not clusters or not llm:
        return {}

    # Flatten to list of items with (cluster_id, story_dict) for prompt
    flat_items: List[dict] = []
    for c in clusters:
        for story in c["user_stories"]:
            flat_items.append(
                {
                    "cluster_id": c["cluster_id"],
                    "sentence_idx": story["sentence_idx"],
                    "actor": story["actor"],
                    "goal": story["goal"],
                    "original_sentence": story["original_sentence"],
                }
            )

    clusters_text = ""
    for c in clusters:
        cid = c["cluster_id"]
        stories = c["user_stories"]
        clusters_text += f"\n### Cluster {cid}:\n"
        for s in stories:
            clusters_text += f"  - [{s['actor']}] {s['goal']}\n"
            clusters_text += f'    "{s["original_sentence"]}"\n'

    item_lines = "\n".join(
        f"  - sentence_idx={x['sentence_idx']} | actor={x['actor']} | goal={x['goal']}"
        for x in flat_items
    )

    n_clusters = len(clusters)
    n_items = len(flat_items)
    # Allow at most +20% more clusters than original (ceil). Prefer merging.
    cluster_budget = math.ceil(n_clusters * 1.2) if n_clusters > 0 else 0

    refine_clustering_system_prompt = """You are a senior Software Architect and UML specialist. 
    Your task is to refine clusters of user stories into distinct, valid UML Use Cases.

    CRITICAL PRINCIPLE:
    A Use Case represents a single, discrete goal for an actor. 
    - IDENTITY vs. ADMINISTRATION: 'Logging in' (Identity) is NOT the same goal as 'Managing User Accounts' (Administration).
    - CRUD vs. TRANSACTION: 'Managing Products' (Inventory) is NOT the same as 'Browsing Products' (Shopping).
    
    You must separate actions that belong to different architectural layers or business contexts, even if they involve the same data entity (like 'User')."""

    human_prompt = f"""Current clusters (count={n_clusters}, items={n_items}):
    {clusters_text}

    All items to re-cluster:
    {item_lines}

    STRICT RULES FOR CLUSTERING:
    1. **Goal-Oriented Clustering:** Group by the primary objective. 
       - AUTHENTICATION: Login, Logout, Register, Reset Password belong together.
       - ADMIN/CRUD: Create/Edit/Delete/List [Entity] (e.g., Manage Accounts, Manage Products) MUST be in their own separate clusters.
    2. **Entity vs. Action:** Do NOT group just because they share the word 'User'. 
       - [User] 'Log in' is a Security action.
       - [Admin] 'Delete User' is a Management action. 
       - These MUST be in DIFFERENT clusters.
    3. **Actor Context:** While different actors can share a Use Case (e.g., both Teacher and Student 'View Schedule'), if the action changed context (e.g., Student 'View Course' vs Admin 'Create Course'), they must be split.
    4. **Granularity:** If a cluster has more than 5-6 items that seem slightly different, split them into more specific Use Cases (e.g., 'View Reports' vs 'Export Reports').
    5. **Cluster Budget (important):** Total clusters after refinement MUST NOT exceed {cluster_budget} (≈ +20% over the current {n_clusters}). Prefer MERGING similar clusters; only SPLIT when goals are clearly unrelated.
    6. **Avoid tiny splits:** Do not create a new cluster with a single item unless it is semantically incompatible with every other item.

    Output exactly one entry per line, preserving the original strings and providing a logical 'target_cluster_id'."""

    structured_llm = llm.with_structured_output(RefineClusteringResponse)
    response = structured_llm.invoke(
        [("system", refine_clustering_system_prompt), ("human", human_prompt)]
    )

    # Build lookup (sentence_idx, actor, goal) -> target_cluster_id (RefinedClusterItem.usecase = goal text)
    def _item_key(i):
        return (i.sentence_idx, i.actor, i.usecase)

    assign = {_item_key(i): i.target_cluster_id for i in response.items}

    # Regroup by target_cluster_id, keeping full story dicts; fallback to original cluster_id if missing
    new_clusters: dict = {}
    for c in clusters:
        for story in c["user_stories"]:
            k = (story["sentence_idx"], story["actor"], story["goal"])
            tid = assign.get(k, c["cluster_id"])
            if tid not in new_clusters:
                new_clusters[tid] = []
            new_clusters[tid].append(story)

    clusters_list = [
        {"cluster_id": cid, "user_stories": items}
        for cid, items in sorted(new_clusters.items())
    ]

    # print("\n--- REFINED CLUSTERS (after LLM) ---")
    # for c in clusters_list:
    #     print(f"\nCluster {c['cluster_id']}:")
    #     for s in c["user_stories"]:
    #         print(f"  - [{s['actor']}] {s['goal']}")

    return {"user_story_clusters": clusters_list}
