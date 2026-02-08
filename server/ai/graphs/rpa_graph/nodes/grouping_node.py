import math
from langchain_openai import OpenAIEmbeddings
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from ai.graphs.rpa_graph.state import GraphState
import numpy as np


def _extract_benefit(sentence: str) -> str:
    """Lấy phần benefit (sau 'so that') của user story. Nếu không có thì trả về ''."""
    if not sentence or "so that" not in sentence:
        return ""
    parts = sentence.split("so that", 1)
    return parts[-1].strip() if len(parts) > 1 else ""


def grouping_node(state: GraphState):
    # Skip if already processed (prevent duplicate execution from fan-in)
    if state.get("grouping_done"):
        return {}

    actor_results = state.get("actor_results")

    print(actor_results)

    refined_goals = state.get("refined_goals")

    # Skip if either branch hasn't completed yet
    if not actor_results or not refined_goals:
        return {}

    sentences = state.get("sentences") or []

    input_pairs = []  # For embedding: "actor goal [benefit]"
    input_details = []  # Detailed info for each pair (actor, goal, benefit, ...)
    actors_lookup = {}

    # Build lookup: sentence_idx -> list of actors
    for actor in actor_results:
        for idx in actor.sentence_idx:  # Indices where canonical name appears
            if idx not in actors_lookup:
                actors_lookup[idx] = []
            if actor not in actors_lookup[idx]:  # Avoid duplicates
                actors_lookup[idx].append(actor)

        for alias in actor.aliases:
            for sentence in alias.sentences:  # Indices where alias appears
                if sentence not in actors_lookup:
                    actors_lookup[sentence] = []
                if actor not in actors_lookup[sentence]:  # Avoid duplicates
                    actors_lookup[sentence].append(actor)

    for item in refined_goals:
        goals = item.refined + item.added
        orig_sentence = (
            sentences[item.sentence_idx] if item.sentence_idx < len(sentences) else ""
        )
        benefit = _extract_benefit(orig_sentence)

        for goal in goals:
            # Use item.sentence_idx (int) instead of actor.sentence_idx
            if item.sentence_idx in actors_lookup:
                for actor in actors_lookup[item.sentence_idx]:
                    # String for embedding: actor + goal + benefit (benefit giúp gom cụm theo ngữ nghĩa)
                    embed_text = f"{actor.actor} {goal}"
                    if benefit:
                        embed_text += f" {benefit}"
                    input_pairs.append(embed_text)
                    # Detailed info (gắn benefit cùng actor, goal)
                    input_details.append(
                        {
                            "actor": actor.actor,
                            "goal": goal,
                            "benefit": benefit,
                            "sentence_idx": item.sentence_idx,
                            "original_sentence": orig_sentence,
                        }
                    )

    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

    silhouette_scores = []
    # K = range(2, len(input_pairs))
    # K_min = max(3, math.floor(len(input_pairs) / 10))
    # K_max = math.ceil(len(input_pairs) / 3)

    # K = range(K_min, K_max + 1)
    K = range(3, 11)
    X = np.array(embeddings_model.embed_documents(input_pairs))

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
        print(f"Số cụm k={k}, silhouette={score:.4f}")

    best_k = K[np.argmax(silhouette_scores)]
    print(f"\n=> Số cụm hợp lý nhất dựa trên ngữ nghĩa là: {best_k}")

    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    final_labels = kmeans_final.fit_predict(X)
    clusters = {}
    for i, label in enumerate(final_labels):
        # Ensure we always use plain Python int for cluster ids
        cid = int(label)
        if cid not in clusters:
            clusters[cid] = []
        # Store detailed info instead of just string
        clusters[cid].append(input_details[i])

    # print(f"\n--- CHI TIẾT PHÂN CỤM (K={best_k}) ---")
    # for label, items in sorted(clusters.items()):
    #     print(f"\nCụm {label + 1}:")
    #     for item in items:
    #         print(f"  - [{item['actor']}] {item['goal']}")
    #         if item.get("benefit"):
    #             print(f"    Benefit: {item['benefit']}")
    #         print(f"    Original: {item['original_sentence']}")

    # Convert clusters dict to list format for state.
    # IMPORTANT: keep cluster_id as plain Python int so that
    # LangGraph's checkpointer (msgpack) can serialize it safely.
    clusters_list = [
        {"cluster_id": int(label), "user_stories": items}
        for label, items in sorted(clusters.items())
    ]

    # # DEBUG: in input_details và clusters cuối cùng
    # print("\n==== grouping_node ====")
    # print(">>> input_details:")
    # for d in input_details:
    #     print(f"  - actor={d['actor']}, goal={d['goal']}, sent={d['sentence_idx']}")
    #     print(f"    original={d['original_sentence']}")
    # print("\n>>> clusters_list:")
    # for c in clusters_list:
    #     print(f"Cluster {c['cluster_id']}:")
    #     for s in c["user_stories"]:
    #         print(f"  - [{s['actor']}] {s['goal']} (sent={s['sentence_idx']})")
    #         print(f'    "{s["original_sentence"]}"')

    return {"grouping_done": True, "user_story_clusters": clusters_list}
