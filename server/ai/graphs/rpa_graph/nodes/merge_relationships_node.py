from ai.graphs.rpa_graph.state import GraphState, UseCase, UseCaseRelationship


def merge_relationships_node(state: GraphState):
    """
    Gắn relationships vào UseCase. Nhận từ find_include_extend (include_extend_relationships)
    hoặc legacy 3-step (within_domain_relationships + cross_domain_relationships).
    """
    # print("\n" + "=" * 60)
    # print("MERGE RELATIONSHIPS")
    # print("=" * 60)

    use_cases = state.get("use_cases") or []
    include_extend_rels = state.get("include_extend_relationships") or []
    within_rels = state.get("within_domain_relationships") or []
    cross_rels = state.get("cross_domain_relationships") or []

    if include_extend_rels:
        all_relationships = include_extend_rels
        # print(
        #     f"\n📋 Input: {len(use_cases)} use case(s), {len(all_relationships)} relationship(s) (from find_include_extend)"
        # )
    else:
        all_relationships = within_rels + cross_rels
        # print(
        #     f"\n📋 Input: {len(use_cases)} use case(s), {len(within_rels)} within + {len(cross_rels)} cross relationship(s)"
        # )

    # print(f"\n🔄 Merging {len(all_relationships)} total relationship(s)...")

    # Build lookup: source_use_case -> list of relationships
    rel_lookup = {}
    for rel in all_relationships:
        source = rel["source_use_case"].lower()
        if source not in rel_lookup:
            rel_lookup[source] = []
        rel_lookup[source].append(
            UseCaseRelationship(
                type=rel["type"],
                target_use_case=rel["target_use_case"],
            )
        )

    # Update use cases with their relationships (preserve new schema)
    updated_use_cases = []
    use_cases_with_rels = 0
    for uc in use_cases:
        uc_name = uc.name.lower()
        relationships = rel_lookup.get(uc_name, [])
        if relationships:
            use_cases_with_rels += 1
        updated_use_cases.append(
            UseCase(
                id=uc.id,
                name=uc.name,
                description=uc.description,
                participating_actors=uc.participating_actors,
                user_stories=uc.user_stories,
                relationships=relationships,
            )
        )

    # Print relationships summary
    # print(f"\n✅ Merged relationships into {use_cases_with_rels} use case(s):")
    # print("\n--- FINAL INCLUDE/EXTEND RELATIONSHIPS ---")
    # for uc in updated_use_cases:
    #     if uc.relationships:
    #         print(f"\n  📌 {uc.name}:")
    #         for r in uc.relationships:
    #             print(f"     --{r.type}--> {r.target_use_case}")

    # print("\n" + "=" * 60)
    # print("RELATIONSHIP DETECTION COMPLETED")
    # print("=" * 60)

    return {"use_cases": updated_use_cases}
