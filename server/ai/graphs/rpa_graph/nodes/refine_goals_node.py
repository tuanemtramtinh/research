import json
from typing import List
from langchain.chat_models import BaseChatModel
from ai.graphs.rpa_graph.state import (
    GraphState,
    UsecaseRefinement,
    UsecaseRefinementResponse,
)


def refine_goals_node(state: GraphState):
    """Refine extracted goals using LLM (chưa phải Usecase; đặt tên Usecase sau khi gom cụm)."""

    def _refine_goals(
        model: BaseChatModel, sentences: List[str], goals: dict
    ) -> List[UsecaseRefinement]:
        """Use LLM to refine and complete extracted goals."""
        if model is None or not goals:
            # Fallback: return as-is without refinement
            return [
                UsecaseRefinement(
                    sentence_idx=int(idx),
                    original=ucs,
                    refined=ucs,
                    added=[],
                    reasoning=None,
                )
                for idx, ucs in goals.items()
            ]

        # Chỉ dùng phần trước "so that" khi gửi cho LLM, tránh để benefit ảnh hưởng việc refine goal
        def _strip_benefit_clause(sentence: str) -> str:
            if not sentence:
                return sentence
            lower = sentence.lower()
            idx = lower.find("so that")
            if idx == -1:
                return sentence
            return sentence[:idx].strip()

        sentences_no_benefit = [_strip_benefit_clause(sent) for sent in sentences]

        sents_text = "\n".join(
            [f'{i}: "{sent}"' for i, sent in enumerate(sentences_no_benefit)]
        )
        goal_text = json.dumps(goals, indent=2, ensure_ascii=False)

        system_prompt = """You are an expert in UML use case modeling and software requirements analysis.

    Your task is to refine extracted user GOALS (from "I want to ...") so that they are
    clear, verb–object phrases. These will later be clustered and named as Use Cases.

    OBJECTIVES:
    1. REVIEW
    - Verify whether each extracted item represents a true user goal.

    2. REFINE
    - Rewrite each goal into a concise verb–object phrase (verb + noun phrase).
    - Remove all implementation details, conditions, and variations.

    3. COMPLETE
    - Add missing goals ONLY if they are explicitly stated in the sentence.
    - Do NOT infer system behavior or business rules not written in the text.

    GOAL PHRASE RULES (sau sẽ gom cụm để đặt tên Usecase):
    - Start with an action verb in base form (e.g., Browse, View, Create, Update, Manage).
    - Represent exactly ONE goal per phrase.
    - Be short and abstract (typically 2–5 words).
    - Use business-level actions, not technical steps.
    - Use lowercase for all goal phrases.
    - Do NOT include:
    - "so that", purposes, or outcomes
    - constraints or options (e.g., payment methods, device types)
    - UI actions (click, tap, select) unless explicitly stated

    GOOD GOAL PHRASES:
    - browse products
    - view order history
    - checkout order
    - manage account
    - update profile

    BAD PHRASES:
    - browse products by category for easier searching
    - checkout using multiple payment methods
    - view order history to track purchases

    IMPORTANT CONSTRAINTS:
    - Do NOT merge multiple user goals into one phrase.
    - Do NOT invent goals not explicitly present in the sentence.
    - If a sentence does not contain a valid goal, return empty lists.

    OUTPUT REQUIREMENTS:
    - Follow the provided structured output schema exactly.
    - Populate:
    - original: extracted goals
    - refined: refined goal phrases
    - added: only missing but explicitly stated goals
    - Provide brief reasoning for any change or addition.
    """

        human_prompt = f"""## User Stories:
    {sents_text}

    ## Extracted Goals (by sentence index):
    {goal_text}

    TASK:
    - Refine each extracted goal into a clear verb–object phrase.
    - Add missing goals ONLY if they are explicitly stated in the sentence.
    - If no valid goal exists, return empty refined and added lists.

    Return the result strictly in the structured output format.
    """

        structured_llm = model.with_structured_output(UsecaseRefinementResponse)
        response: UsecaseRefinementResponse = structured_llm.invoke(
            [("system", system_prompt), ("human", human_prompt)]
        )

        return response.refinements

    model: BaseChatModel = state.get("llm")
    sentences = state.get("sentences") or []
    raw_goals = state.get("raw_goals") or {}
    refined_goals = _refine_goals(model, sentences, raw_goals)

    # DEBUG: in goals trước và sau refine
    print("\n==== refine_goals_node ====")
    print("raw_goals:", json.dumps(raw_goals, indent=2, ensure_ascii=False))
    for item in refined_goals:
        print(f"- sentence_idx={item.sentence_idx}")
        print(f"  original={item.original}")
        print(f"  refined={item.refined}")
        print(f"  added={item.added}")

    return {"refined_goals": refined_goals}
