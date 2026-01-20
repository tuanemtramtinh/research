from __future__ import annotations

from typing import List, TypedDict

import os
import re

import spacy
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from spacy.tokens import Doc
from textacy import extract, preprocessing

from ..state import ActorAlias, AliasItem, RpaState, TaskItem, UseCase


# Keep the same semantics as the notebook
INTERNAL_SYSTEM_KEYWORDS = {"system", "software", "application", "app", "platform"}


class ActorList(BaseModel):
    actors: List[str] = Field(description="A list of actors who perform actions in the requirement.")


class ActorAliasMapping(BaseModel):
    mappings: List[ActorAlias] = Field(description="List of actor-alias mappings")


class GraphState(TypedDict, total=False):
    requirement_text: str
    input_text: str
    doc: Doc
    actors: List[str]
    actor_aliases: List[ActorAlias]
    tasks: List[TaskItem]
    use_cases: List[UseCase]


def _get_model():
    load_dotenv()
    # Match the notebook default; user can override via env if desired.
    model_name = "gpt-5-mini"
    if not os.getenv("OPENAI_API_KEY"):
        return None
    return init_chat_model(model_name, model_provider="openai")


def _get_nlp():
    # Same as notebook
    return spacy.load("en_core_web_lg")


def _clean_np(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^\b(the|a|an)\b\s+", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def _heuristic_actors_from_doc(doc: Doc) -> List[str]:
    actors: List[str] = []
    seen = set()

    # Prefer grammatical subjects as actors
    for token in doc:
        if token.dep_ in {"nsubj", "nsubjpass"} and token.pos_ in {"NOUN", "PROPN"}:
            lemma = (token.lemma_ or token.text).strip()
            if not lemma:
                continue
            if lemma.lower() in INTERNAL_SYSTEM_KEYWORDS:
                continue

            span_text = _clean_np(doc[token.left_edge.i : token.right_edge.i + 1].text)
            # Use lemma if it's a single word; otherwise keep phrase text
            candidate = lemma if " " not in span_text else span_text
            key = candidate.lower()
            if key not in seen:
                seen.add(key)
                actors.append(candidate)

    # Fallback: any noun chunks in the doc
    if not actors:
        for chunk in doc.noun_chunks:
            root = chunk.root
            if root.pos_ == "PRON":
                continue
            lemma = (root.lemma_ or root.text).strip()
            if not lemma or lemma.lower() in INTERNAL_SYSTEM_KEYWORDS:
                continue
            candidate = _clean_np(chunk.text)
            if not candidate:
                continue
            key = candidate.lower()
            if key not in seen:
                seen.add(key)
                actors.append(candidate)

    return actors


def _heuristic_aliases(doc: Doc, actors: List[str]) -> List[ActorAlias]:
    sentences = [s.text.strip() for s in doc.sents]
    out: List[ActorAlias] = []
    for actor in actors:
        if not actor:
            continue
        pattern = re.compile(rf"\b{re.escape(actor)}\b", flags=re.IGNORECASE)
        hits = [i + 1 for i, s in enumerate(sentences) if pattern.search(s)]
        out.append(ActorAlias(actor=actor, aliases=[AliasItem(alias=actor, sentences=hits)]))
    return out


def pre_process_node(state: GraphState):
    text = state.get("requirement_text") or state.get("input_text") or ""
    text = preprocessing.normalize.whitespace(text)
    nlp = _get_nlp()
    doc = nlp(text)

    chunks = extract.noun_chunks(doc, min_freq=1)
    chunks = [
        chunk.text
        for chunk in chunks
        if chunk.root.pos_ != "PRON" and chunk.root.lemma_.lower() not in INTERNAL_SYSTEM_KEYWORDS
    ]

    return {"input_text": text, "doc": doc, "actors": chunks, "actor_aliases": [], "tasks": []}


def tasks_node(state: GraphState):
    doc = state.get("doc")
    sentences = list(doc.sents) if doc is not None else []
    tasks = [TaskItem(id=i + 1, text=s.text.strip()) for i, s in enumerate(sentences) if s.text.strip()]
    return {"tasks": tasks}


def _actors_in_sentence(sentence: str, actors: List[str]) -> List[str]:
    s = sentence.lower()
    found: List[str] = []
    for a in actors:
        if not a:
            continue
        if " " in a:
            ok = a.lower() in s
        else:
            ok = re.search(rf"\b{re.escape(a.lower())}\b", s) is not None
        if ok:
            found.append(a)
    return found


def _heuristic_usecases(task_id: int, sentence: str, actors: List[str]) -> List[UseCase]:
    nlp = _get_nlp()
    doc = nlp(sentence)

    participating = _actors_in_sentence(sentence, actors)

    verbs = []
    root = next((t for t in doc if t.dep_ == "ROOT" and t.pos_ == "VERB"), None)
    if root is not None:
        verbs.append(root)
        verbs.extend([t for t in root.conjuncts if t.pos_ == "VERB"])
    else:
        verbs = [t for t in doc if t.pos_ == "VERB"]

    use_cases: List[UseCase] = []
    seen = set()
    for v in verbs:
        verb = (v.lemma_ or v.text).strip()
        if not verb:
            continue

        obj = None
        for child in v.children:
            if child.dep_ in {"dobj", "obj", "pobj"}:
                obj = child
                break

        if obj is not None:
            obj_text = doc[obj.left_edge.i : obj.right_edge.i + 1].text.strip()
            name = f"{verb.title()} {obj_text.title()}".strip()
        else:
            name = f"{verb.title()}".strip()

        key = (name.lower(), task_id)
        if key in seen:
            continue
        seen.add(key)

        if not participating:
            # If no explicit actor mention, skip to avoid producing actor-less use cases.
            continue

        use_cases.append(
            UseCase(
                name=name,
                participating_actors=participating,
                sentence_id=task_id,
                sentence=sentence,
                relationships=[],
            )
        )

    return use_cases


class UseCaseList(BaseModel):
    use_cases: List[UseCase] = Field(description="Use cases extracted from the sentence")


def _extract_use_cases_for_task(
    *,
    model,
    task_id: int,
    sentence: str,
    requirement_text: str,
    actors: List[str],
    actor_aliases: List[ActorAlias],
) -> List[UseCase]:
    if model is None:
        return _heuristic_usecases(task_id, sentence, actors)

    structured_llm = model.with_structured_output(UseCaseList)

    actors_list = ", ".join(actors)
    aliases_str = "\n".join([f"- {aa.actor}: {[a.alias for a in aa.aliases]}" for aa in actor_aliases])

    system_prompt = (
        "You are a Senior Systems Analyst. Your task is to extract UML-style Use Cases "
        "from a single requirement sentence. A use case should be an action/goal expressed "
        "as a verb phrase and must have participating actors."
    )

    human_prompt = f"""
Extract use cases from the sentence below.

### CONTEXT
- Full Requirement Text (for reference only):
{requirement_text}

### ACTORS (canonical list)
{actors_list}

### ACTOR ALIASES (for reference)
{aliases_str}

### SENTENCE
- sentence_id: {task_id}
- sentence: {sentence}

### RULES
1. Only output use cases that are supported by the sentence.
2. UseCase.name should be a short verb phrase (e.g., "Search Book", "Borrow Book", "Generate Report").
3. participating_actors must be chosen from the canonical ACTORS list.
4. Always set sentence_id to {task_id} and sentence to the exact sentence string.
5. relationships can be empty unless the sentence explicitly implies include/extend.
6. Do not output a use case unless there is at least one participating actor.

Return structured output only.
"""

    response: UseCaseList = structured_llm.invoke([("system", system_prompt), ("human", human_prompt)])
    # Filter to guarantee each use case has actors.
    return [uc for uc in (response.use_cases or []) if uc.participating_actors]


def use_cases_node(state: GraphState):
    model = _get_model()
    requirement_text = (state.get("input_text") or state.get("requirement_text") or "").strip()
    tasks = state.get("tasks") or []
    actors = state.get("actors") or []
    actor_aliases = state.get("actor_aliases") or []

    use_cases: List[UseCase] = []
    for t in tasks:
        sentence = (t.text or "").strip()
        if not sentence:
            continue
        use_cases.extend(
            _extract_use_cases_for_task(
                model=model,
                task_id=int(t.id),
                sentence=sentence,
                requirement_text=requirement_text,
                actors=actors,
                actor_aliases=actor_aliases,
            )
        )

    return {"use_cases": use_cases}


def actors_node(state: GraphState):
    model = _get_model()
    doc = state.get("doc")
    if model is None:
        if doc is None:
            return {"actors": []}
        return {"actors": _heuristic_actors_from_doc(doc)}

    structured_llm = model.with_structured_output(ActorList)
    candidate_chunks = ", ".join(state.get("actors") or [])

    system_prompt = (
        "You are a Senior Systems Analyst and Linguistic Expert. Your task is to perform "
        "Entity Extraction specifically for 'Actors' in a system description or user story. "
        "An 'Actor' is defined as a person, organization, or external system that "
        "performs actions, initiates processes, or interacts with the system described."
    )

    user_prompt = f"""
I will provide you with a raw text and a list of potential 'Noun Chunks' extracted by a parser.

### RULES:
1. **Filtering**: From the 'Candidate Noun Chunks', select only those that function as an active agent (Actor) in the 'Raw Text'.
2. **Standardization**: Convert all extracted actors to their **singular form** (e.g., 'customers' -> 'customer').
3. **Cleaning**: Remove any unnecessary articles (a, an, the) and honorifics.
4. **Context Check**: Ensure the noun chunk is actually performing an action in the text, not just being mentioned as an object.
5. **Exclude Self-References**: Do NOT include 'the system', 'the software', or 'the application' as an Actor if it refers to the system being described. These are internal components, not external actors.
6. **External Systems**: Only include other specific systems if they are external entities that your system interacts with (e.g., 'Payment Gateway', 'External Database').

### INPUT DATA:
- Raw Text: {state.get('input_text')}
- Candidate Noun Chunks: {candidate_chunks}

### OUTPUT INSTRUCTIONS:
Return only the final list of singularized actors.
"""

    response: ActorList = structured_llm.invoke([("system", system_prompt), ("human", user_prompt)])
    return {"actors": response.actors}


def actors_alias_node(state: GraphState):
    model = _get_model()
    doc = state.get("doc")
    if model is None:
        if doc is None:
            return {"actor_aliases": []}
        return {"actor_aliases": _heuristic_aliases(doc, state.get("actors") or [])}

    structured_llm = model.with_structured_output(ActorAliasMapping)

    sentences = list(doc.sents) if doc is not None else []

    input_sentences = [s.text.strip() for s in sentences]
    input_sentences = "\n".join([f"{i + 1}. {s}" for i, s in enumerate(input_sentences)])

    actors_list = ", ".join(state.get("actors") or [])

    system_prompt = (
        "You are an expert in Natural Language Processing and Entity Resolution. "
        "Your task is to identify all alternative references (aliases) to specific actors "
        "in a given text and map EACH UNIQUE ALIAS to the specific sentences where it appears."
    )

    user_prompt = f"""
Analyze the following pre-segmented numbered sentences and identify all references to the given actors.

### ACTORS TO TRACK:
{actors_list}

### PRE-SEGMENTED SENTENCES:
{input_sentences}

### CRITICAL INSTRUCTIONS:
- **Each unique alias must be tracked separately** with its own sentence list
- Even slight variations count as different aliases (e.g., "customer" vs "the customer" are TWO separate aliases)
- DO NOT merge aliases together - keep them distinct
- Each alias should map ONLY to sentences where that EXACT form appears

### TASK:
For each actor:
1. Identify ALL distinct ways it is referenced (exact name, pronouns, role titles, variations)
2. For EACH unique alias, list the sentence numbers where THAT SPECIFIC alias appears
3. Treat each variation as a separate alias entry

### OUTPUT REQUIREMENTS:
- In the 'sentences' field, use ONLY the sentence number (integer starting from 1)
- Each alias must appear as a separate AliasItem
- Do NOT combine or merge similar aliases
"""

    response: ActorAliasMapping = structured_llm.invoke([("system", system_prompt), ("human", user_prompt)])
    return {"actor_aliases": response.mappings}


def finalize_node(state: GraphState):
    return {
        "requirement_text": state.get("input_text") or state.get("requirement_text") or "",
        "tasks": state.get("tasks") or [],
        "actors": state.get("actors") or [],
        "actor_aliases": state.get("actor_aliases") or [],
        "use_cases": state.get("use_cases") or [],
    }


def build_rpa_graph():
    """Agent 1 (Generator): extract tasks (sentences) + actors + aliases from the requirement."""

    workflow = StateGraph(GraphState)

    workflow.add_node("pre_process", pre_process_node)
    workflow.add_node("tasks", tasks_node)
    workflow.add_node("actors", actors_node)
    workflow.add_node("aliases", actors_alias_node)
    workflow.add_node("use_cases", use_cases_node)
    workflow.add_node("finalize", finalize_node)

    workflow.add_edge(START, "pre_process")
    workflow.add_edge("pre_process", "tasks")
    workflow.add_edge("tasks", "actors")
    workflow.add_edge("actors", "aliases")
    workflow.add_edge("aliases", "use_cases")
    workflow.add_edge("use_cases", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile()


def run_rpa(requirement_text: str) -> RpaState:
    app = build_rpa_graph()
    out = app.invoke({"requirement_text": requirement_text})
    return {
        "requirement_text": out.get("requirement_text", requirement_text),
        "tasks": out.get("tasks", []),
        "actors": out.get("actors", []),
        "actor_aliases": out.get("actor_aliases", []),
        "use_cases": out.get("use_cases", []),
    }
