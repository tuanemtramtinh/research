from typing import List
import spacy
from spacy.language import Language
from spacy.tokens import Token

from ai.graphs.rpa_graph.state import GraphState


def _get_nlp():
    return spacy.load("en_core_web_lg")


def _find_main_verb(xcomp: Token) -> Token:
    """Find the main verb from an xcomp token."""
    # Handle case: "be Adjective to [VERB]"
    for child in xcomp.children:
        if child.dep_ == "acomp":
            for subchild in child.children:
                if subchild.dep_ == "xcomp" and subchild.pos_ == "VERB":
                    return subchild
            for subchild in child.children:
                if subchild.dep_ == "prep" and subchild.text.lower() == "of":
                    for grandchild in subchild.children:
                        if (
                            grandchild.dep_ in {"pcomp", "pobj"}
                            and grandchild.pos_ == "VERB"
                        ):
                            return grandchild

    # Handle case: "be V_3 to [VERB]"
    has_auxpass = any(
        child.dep_ == "auxpass" and child.text == "be" for child in xcomp.children
    )
    if has_auxpass:
        for child in xcomp.children:
            if child.dep_ == "xcomp" and child.pos_ == "VERB":
                return child

    return xcomp


def _get_verb_phrase(verb: Token) -> str:
    """Extract verb phrase from a verb token."""
    exclude_tokens = set()
    has_dobj = any(child.dep_ == "dobj" for child in verb.children)

    for child in verb.children:
        if child.dep_ == "conj":
            if not has_dobj:
                exclude_list = [
                    subchild for subchild in child.subtree if subchild.dep_ != "dobj"
                ]
            else:
                exclude_list = list(child.subtree)
            exclude_tokens.update(exclude_list)
        if child.dep_ == "cc":
            exclude_tokens.add(child)

    tokens = [t for t in verb.subtree if t not in exclude_tokens]
    tokens = tokens[tokens.index(verb) :]
    tokens = sorted(tokens, key=lambda t: t.i)

    # Remove "so that" clause if present
    cut_index = -1
    for i, token in enumerate(tokens):
        if (
            token.text.lower() == "so"
            and i + 1 < len(tokens)
            and tokens[i + 1].text.lower() == "that"
        ):
            cut_index = i
            break

    if cut_index != -1:
        tokens = tokens[:cut_index]

    EXCLUDE_DEPS = {"poss", "det", "nummod", "quantmod"}
    relevant_tokens = []

    for token in tokens[1:]:
        if token.dep_ == "dobj" or token.head.dep_ == "dobj":
            if token.dep_ not in EXCLUDE_DEPS:
                relevant_tokens.append(token)
        elif token.dep_ in {"prep", "pobj"} or token.head.dep_ in {"prep", "pobj"}:
            if token.dep_ not in EXCLUDE_DEPS:
                relevant_tokens.append(token)
        elif token.dep_ == "prt":
            relevant_tokens.append(token)
        elif token.dep_ in {"acomp", "advmod"}:
            relevant_tokens.append(token)
        elif token.dep_ in {"compound", "amod"}:
            relevant_tokens.append(token)

    tokens = [tokens[0]] + sorted(relevant_tokens, key=lambda t: t.i)
    result = [token.text for token in tokens]

    return " ".join(result)


def _get_all_conj(verb: Token) -> List[Token]:
    """Find all conjunctions of the root verb."""
    result = []
    for child in verb.children:
        if child.dep_ == "conj" and child.pos_ == "VERB":
            result.append(child)
            result.extend(_get_all_conj(child))
    return result


def _find_goals_nlp(nlp: Language, sentences: List[str]) -> dict:
    """Extract user goals from all sentences using NLP pattern 'want to [verb]'."""
    res = {}

    for i, sent in enumerate(sentences):
        doc = nlp(sent)
        for token in doc:
            if token.lemma_ == "want":
                for children in token.children:
                    if children.dep_ == "xcomp" and children.pos_ in {"VERB", "AUX"}:
                        # Exclude V-ing case
                        if children.tag_ == "VBG":
                            continue

                        main_verb = _find_main_verb(children)
                        verb_phrase = _get_verb_phrase(main_verb)

                        if str(i) not in res:
                            res[str(i)] = []
                        res[str(i)].append(verb_phrase)

                        # Find ALL conj verbs (recursive)
                        all_conj_verbs = _get_all_conj(main_verb)
                        for conj in all_conj_verbs:
                            conj_verb_phrase = _get_verb_phrase(conj)
                            res[str(i)].append(conj_verb_phrase)

    return res


def find_goals_node(state: GraphState):
    """Extract user goals using NLP pattern (want to [verb])."""
    nlp = _get_nlp()

    print(state.get("actor_results"))
    sentences = state.get("sentences") or []
    raw_goals = _find_goals_nlp(nlp, sentences)
    return {"raw_goals": raw_goals}
