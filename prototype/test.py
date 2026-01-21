import os
from pydantic import BaseModel, Field
import spacy
from spacy import displacy
from textacy import extract, preprocessing
from typing import Annotated, List
from spacy.tokens import Doc, Token
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import re

from thinc import api

#load configuration
load_dotenv()
nlp = spacy.load("en_core_web_lg")

#DTO
class ActorList(BaseModel):
  actors: List[str] = Field(description="A list of actors who perform actions in the user stories.")

class AliasItem(BaseModel):
  alias: str = Field(description="Name of the alias")
  sentences: List[int] = Field(description="List of sentence indices where THIS specific alias appears (starting from 0)")
  
  def __str__(self):
    return f"'{self.alias}' -> sentences: {self.sentences}"

class ActorAlias(BaseModel):
  actor: str = Field(description="The original actor's name")
  aliases: List[AliasItem] = Field(description="List of alternative names/references for this actor")
  
  def __str__(self):
    if not self.aliases:
      return f"Actor: {self.actor} (no aliases)"
    aliases_str = ", ".join(str(alias) for alias in self.aliases)
    return f"Actor: {self.actor} | Aliases: [{aliases_str}]"
  
class ActorAliasList(BaseModel):
  mappings: List[ActorAlias] = Field(description="List of actor-alias mappings")

#Program
with open("./input_user_stories.txt", "r", encoding="UTF-8") as f:
  input = f.read()
  
sents = input.split("\n")
llm = ChatOpenAI(model="gpt-5-mini", api_key=os.getenv("OPENAI_API_KEY"))

#===Finding Actors===

#Hàm tìm kiếm actors
def find_actors(input: str, sents: List[str]):
  pattern = r"As\s+(?:a|an|the)\s+([^,]+)" #Pattern để trích xuất actor từ user story
  actors = set(map(lambda sent: re.search(pattern, sent).group(1).strip(), sents))
  return list(actors)
  
def synonym_actors_check(actors: List[str]):
  
  structured_llm = llm.with_structured_output(ActorList)
  
  system_prompt = """
  You are a Business Analyst AI specializing in requirement analysis.

  Your task is to analyze a list of actors and remove synonymous or semantically equivalent actors.

  Rules:
  - Actors that represent the same logical role MUST be merged.
  - Choose ONE clear and generic canonical name for each group.
  - Prefer business-level, role-based names over wording variants.
  - ALL returned actor names MUST be lowercase.
  - Do NOT invent new actors that are not implied by the list.
  - Do NOT explain your reasoning.
  - Return only structured data according to the output schema.
  """
  
  human_prompt = f"""
  The following is a list of actors extracted from user stories.

  Actors:
  {actors}

  Remove synonymous actors and return a list of unique canonical actors.
  """
  
  messages = [
    (
      "system",
      system_prompt
    ),
    (
      "human",
      human_prompt
    )
  ]
  
  responses = structured_llm.invoke(messages)
    
  return responses.actors
  
def find_actors_alias(actors: List[str]):
  structured_llm = llm.with_structured_output(ActorAliasList)
  indexed_sents = "\n".join(f"{i}: {sent}" for i, sent in enumerate(sents))
  
  system_prompt = """
  You are a Business Analyst AI specializing in requirement analysis.

  Your task is to identify aliases (alternative names or references) for each canonical actor
  based on a list of user story sentences.

  Rules:
  - An alias is a different term that refers to the SAME logical actor.
  - Canonical actor names MUST NOT be listed as aliases of themselves.
  - Each alias MUST map to exactly one canonical actor.
  - Aliases must be explicitly present in the provided sentences.
  - Sentence indices are ZERO-BASED.
  - If an actor has no aliases, return an empty alias list for that actor.
  - ALL actor and alias names MUST be lowercase.
  - Do NOT invent aliases.
  - Do NOT explain your reasoning.
  - Return only structured data according to the output schema.
  """
  
  human_prompt = f"""
  Canonical actors:
  {actors}

  User story sentences (with indices):
  {indexed_sents}

  For each canonical actor, find all aliases used in the sentences above and list the sentence indices where each alias appears.
  """
  
  message = [
    (
      "system",
      system_prompt
    ),
    (
      "human",
      human_prompt
    )
  ]
  
  response = structured_llm.invoke(message)
  
  # for item in response.mappings:
  #   print(item)
  
  return response.mappings
  
#===Finding Usecase===

def find_usecases(sents: List[str]):
  
  def find_main_verb(xcomp: Token):
    
    # Xử lý cho trường hợp "be Adjective to [VERB]"
    for child in xcomp.children:
      if child.dep_ == "acomp":
        for subchild in child.children:
          if subchild.dep_ == "xcomp" and subchild.pos_ == "VERB":
            return subchild
          
    # Xử lý cho trường hợp "be V_3 to [VERB]"
    has_auxpass = any(child.dep_ == "auxpass" and child.dep_ == "be" for child in xcomp.children)
    if has_auxpass:
      for child in xcomp.children:
        if child.dep_ == "xcomp" and child.pos_ == "VERB":
          return child
        
    return xcomp
  
  def get_verb_phrase(verb: Token):
    exclude_tokens = set()
    for child in verb.children:
      if child.dep_ == "conj":  # login, swim, etc.
        exclude_tokens.update(child.subtree)  # loại cả subtree của conj
      if child.dep_ == "cc":  # and, or, but
        exclude_tokens.add(child)
                
    tokens = [t for t in verb.subtree if t not in exclude_tokens]
    tokens = tokens[tokens.index(verb):]
    tokens = sorted(tokens, key=lambda t: t.i)
    
    cut_index = -1
    
    #Loại bỏ so that (nếu có)
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
    
      # 2. Giữ prepositional phrases
      elif token.dep_ in {"prep", "pobj"} or token.head.dep_ in {"prep", "pobj"}:
        if token.dep_ not in EXCLUDE_DEPS:
          relevant_tokens.append(token)
      
      # 3. Giữ particle (phrasal verbs)
      elif token.dep_ == "prt":
        relevant_tokens.append(token)
      
      # 4. Giữ adjective/adverb complements
      elif token.dep_ in {"acomp", "advmod"}:
        relevant_tokens.append(token)
      
      # 5. Giữ compound nouns và adjective modifiers
      elif token.dep_ in {"compound", "amod"}:
        relevant_tokens.append(token)
    
    tokens = [tokens[0]] + sorted(relevant_tokens, key=lambda t: t.i)
    # result = [token.lemma_ for token in tokens]
    result = [token.text for token in tokens]
        
    return " ".join(result)
  
  def get_all_conj(verb: Token):
    """Tìm tất cả conj của verb gốc (swim and sleep and eat). Với verb gốc là swim, conj verb là sleep và eat"""
    result = []
    for child in verb.children:
      if child.dep_ == "conj" and child.pos_ == "VERB":
        result.append(child)
        result.extend(get_all_conj(child))  # Đệ quy tìm tiếp
    return result
    
  res = {}
    
  for i,sent in enumerate(sents):
    doc = nlp(sent)
    for token in doc:
      if token.lemma_ == "want":
        for children in token.children:          
          if children.dep_ == "xcomp" and children.pos_ in {"VERB", "AUX"}:
            #Loại bỏ trường hợp V-ing cũng trở thành xcomp của want 
            # (checkout using multiple payment methods)
            if children.tag_ == "VBG":
              continue
            
            main_verb = find_main_verb(children)
            verb_phrase = get_verb_phrase(main_verb)
            
            if str(i) not in res: 
              res[str(i)] = {}
            res[str(i)][main_verb.text] = verb_phrase
            
            # Tìm TẤT CẢ các động từ conj (đệ quy)
            all_conj_verbs = get_all_conj(main_verb)
            for conj in all_conj_verbs:
              conj_verb_phrase = get_verb_phrase(conj)
              res[str(i)][conj.text] = conj_verb_phrase
            
  return res

#===Visualization===
def print_dependency_table(sentence: str):
  """
  In bảng dependency của câu (dễ đọc trong terminal)
  """
  doc = nlp(sentence)
  print(f"\n{'TOKEN':<15} {'DEP':<12} {'HEAD':<15} {'POS':<8} {'INDEX'}")
  print("-" * 60)
  for token in doc:
    print(f"{token.text:<15} {token.dep_:<12} {token.head.text:<15} {token.pos_:<8} {token.i}")

#===Main===

#ACTORS
# actors: List[str] = find_actors(input, sents)
# print(actors)
# synonym_actors = synonym_actors_check(actors)
# print(synonym_actors)
# actors_alias = find_actors_alias(synonym_actors)
# print(actors_alias)

#USECASES
# print_dependency_table("As a buyer, I want to checkout using multiple payment methods so that I can complete my purchase.")
print(find_usecases(sents))