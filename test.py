import os
from pydantic import BaseModel, Field
import spacy
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
  
  print(responses.actors)
  
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
  
  for item in response.mappings:
    print(item)
  
  return
  
#===Finding Usecase===

def find_usecases(sents: List[str]):
  
  def get_verb_phrase(verb: Token):
    tokens = list(verb.subtree)
    tokens = tokens[tokens.index(verb):]
    tokens = sorted(tokens, key=lambda t: t.i)
    
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
        
    return tokens
    
  res = {}
    
  for i,sent in enumerate(sents):
    doc = nlp(sent)
    for token in doc:
      if token.lemma_ == "want":
        for children in token.children:
          if children.dep_ == "xcomp" and children.pos_ == "VERB":
            print(children.text)
            verb_phrase = get_verb_phrase(children)
            if str(i) not in res:
              res[str(i)] = {}
            res[str(i)][children.text] = verb_phrase
  return res

#===Main===
# actors: List[str] = find_actors(input, sents)
# print(actors)
# actors = synonym_actors_check(actors)
# find_actors_alias(actors)


find_usecases(sents)