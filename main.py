"""
User Story Analysis Tool
Organized by Classes and Functional Groups
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
from typing import List
from dotenv import load_dotenv
import spacy
from langchain_openai import ChatOpenAI
from actor_finder import ActorFinder
from usecase_finder import UsecaseFinder, UsecaseRefinementResponse


# =============================================================================
# MAIN CLASS (Configuration & Initialization)
# =============================================================================
class Main:
    """Central class containing all configuration, initialization, and shared resources."""
    
    _instance = None  # Singleton instance
    
    def __new__(cls, input_file: str = "./input_user_stories.txt"):
        """Singleton pattern to ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, input_file: str = "./input_user_stories.txt"):
        if self._initialized:
            return
        
        # Load environment variables
        load_dotenv()
        
        # Initialize NLP model
        self.nlp = spacy.load("en_core_web_lg")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-5-mini", 
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Load input data
        with open(input_file, "r", encoding="UTF-8") as f:
            self.input_text = f.read()
        
        self.sents = self.input_text.split("\n")
        
        self._initialized = True
    
    def get_actor_finder(self) -> ActorFinder:
        """Factory method to create ActorFinder instance."""
        return ActorFinder(self.llm, self.sents)
    
    def get_usecase_finder(self) -> UsecaseFinder:
        """Factory method to create UsecaseFinder instance."""
        return UsecaseFinder(self.nlp, self.sents, self.llm)
    
    def get_visualizer(self) -> "DependencyVisualizer":
        """Factory method to create DependencyVisualizer instance."""
        return DependencyVisualizer(self.nlp)


# =============================================================================
# VISUALIZATION CLASS
# =============================================================================
class DependencyVisualizer:
    """Handles visualization of NLP dependency parsing."""
    
    def __init__(self, nlp_model):
        self.nlp = nlp_model
    
    def print_dependency_table(self, sentence: str):
        """Print dependency table of a sentence (readable in terminal)."""
        doc = self.nlp(sentence)
        print(f"\n{'TOKEN':<15} {'DEP':<12} {'HEAD':<15} {'POS':<8} {'INDEX'}")
        print("-" * 60)
        for token in doc:
            print(f"{token.text:<15} {token.dep_:<12} {token.head.text:<15} {token.pos_:<8} {token.i}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # Initialize Main class (loads all configuration and resources)
    app = Main(input_file="./input_user_stories.txt")
    
    # Get instances from factory methods
    actor_finder = app.get_actor_finder()
    usecase_finder = app.get_usecase_finder()
    visualizer = app.get_visualizer()
    
    # --- ACTORS ---
    # actors: List[str] = actor_finder.find_actors(app.input_text)
    # print(actors)
    # synonym_actors = actor_finder.synonym_actors_check(actors)
    # print(synonym_actors)
    # actors_alias = actor_finder.find_actors_alias(synonym_actors)
    # print(actors_alias)
    
    # --- USECASES ---
    visualizer.print_dependency_table(
        "As a customer, I want to view and download reports so that I sleep"
    )
    
    usecases = usecase_finder.find_usecases()
    print("=== Extracted Use Cases (NLP) ===")
    print(usecases)
    
    # print("\n=== Refined Use Cases (LLM) ===")
    # refined: List[UsecaseRefinementResponse] = usecase_finder.refine_usecases(usecases)
    
    # Print detailed refinements
    # for r in refined:
    #     print(f"\nSentence {r.sentence_idx}:")
    #     print(f"  Original: {r.original}")
    #     print(f"  Refined:  {r.refined}")
    #     if r.added:
    #         print(f"  Added:    {r.added}")
    #     if r.reasoning:
    #         print(f"  Reason:   {r.reasoning}")
    
    # Get simple dict format
    # print("\n=== Final Use Cases Dict ===")
    # print(refined.to_dict())
