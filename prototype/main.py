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
from actor_finder import ActorAliasMapping, ActorFinder, ActorResult, AliasItem
from usecase_finder import UsecaseFinder, UsecaseRefinement, UsecaseRefinementResponse


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
        self.llm = ChatOpenAI(model="gpt-5-mini", api_key=os.getenv("OPENAI_API_KEY"))

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

    def link_usecase_and_actor(self):
        return


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
            print(
                f"{token.text:<15} {token.dep_:<12} {token.head.text:<15} {token.pos_:<8} {token.i}"
            )


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # Initialize Main class (loads all configuration and resources)
    app = Main(input_file="./input_user_stories_new.txt")

    # Get instances from factory methods
    actor_finder = app.get_actor_finder()
    usecase_finder = app.get_usecase_finder()
    visualizer = app.get_visualizer()

    # --- ACTORS ---
    # actors: List[str] = actor_finder.find_actors(app.input_text)
    # print("\n=== Find Actors Manually ===")
    # print(actors)

    # print("\n=== Remove Synonym Actors ===")
    # synonym_actors = actor_finder.synonym_actors_check(actors)
    # print(synonym_actors)

    # print("\n=== Find the Alias of Actors ===")
    # actors_alias_result = actor_finder.find_actors_alias(synonym_actors)
    # print(actors_alias_result)

    # # --- USECASES ---
    visualizer.print_dependency_table(
        "As a customer, I want to I want to be allowed to update profile so that I can sleep."
    )

    usecases = usecase_finder.find_usecases()
    print("\n=== Extracted Use Cases (NLP) ===")
    print(usecases)

    # print("\n=== Refined Use Cases (LLM) ===")
    # refined_usecases_result: List[UsecaseRefinementResponse] = (
    #     usecase_finder.refine_usecases(usecases)
    # )
    # # Print detailed refinements
    # for r in refined_usecases_result:
    #     print(f"\nSentence {r.sentence_idx}:")
    #     print(f"  Original: {r.original}")
    #     print(f"  Refined:  {r.refined}")
    #     if r.added:
    #         print(f"  Added:    {r.added}")
    #     if r.reasoning:
    #         print(f"  Reason:   {r.reasoning}")

    # linking
    # actor_input = [
    #     ActorResult(
    #         actor="customer",
    #         aliases=[
    #             AliasItem(alias="shopper", sentences=[1, 2, 3, 4, 5, 6]),
    #             AliasItem(alias="customers", sentences=[16]),
    #         ],
    #         sentence_idx=[0],
    #     ),
    #     ActorResult(actor="user", aliases=[], sentence_idx=[7, 8, 9, 10, 11]),
    #     ActorResult(
    #         actor="admin",
    #         aliases=[AliasItem(alias="system operator", sentences=[14])],
    #         sentence_idx=[12, 16, 17, 18, 19],
    #     ),
    #     ActorResult(actor="manager", aliases=[], sentence_idx=[13]),
    #     ActorResult(actor="power user", aliases=[], sentence_idx=[15]),
    #     ActorResult(actor="system", aliases=[], sentence_idx=[20, 21]),
    # ]

    # usecase_input = [
    #     UsecaseRefinement(
    #         sentence_idx=0,
    #         original=["view reports", "download reports"],
    #         refined=["view reports", "download reports"],
    #         added=[],
    #         reasoning="Kept both goals. Removed the sentence purpose ('so that I sleep') because use case names must be action–object only.",
    #     ),
    #     UsecaseRefinement(
    #         sentence_idx=1,
    #         original=["browse products by category"],
    #         refined=["browse products"],
    #         added=[],
    #         reasoning="Removed the constraint 'by category' to produce a concise business-level use case name.",
    #     ),
    #     UsecaseRefinement(
    #         sentence_idx=2,
    #         original=["search for products by keyword"],
    #         refined=["search products"],
    #         added=[],
    #         reasoning="Removed the technical constraint 'by keyword' to keep a short verb–object use case name.",
    #     ),
    #     UsecaseRefinement(
    #         sentence_idx=3,
    #         original=["view product details"],
    #         refined=["view product details"],
    #         added=[],
    #         reasoning="Already a concise, goal-oriented use case name.",
    #     ),
    #     UsecaseRefinement(
    #         sentence_idx=4,
    #         original=["add products to cart"],
    #         refined=["add products to cart"],
    #         added=[],
    #         reasoning="Kept as-is; it is a single, clear user goal in verb–object form.",
    #     ),
    #     UsecaseRefinement(
    #         sentence_idx=5,
    #         original=["checkout"],
    #         refined=["checkout order"],
    #         added=[],
    #         reasoning="Converted to verb–object form and removed the payment-method constraint from the original sentence.",
    #     ),
    #     UsecaseRefinement(
    #         sentence_idx=6,
    #         original=["track orders"],
    #         refined=["track orders"],
    #         added=[],
    #         reasoning="Already a concise user goal; left unchanged.",
    #     ),
    #     UsecaseRefinement(
    #         sentence_idx=7,
    #         original=["register account"],
    #         refined=["register account"],
    #         added=[],
    #         reasoning="Kept the goal-oriented phrase; suitable as a use case name.",
    #     ),
    #     UsecaseRefinement(
    #         sentence_idx=8,
    #         original=["log in to system"],
    #         refined=["log in to account"],
    #         added=[],
    #         reasoning="Reworded to the business-level goal 'log in to account', avoiding reference to the internal 'system'.",
    #     ),
    #     UsecaseRefinement(
    #         sentence_idx=9,
    #         original=["update profile information"],
    #         refined=["update profile"],
    #         added=[],
    #         reasoning="Shortened to a concise verb–object phrase by removing the extra noun 'information'.",
    #     ),
    #     UsecaseRefinement(
    #         sentence_idx=10,
    #         original=["reset password"],
    #         refined=["reset password"],
    #         added=[],
    #         reasoning="Already a clear user goal in verb–object form.",
    #     ),
    #     UsecaseRefinement(
    #         sentence_idx=11,
    #         original=["download reports"],
    #         refined=["download reports"],
    #         added=[],
    #         reasoning="Duplicate of an earlier goal but still a valid, concise use case name.",
    #     ),
    #     UsecaseRefinement(
    #         sentence_idx=12,
    #         original=["manage user accounts"],
    #         refined=["manage user accounts"],
    #         added=[],
    #         reasoning="Already a proper business-level use case name.",
    #     ),
    #     UsecaseRefinement(
    #         sentence_idx=13,
    #         original=["approve orders"],
    #         refined=["approve orders"],
    #         added=[],
    #         reasoning="Kept as-is; single clear goal for the manager actor.",
    #     ),
    #     UsecaseRefinement(
    #         sentence_idx=14,
    #         original=["configure system settings"],
    #         refined=["configure system settings"],
    #         added=[],
    #         reasoning="Already a concise, appropriate use case name for an operator role.",
    #     ),
    #     UsecaseRefinement(
    #         sentence_idx=15,
    #         original=["handling bulk operations"],
    #         refined=["handle bulk operations"],
    #         added=[],
    #         reasoning="Converted to base-verb form ('handle') to follow verb–object naming conventions.",
    #     ),
    #     UsecaseRefinement(
    #         sentence_idx=16,
    #         original=["create new products"],
    #         refined=["create products"],
    #         added=[],
    #         reasoning="Removed the non-essential adjective 'new' to keep the name short and abstract.",
    #     ),
    #     UsecaseRefinement(
    #         sentence_idx=17,
    #         original=["update product prices"],
    #         refined=["update product prices"],
    #         added=[],
    #         reasoning="Already a clear, business-level use case name.",
    #     ),
    #     UsecaseRefinement(
    #         sentence_idx=18,
    #         original=["manage inventory levels"],
    #         refined=["manage inventory"],
    #         added=[],
    #         reasoning="Shortened to 'manage inventory' for conciseness while preserving the stated goal.",
    #     ),
    #     UsecaseRefinement(
    #         sentence_idx=19,
    #         original=["view sales reports"],
    #         refined=["view sales reports"],
    #         added=[],
    #         reasoning="Already a concise, goal-oriented use case name.",
    #     ),
    #     UsecaseRefinement(
    #         sentence_idx=20,
    #         original=["log user activities"],
    #         refined=[],
    #         added=[],
    #         reasoning="This is an internal/system technical task (system actor) rather than a user goal; not suitable as a UML use case name.",
    #     ),
    #     UsecaseRefinement(
    #         sentence_idx=21,
    #         original=["cache frequently accessed data"],
    #         refined=[],
    #         added=[],
    #         reasoning="This describes an internal performance mechanism (system-level), not a user goal; therefore not appropriate as a UML use case.",
    #     ),
    # ]

    # print("\n=== Mapping Actor with Usecase ===")
    # usecase_finder.format_usecase_output(
    #     usecases=refined_usecases_result, actors=actors_alias_result
    # )

    # Get simple dict format
    # print("\n=== Final Use Cases Dict ===")
    # print(refined.to_dict())
