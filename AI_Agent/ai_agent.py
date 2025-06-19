from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import re

load_dotenv()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

prompt = PromptTemplate(
    input_variables=["item", "location", "curbside_composting", "home_composting"],
    template="""
You are a Waste Disposal Classification Assistant.

Classify the correct curbside or at-home disposal method for a given item based on:
- City location
- Whether curbside composting is available
- Whether the user has home composting access

Use one of:
- recycle
- compost
- trash

Only return:
classification: <recycle / compost / trash>
reason: <brief explanation based ONLY on the rules below>

Compost if:
- The item is food waste, food-soiled paper, or certified compostable **AND**
  - (Curbside composting is available OR home composting is available)

Recycle if:
- The item is accepted by curbside recycling in most U.S. cities (e.g., clean paper, aluminum cans, #1/#2 plastics)

Trash if:
- The item is not accepted in local curbside recycling or composting
- The item is plastic-lined, ambiguous, mixed-material, or not explicitly allowed

If home composting is available, compost the item if it is:
- A food scrap (e.g., banana peel, vegetable trimmings)
- Food-soiled paper (e.g., greasy napkin, coffee filter)

Avoid suggesting drop-off locations or general advice. Be strict.

Now classify the following:
Item: {item}  
Location: {location}  
Curbside Composting: {curbside_composting}  
Home Composting: {home_composting}
"""
)

known_composting_cities = {
    "San Francisco, CA", "Seattle, WA", "Portland, OR", "Minneapolis, MN",
    "Denver, CO", "Austin, TX", "Madison, WI", "Boston, MA", "Charlotte, NC",
    "New York, NY", "Los Angeles, CA", "Philadelphia, PA", "Chicago, IL",
    "Washington, DC", "Atlanta, GA", "Berkeley, CA"
}

home_compost_items = [
    "banana peel", "apple core", "vegetable trimmings", "coffee filter",
    "greasy napkin", "used paper towel", "fruit scraps", "tea leaves",
    "food scraps", "compostable plate", "compostable fork", "rotten tomato",
    "eggshell", "leftover food", "food-soiled paper", "pizza crust",
    "lettuce", "cabbage", "bread crust", "orange peel", "vegetable scraps",
    "moldy bread", "avocado skin", "napkin", "paper towel", "paper plate",
    "fruit peel", "greasy pizza box", "banana peel", "stained napkin",
    "used coffee filter", "soiled cardboard"
]


known_recyclables = [
    "paper towel roll", "cardboard", "clean paper", "paper cup (clean)", "aluminum can",
    "tin can", "milk jug", "soda can", "#1 plastic", "#2 plastic", "plastic bottle"
]

always_trash_terms = [
    "chip bag", "plastic utensil", "foil wrapper", "styrofoam", "foam cup",
    "plastic straw", "plastic-lined", "candy wrapper", "crinkly", "k-cup",
    "ceramic", "broken glass", "tissue", "diaper",
    "compostable cup", "bioplastic", "PLA", "takeout container",
    "soup cup", "cold cup", "plant-based plastic", "clamshell", "compostable wrapper"
]

def has_curbside_composting(city):
    return "yes" if city.strip() in known_composting_cities else "no"

chain = prompt | llm

def classify_item(item, location, home_composting="no"):
    item_lower = item.lower()

    # Tier 1: Home compost
    if home_composting == "yes" and any(term in item_lower for term in home_compost_items):
        return {
            "response": "classification: compost\nreason: This item is food waste or soiled paper accepted in home composting.",
            "classification": "compost"
        }

    # Tier 2: Recyclables
    if any(term in item_lower for term in known_recyclables):
        return {
            "response": "classification: recycle\nreason: This item is accepted in most curbside recycling programs.",
            "classification": "recycle"
        }

    # Tier 3: Always trash
    if any(term in item_lower for term in always_trash_terms):
        return {
            "response": "classification: trash\nreason: This item is plastic-lined, ambiguous, or not accepted curbside.",
            "classification": "trash"
        }

    # Tier 4: LLM fallback
    curbside_composting = has_curbside_composting(location)
    response = chain.invoke({
        "item": item,
        "location": location,
        "curbside_composting": curbside_composting,
        "home_composting": home_composting
    })

    content = response.content.strip().lower()
    match = re.search(r"classification:\s*(recycle|compost|trash)", content)
    classification = match.group(1) if match else "unknown"

    return {
        "response": response.content.strip() if response.content else "No response",
        "classification": classification
    }

__all__ = ["classify_item"]
