from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

# Track user interaction history with memory
memory = ConversationBufferMemory(
    memory_key="history",
    return_messages=True,
    input_key="item"
)

# Refined prompt for classification
prompt = PromptTemplate(
    input_variables=["item", "location"],
    template="""
You are a Recycling Classification Assistant.

Classify household waste according to the **curbside** rules in the provided U.S. city. Use the latest known local disposal policies.

Use ONLY these labels:
- recycle
- compost
- trash
Be aware of city-specific exceptions. Some items like tea bags, shredded paper, and wooden utensils may seem compostable, but are not accepted in all municipal curbside compost programs.

Respond in this format:
classification: <recycle / compost / trash>
reason: <concise explanation with specific local rule>

Always follow these key guidelines:
- **Compostables**: In cities like Berkeley, San Francisco, and Portland, curbside composting often accepts food waste and certified compostable items like utensils, plates, or cups.
- **Plastic bags, film, utensils**: Usually trash, even if marked recyclable.
- **Greasy or soiled paper**: Trash unless a city accepts them in compost.
- **Clean aluminum foil or metal**: Often recyclable unless city explicitly excludes.
- **Uncertain?** Do NOT say "check local guidelines" — make the best decision based on known rules for that city.

Item: {item}
Location: {location}
"""
)

# LLM Chain
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory
)

# Function to classify the item
def classify_item(item, location):
    response = chain.run(item=item, location=location)
    lines = response.lower().splitlines()
    label_line = next((l for l in lines if "classification:" in l), None)
    predicted = label_line.split(":")[1].strip() if label_line else "unknown"

    return {
        "response": response,
        "classification": predicted
    }

__all__ = ["classify_item", "memory"]
