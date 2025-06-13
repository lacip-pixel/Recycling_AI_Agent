# ai_agent.py

from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

# Conversation memory to support ongoing context (optional)
memory = ConversationBufferMemory(
    memory_key="history",
    return_messages=True,
    input_key="item"
)

# Refined prompt template
prompt = PromptTemplate(
    input_variables=["item", "location"],
    template="""
You are a Recycling Classification Assistant.

Your task is to classify household waste items using the **curbside collection rules** for the specified U.S. city. Make decisions based on known local policies.

Use ONLY one of these labels:
- recycle
- compost
- trash

Follow these instructions:
- Be aware of city-specific exceptions. For example, some cities allow compostable utensils or soiled paper, while others do not.
- Plastic bags, plastic utensils, packaging film, chip bags: usually **trash**.
- Shredded paper: often **trash** unless city explicitly allows composting.
- Greasy paper: only **compost** if the city program permits it.
- Compostable plates/cups/utensils: only **compost** in cities like San Francisco, Berkeley, and Portland.
- Clean glass, aluminum, tin: usually **recyclable**.
- Hazardous items (batteries, expired meds): **trash** unless city accepts special handling — never recycle.

NEVER suggest “check local guidelines.” Always choose the best classification based on known policies.

Respond in **exactly** this format:
classification: <recycle / compost / trash>  
reason: <brief city-specific explanation>

Item: {item}  
Location: {location}
"""
)

# LLM Chain setup
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory
)

# Classification function
def classify_item(item, location):
    response = chain.run(item=item, location=location)
    lines = response.lower().splitlines()
    label_line = next((l for l in lines if "classification:" in l), None)
    predicted = label_line.split(":")[1].strip() if label_line else "unknown"

    return {
        "response": response,
        "classification": predicted
    }

# For import
__all__ = ["classify_item", "memory"]
