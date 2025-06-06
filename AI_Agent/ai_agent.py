# ai_agent.py
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import os

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

# Add memory to track past interactions
memory = ConversationBufferMemory(
    memory_key="history",
    return_messages=True,
    input_key="item"  
)

# Local recycling rules prompt
rules_prompt = """
You are a Recycling Classification Assistant. Only consider curbside rules. 
If a material is not accepted in curbside bins in the provided location, classify it as 'trash' or 'compost' depending on what it is. 
Do not recommend general advice like "check local guidelines."

Item: {item}
Location: {location}
Respond in the format:
classification: <trash/recycle/compost>
reason: <justification>
"""

prompt = ChatPromptTemplate.from_template(rules_prompt)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

def classify_item(item, location):
    response = chain.run(item=item, location=location)
    lines = response.lower().splitlines()
    label_line = next((l for l in lines if "classification:" in l), None)
    predicted = label_line.split(":")[1].strip() if label_line else "unknown"
    return {
        "response": response,
        "classification": predicted
    }
