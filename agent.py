### ========================================= MULTI-PHASE AI AGENT ========================================= ###

import os
from pydantic import BaseModel
from typing import List, Dict, Any

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, END

### ================= LLM ================= ###
ollama_llm = ChatOllama(model="llama3.1")

### ================= RESPONSE STRUCTURE ================= ###
class ResponseStructure(BaseModel):
    question: str
    answer: str
    sources: List[str]
    tools_used: List[str]

# Defining the Parser for the response structure #
response_parser = PydanticOutputParser(pydantic_object=ResponseStructure)

### ================= PROMPT TEMPLATE FRAMEWORK ================= ###
response_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant that provides accurate information and follows the ResponseStructure format.
            Include the answer, sources, and tools_used.
            Return ONLY the structured output.
            {format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=response_parser.get_format_instructions())


### ================= LLM INPUT vs OUTPUT AGENT STATE FRAMEWORK ================= ###
class AgentState(Dict):
    query: str
    output: Any


### ================= MODEL (LLM) NODE ================= ###
def model_node(state: AgentState):
    chain = response_prompt_template | ollama_llm | response_parser
    result = chain.invoke({"query": state["query"]})
    return {"output": result}


# ================= BUILD GRAPH ================= #
graph = StateGraph(AgentState)
graph.add_node("model", model_node)
graph.set_entry_point("model")
graph.add_edge("model", END)

agent = graph.compile()


# ================= RUN AGENT ================= #
query = input("Welcome! You can ask me anything: ")

response = agent.invoke({"query": query})

print(response["output"])