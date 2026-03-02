### ========================================= MULTI-PHASE AI AGENT ========================================= ###

import os
from pydantic import BaseModel
from typing import List, Dict, Any

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, END

from tools import web_search_tool
import json

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


### ================= LLM INPUT vs OUTPUT AGENT STATE FRAMEWORK ================= ###
"""This is where you do your tool calling and other operations to get the output in the format defined in the ResponseStructure class."""

class AgentState(Dict):
    query: str
    output: Any
    tool_result: str | None
    tool_call: dict | None


### DEFINING THE TOOL CALLING PROMPT TEMPLATE - TELL THE LLM HOW TO CALL THE TOOL ###

# tool_calling_instructions = """
# You are an agent that can call tools using this JSON format:
# {{
#   "tool": "<tool_name>",
#   "input":  "<input string>"
# }}

# If no tool is needed, respond with the answer in the format defined in the ResponseStructure class.
# """

### ================= UPDATE THE ORIGINAL PROMPT TEMPLATE FRAMEWORK ================= ###
"""
response_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system", tool_calling_instructions + "{format_instructions}"
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=response_parser.get_format_instructions())
"""

response_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer using the ResponseStructure format.\n{format_instructions}",
        ),
        ("human", "{query}"),
    ]
).partial(format_instructions=response_parser.get_format_instructions())


### ================= UPDATE THE MODEL (LLM) NODE WITH TOOL-AWARE NODE ================= ###

def model_node(state: AgentState):
    # RUN PROMPT + LLM TO GET RAW RESPONSE

    RAW = (response_prompt_template | ollama_llm).invoke({"query": state["query"]})

    # GET THE TEXT CONTENT FROM THE LLM

    if hasattr(RAW, "content"):
        raw_msg = RAW.content
    else:
        raw_msg = str(RAW)

    # TRY TO PARSE AS A TOOL CALL
    try:
        DATA = json.loads(raw_msg)
        if isinstance(DATA, dict) and "tool" in DATA and "input" in DATA:
            return {"tool_call": DATA}
    except Exception:
        pass

    # OTHERWISE PARSE AS FINAL STRUCTURED ANSWER NORMALLY
    parsed = response_parser.invoke(raw_msg)
    return {"output": parsed}


### ================= CREATING THE WEB SEARCH TOOL NODE ================= ###

def web_search_tool_node(state: AgentState):
    tool_name = state["tool_call"]["tool"]
    tool_input = state["tool_call"]["input"]

    if tool_name == "WebSearch":
        result = web_search_tool.run(tool_input)
        return {"tool_result": result}
    else:    
        return {"tool_result": f"Unknown tool: {tool_name}"}


### ========= UPDATE THE FINAL LLM NODE TO HANDLE TOOL CALLS AND EXECUTE TOOLS ========= ###

def final_model_node(state: AgentState):
    # Build the query the model should answer
    if state.get("tool_result"):
        query_with_tool = f"{state['query']}\nTool_Result: {state['tool_result']}"
    else:
        query_with_tool = state["query"]

    chain = response_prompt_template | ollama_llm | response_parser
    result = chain.invoke({"query": query_with_tool})
    return {"output": result}


### ================= UPDATE THE LANGGRAPH BUILD AGENT GRAPH ================= ###
graph = StateGraph(AgentState)

graph.add_node("model", model_node)
graph.add_node("web_search_tool", web_search_tool_node)
graph.add_node("final_model", final_model_node)

graph.set_entry_point("model")

# ROUTE: LLM -> TOOL -> TOOL (IF TOOL_CALL EXISTS)

graph.add_conditional_edges(
    "model",
    lambda state: "tool" if state.get("tool_call") else "final_model",
    {
        "tool": "web_search_tool",
        "final_model": "final_model"
    }
)

# AFTER TOOL -> FINAL MODEL -> END
graph.add_edge("web_search_tool", "final_model")
graph.add_edge("final_model", END)

agent = graph.compile()


# ================= RUN AGENT ================= #
query = input("Welcome! You can ask me anything: ")

response = agent.invoke({"query": query})

print(response["output"])

#print(response["output"].answer)

# LLM -> TOOL -> FINAL_LLM -> END & LLM -> FINAL_LLM -> END

