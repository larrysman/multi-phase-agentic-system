### ========================================= MULTI-PHASE AI AGENT ========================================= ###

### ============== IMPORTING NECESSARY LIBRARIES ============== ###
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.llms import ollama
from langchain_community.chat_models import ChatOllama
from langchain.tools import BaseTool, tool
from langchain_core.agents import create_react_agent
from langchain_ollama import OllamaLLM
from langchain.agents import create_tool_calling_agent
from langchain_core.agents import AgentExecutor


### ============== LOADING ENVIRONMENT VARIABLES ============== ###
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")


### ============== LOAD THE LLM MODEL ============== ###
"""
llm1 = ChatOpenAI(model="gpt-4", temperature=0.7, max_tokens=2048, api_key=api_key)
llm2 = ChatAnthropic(model="claude-2", temperature=0.7, max_tokens=2048, api_key=anthropic_api_key)
"""

"""
llm1 = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=2048, api_key=api_key)
response = llm1.invoke("What is the capital of France?")
print(response)
"""

"""
llm2 = ChatAnthropic(model="claude-3-5-sonnet-20241022")
llm2 = ChatAnthropic(model="claude-3-5-sonnet-20241022", api_key=anthropic_api_key)
response = llm2.invoke("What is the capital of France?")
print(response)
"""

##### LOADING AND INSTANTIATING OLLAMA #####
"""
ollama_llm = ollama.Ollama(model="llama3.1")
response = ollama_llm.invoke("What is the capital of France?")
print(response)
"""

"""
ollama_llm = OllamaLLM(model="llama3.1")

response = ollama_llm.invoke("What is the capital of Nigeria?")
print(response)
"""

ollama_llm = ChatOllama(model="llama3.1")
#response = ollama_llm.invoke("What is the capital of Nigeria?")
#print(response)

### ============== DEFINING THE OUTPUT PARSER ============== ###

##### Defining the response structure with Python class #####
class ResponseStructure(BaseModel):
    question: str
    answer: str
    sources: List[str]
    tools_used: List[str]

##### Defining the Parser for the response structure #####
response_parser = PydanticOutputParser(pydantic_object=ResponseStructure)

### ============== DEFINING THE PROMPT TEMPLATE FOR THE RESPONSE STRUCTURE ============== ###
response_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant that provides accurate information and follows the response structure defined in the ResponseStructure class.
            The response should include the answer to the question, a list of sources used, and a list of tools used.
            Wrap the response in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=response_parser.get_format_instructions())

### ============== CREATING AND RUNNIG A SIMPLE AGENT ============== ###
agent_calling = create_tool_calling_agent(
    llm=ollama_llm,
    tools=[],
    prompt=response_prompt_template,
)

agent_executor = AgentExecutor(agent=agent_calling, tools=[], verbose=True)
query = input("How can I help you today? ")
raw_response = agent_calling.invoke({"query": query})
print(raw_response)

structured_response = response_parser.parse(raw_response.get("output")[0]["text"])
print("\nStructured Response:")
print(structured_response)

try:
    structured_response = response_parser.parse(raw_response.get("output")[0]["text"])
except Exception as e:
    print(f"Error parsing response: {e}, Raw response: {raw_response.get('output')[0]['text']}")
