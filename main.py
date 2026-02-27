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
from langchain_ollama import OllamaLLM

### ============== LOADING ENVIRONMENT VARIABLES ============== ###
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")


### ============== LOAD THE LLM MODEL ============== ###
"""llm1 = ChatOpenAI(model="gpt-4", temperature=0.7, max_tokens=2048, api_key=api_key)
llm2 = ChatAnthropic(model="claude-2", temperature=0.7, max_tokens=2048, api_key=anthropic_api_key)"""

"""llm1 = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=2048, api_key=api_key)
response = llm1.invoke("What is the capital of France?")
print(response)"""

"""llm2 = ChatAnthropic(model="claude-3-5-sonnet-20241022")
llm2 = ChatAnthropic(model="claude-3-5-sonnet-20241022", api_key=anthropic_api_key)
response = llm2.invoke("What is the capital of France?")
print(response)"""

##### LOADING AND INSTANTIATING OLLAMA #####
"""ollama_llm = ollama.Ollama(model="llama3.1")
response = ollama_llm.invoke("What is the capital of France?")
print(response)"""

ollama_llm = OllamaLLM(model="llama3.1")
response = ollama_llm.invoke("What is a Noun?")
print(response)

