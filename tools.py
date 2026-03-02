##### =============== DEVELOPING TOOLS FOR COMPLEX AGENTS =============== #####
"""
In this section, we will explore how to develop tools that can be used by agents to perform specific tasks.
We will define a simple tool and integrate it into an agent's workflow.
This will allow the agent to call the tool when needed and use its output to generate a response.
We will also see how to structure the agent's response to include the answer, sources, and tools used.

We will start by defining a simple tool that can perform a specific task, such as fetching the current weather for a given location.
Then, we will integrate this tool into an agent's workflow, allowing the agent to call the tool when it receives a query that requires weather information.
Finally, we will structure the agent's response to include the answer, sources, and tools used in a clear and organized manner.
"""

"""
Tools like using the DuckDuckGo to search the web, One custom tool that will perform a specific task using Python function, wikipedia, WolframAlpha, etc.
can be used to enhance the capabilities of an agent.
In this section, we will explore how to develop tools that can be used by agents to perform specific tasks.
We will define a simple tool and integrate it into an agent's workflow.
This will allow the agent to call the tool when needed and use its output to generate a response.
We will also see how to structure the agent's response to include the answer, sources, and tools used.
"""

from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import StructuredTool
from datetime import datetime
import os


### ================= DUCKDUCKGO SEARCH TOOL ================= ###
web_search = DuckDuckGoSearchRun()
def web_search_fn(query: str):
    return web_search.run(query)

web_search_tool = StructuredTool.from_function(
    func=web_search_fn,
    name="WebSearch",
    description="Use this tool to search the web for information. Input should be a search query.",
)

"""
web_search_tool = Tool(
    name="WebSearch",
    description="Use this tool to search the web for information. Input should be a search query.",
    func=web_search.run,
)
"""

### ================ WIKIPEDIA TOOL ================= ###
wikipedia_api_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=2)
def wikipedia_search_fn(query: str):
    return wikipedia_api_wrapper.run(query)

wikipedia_tool = StructuredTool.from_function(
    func=wikipedia_search_fn,
    name="WikipediaSearch",
    description="Use this tool to search Wikipedia for information. Input should be a search query.",
)

wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=2)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)


### CUSTOM TOOL FOR CALLING A PYTHON FUNCTION ###

def save_to_text(data: str, filename: str = "multi_phase_agent_output.txt"):

    save_dir = r"C:\Users\Olanrewaju Adegoke\AI_AGENT\outputs"

    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Multi Phase Agent Output ---\nTimestamp: {timestamp}\nData: {data}\n\n"

    with open(filepath, "a", encoding="utf-8") as file:
        file.write(formatted_text)

    #return f"Data saved to {filename} successfully."
    return f"Data saved to {filepath} successfully."


### AFTER YOU DEFINE THE CUSTOM CALLING TOOL, ENSURE YOU WRAPPED IT IN A STRUCTURED TOOL SO THAT THE LLM CAN CALL IT PROPERLY ###
save_to_text_tool = StructuredTool.from_function(
    func=save_to_text,
    name="SaveToText",
    description="Use this tool to save data to a text file. Input should be the data to be saved and an optional filename.",
)
