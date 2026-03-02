
# MULTI-PHASE AGENTIC SYSTEM
**This project explore the compability of the LangChain and LangGraph and the Ollama in developing a multi-phase agentic system.**

##### INSTRUCTION TO ACCESS THE OPEN-SOURCE LANGUAGE MODEL ###
- *You can use any of the available private LLM such as the OPENAI, ANTHROPIC etc but for the purpose of this piece of work and to allow to locally explore the ope-source free LLM, I worked with the OLLAMA LLM.*

```bash
For Windows: visit: https://ollama.com/download ,downlod and install the ollama in the system OS directory.
Open your CMD of Terminal and pull the model locally to your system using: ollama pull llama3.1 or you select the model you desire.
You can also pull: ollama pull mistral
                   ollama pull deepseek-r1:7b
                   ollama pull qwen2.5
```

##### CREATE AND ACTIVATE YOUR VIRTUAL ENVIROMENT ###
**Create your local project directory - AI_AGENT and cd into the new directory and create the virtual envronment using:**

```bash
$ python -m venv .venv
Activate the virtual environment
$ .venv\Scripts\activate
```

##### CREATE AND INSTALL REQUIREMENTS FILE ###
**Create the requirements.txt file and insert the all your dependencies such as:**

`requirements.txt`
```bash
langchain
wikipedia
langchain-community
langchain-openai
langchain-anthropic
python-dotenv
Pydantic
langchain-ollama

Then run: $ pip install -r requirements.txt
```

##### INSTANTIATE THE LLM ###
```bash
ollama_llm = OllamaLLM(model="llama3.1")
response = ollama_llm.invoke("What is the capital of Nigeria?")
print(response)

ollama_llm = ChatOllama(model="llama3.1")
```
##### CREATE THE PYTHON CLASS ###
**The Python class is used to specify the type of content that the LLM will take as input and generate output. You will define all the fields you expect from the input/output of the LLM, something like structure of the LLM input and output.**

- **`LLM Response or Output Format Python class:`**
```bash
class ResponseStructure(BaseModel):
    question: str
    answer: str
    sources: List[str]
    tools_used: List[str]
```
`This class define how the agent forces the LLM to return a structured, machine-readable response instead of free-form text. This is central to making the agent predictable, debuggable, and compatible with downstream logic such as tool-calling, memory, or multi-step workflows integration.`

`This defines the exact schema the LLM must follow when generating output and it is a Pydantic model which means that; it enforces strict structure, validates LLM's output, convert the output into a Python object with typed fields, prevents hallucination or malformed responses and gives agent a predictable output format. To correctly utilize this, ensure to add a parser as the bridge between the LLM's text output and the Python object. It is best to us the PydanticOutputParser.`

- **`LLM Input Format Python class - The AGENT STATE:`**
```bash
class AgentState(Dict):
    query: str
    output: Any
```
`This class defines the data structure that flows through the agent graph and is one of the most important concepts in LangGraph because every node in the graph receives a state and returns a modified state. It is the shared memory or data container that moves through the agent's workflow so that each node in the LangGraph pipeline recives the current state, reads values from it, writes new values back into it, passes it to the next node. It simply replcaes the old AgentExecutor in the lower version of langchain version < 1.x.`

`It inherits from Dict because LangGraph requires the state to be a TypedDict-like structure and behaves like a Python dictionary at runtime but has type annotations so LangGraph knows the keys that exist and ensures each node recieves and returns the correct fields.`

`In the highr version of LangChain version >=1.x, we define the state explicitly, control how data flows and build the agent logic using a graph. This gives more power and clarity.`

##### CREATE FUNCTION FOR THE LLM NODE EXECUTION STEP ###
- `Core Execution Function`
```bash
def model_node(state: AgentState):
    chain = response_prompt_template | ollama_llm | response_parser
    result = chain.invoke({"query": state["query"]})
    return {"output": result}
```
`This function is the core execution step in the LangGraph agent and it defines how the agent uses the prompt, the local model (llm), and the Pydantic parser to produce a structure response. Having the understanding of this block is essential because LangGraph agents are built from nodes, and each node transforms the agent's state.`

`This node is a single step in the agent's computation graph and perform the following tasks: receives the current agent state (which contains the user query), runs the prompt, pass into the model (llm) and the parser pipeline, then returns a new state containing the structured output. It is the LangGraph equivalent of the old LangChain AgentExecutor.`

- `Breaking down the code`
```bash
1. def model_node(state: AgentState) -> defines a graph node that accepts the agent's state.
   state -> is a dictionary-like object defined in this project as a Python class: AgentState(Dict).
   When the graph runs, LangGraph passes the current state into this function.
```

```bash
2. chain = response_prompt_template | ollama_llm | response_parser -> This builds a runnable pipeline using the LangGraph's operator |.
   It then creates a 3-step chain and include:
   step 1: Prompt Template Injection
   response_prompt_template -> This injects the instructions, formatting rules, and the user query into the model (llm).
   step 2: Local Model (llm) Integration
   ollama_llm -> This is the model initiated and used for the project and this runs the model and produces raw text output.
   step 3: Pydantic Parser
   response_parser -> This convert the raw text into structured format.

   Summary: Prompt -> LLM -> Structured Ouptut
```

```bash
3. result = chain.invoke({"query": state["query"]}) -> This executes the chain using the following logic:
   - It takes the user's query from the state.
   - It feeds it into the chain.
   - It returns a Pydantic object of the type of the Python class defined in this project ResponseStructure.
```

```bash
4. return {"output": result} -> This returns a partial state update and LangGraph merges this into the global state and this is how agent produces its final ressult.
```

##### LANGGRAPH BUILD UP AND COMPILATION ###
**This is the core of how the LangGraph turns the agent into an executable workflow and it defines the graph (a directed flow of computation) where each node is a function and the edges define how the data moves between them.**

```bash
graph = StateGraph(AgentState)
graph.add_node("model", model_node)
graph.set_entry_point("model")
graph.add_edge("model", END)

agent = graph.compile()

This section is constructing a state machine that LangGraph will execute and replaces the old LangChain AgentExecutor.
```
*To understand this, I have explain each of the important parts:*
- **graph = StateGraph(AgentState):**
    
    *This creates a new computational graph whose state is defined by the Python class AgentState in this project and it contains:*
    - StateGraph is the LangGraph's core engine and it expects a typed dictionary describing what data flows through the graph and for this project, the state contains:
        - query: the user's input
        - output: the final structured response.
    
    **This means every node in the graph recieves and returns a dictionary with these keys.**

- **graph.add_node("model", model_node):**

    *This adds a node to the graph where `model` is the name of the node and `model_node` is the function that will run when this node is executed.*

    *A node is simply a Python function that takes the current state, performs some computation such as LLM call, tool call, routing, etc. and returns a partial update to the state.*

    **In this project, the function `model_node` reads the `state["query"]`, runs the `prompt -> LLM -> parser chain` and returns `{"output": result}` and this is the `heart of the agent`.**

- **graph.set_entry_point("model"):**

    *This tells the LangGraph to start the agent by running the `model` node first where every graph must have exactly one entry point. In this project, the agent is simple (one-step), the model node is the first and only step.*

- **graph.add_edge("model", END):**

    *This defines the flow of execution and it means after the `model node` finishes, stop the graph and `END` is a special terminal marker provided by LangGraph. For example, if multiple nodes are present such as tools, memory, routing, then it would need to add more edges. But this project is a simple one, so the agent is a single-step agent and so, it ends immediately.*

- **agent = graph.compile():**

    *This compiles the graph into an executable agent, validates the graph, optimizes it, create an object with `.invoke()` and `.stream()` methods.*

    *Then you invoke the agent on the query:*

    `query = input("Welcome! You can ask me anything: ")`
    
    `response = agent.invoke({"query": query})`

**The LangGraph will create the initial state: `{"query": query}`, run the `model_node`, update the state with `{"output": result}`, stop at `END` and returns the `final state`. This block is the replacement for the AgentExecutor in LangChain 1.x.**


##### ADDING PREBUILT TOOLS #####
- Tools are the things that the LLM or agent can use that we can either write ourself or we can bring them in from the LangChain Community Hub.

- In the course of the project, I integrated the multi-phase agents for the web_search_tool using duckduckgo and the customized integrated function using Python.

##### CONTINUOUS WORK #####

- Work in Progress...

**AUTHOR:**
- **Name:** `Olanrewaju Adegoke`











