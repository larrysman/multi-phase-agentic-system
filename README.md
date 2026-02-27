### ================= INSTRUCTION TO ACCESS THE OPEN-SOURCE LANGUAGE MODEL =================== ###
You can use any of the available private LLM such as the OPENAI, ANTHROPIC etc but for the purpose of this piece of work and to allow to locally explore the ope-source free LLM, I worked with the OLLAMA LLM.

```bash
For Windows: visit: https://ollama.com/download ,downlod and install the ollama in the system OS directory.
Open your CMD of Terminal and pull the model locally to your system using: ollama pull llama3.1 or you select the model you desire.
You can also pull: ollama pull mistral
                   ollama pull deepseek-r1:7b
                   ollama pull qwen2.5
```

### ==================== CREATE AND ACTIVATE YOUR VIRTUAL ENVIROMENT ====================== ###
Create your local project directory - AI_AGENT and cd into the new directory and create the virtual envronment using:
```bash
$ python -m venv .venv
Activate the virtual environment
$ .venv\Scripts\activate
```

### ================== CREATE AND INSTALL REQUIREMENTS FILE ================= ###
Create the requirements.txt file and insert the all your dependencies such as:

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



