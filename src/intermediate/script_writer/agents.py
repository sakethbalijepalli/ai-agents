import os

from crewai import Agent
from crewai.llm import LLM
from tools import exa_search_tool
from dotenv import load_dotenv

load_dotenv()

llm = LLM(
    model="openrouter/openai/gpt-4o",
    temperature=0.9,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    extra_body={
        "route": "fallback",
        "models": [
            "openai/gpt-4o",
            "deepseek/deepseek-r1",
            "meta-llama/llama-3.3-70b-instruct"
        ]
    }
)

# Agent 1: Content Explorer - Gathers information about the topic from the internet
content_explorer = Agent(
    role="content explorer",
    goal="Gather and provide latest information about the topic from internet",
    llm=llm,
    verbose=True,
    backstory=(
        "You are an expert researcher, who can gather detailed information about a topic.\
                  Gather at least 10 information."
    ),
    tools=[exa_search_tool()],
    cache=True,
    max_iter=5,
)

# Agent 2: Script Writer - Creates a script out of the information
script_writer = Agent(
    role="Script Writer",
    goal="With the details given to you create an interesting conversational script out of it",
    llm=llm,
    verbose=True,
    backstory=(
        "You are an expert in literature. You are very good in creating conversations with the given chain of information.\
        Tell as a script in 200 words."
    ),
)
