from __future__ import annotations

import os

from langchain.agents import create_agent
#from langchain_openai import ChatOpenAI
from langchain_openrouter import ChatOpenRouter
from langgraph.checkpoint.memory import InMemorySaver
from langchain_ollama import ChatOllama
from middleware import AgentLoggingMiddleware
from state import AgentState
from tools import (
    add_to_shortlist,
    get_product_details,
    load_category_info,
    search_products,
    view_shortlist,
)
from dotenv import load_dotenv
load_dotenv()
# api_key=os.getenv(key="OPENROUTER_API_KEY")
SYSTEM_PROMPT = """
You are a product catalog research agent.

Behavior rules:
- Use tools instead of guessing product facts.
- For product lookup, call search_products first.
- If search_products returns status='needs_clarification', present exactly the listed choices and ask the user to choose one by ID or exact name. Do not invent extra follow-up questions.
- If a product is auto-selected, you may summarize it and optionally call get_product_details if more detail is needed.
- Use view_shortlist when the user asks what has been saved.
- Use add_to_shortlist only when the user explicitly asks to save/add/select quantity.
- Use load_category_info when the user asks for deeper details about a category.
- Keep responses concise and operational.
""".strip()


def build_agent():
    #model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    #model_name=os.genenv("OPENROUTER_API_KEY")
    #model = ChatOpenAI(model=model_name, temperature=0)

#     model = ChatOpenRouter(
#         model="openai/gpt-oss-20b:free", 
#         api_key=os.getenv("OPENROUTER_API_KEY"),
#         base_url=os.getenv("OPENAI_API_BASE"),
# )
    model= ChatOllama(
            model="gpt-oss:120b-cloud",
            validate_model_on_init=True,

        )
    tools = [
        search_products,
        get_product_details,
        add_to_shortlist,
        view_shortlist,
        load_category_info,
    ]

    return create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        state_schema=AgentState,
        middleware=[AgentLoggingMiddleware()],
        checkpointer=InMemorySaver(),
    )
