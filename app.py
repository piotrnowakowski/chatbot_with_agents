from flask import Flask, jsonify, render_template, request
from flask_session import Session  # This is for server-side session management
from flask import redirect
import flask
from pathlib import Path
from datetime import datetime
import json
from tenacity import retry, wait_random_exponential, stop_after_attempt
from dotenv import load_dotenv
import os
import pandas as pd

from langchain import hub
from langchain.chains import LLMChain
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import Tool,AgentExecutor, create_openai_functions_agent
from langchain_openai import OpenAI, ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_experimental.utilities import PythonREPL
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool

load_dotenv()
app = Flask(__name__)
API_KEY = os.getenv("API_KEY")

def get_llm(user_data, plant_list):
    global AGENT_CHAIN
    search = DuckDuckGoSearchRun()
    python_repl = PythonREPL()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    
    #still need to add the data
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever()
    retriever = create_retriever_tool(
            retriever,
            "search_state_of_union",
            "Searches and returns excerpts from the 2022 State of the Union.",
        )
    

    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="Use when You want to search information for the plants for the user.",
        ),
        Tool(
            name="python_repl",
            description="This is a working Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`. Remember To double check the code before running it checking it part by part before passing and executing it.",
            func=python_repl.run,
        ),
        create_retriever_tool(
            retriever,
            "search_state_of_union",
            "Searches and returns excerpts from the 2022 State of the Union.",
        )
    ]    
    prompt = hub.pull("hwchase17/openai-tools-agent")
    llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0, api_key=API_KEY)
    #prompt.set_input_variables(input=prefix, chat_history="", agent_scratchpad="")
    prefix = "You are HelpGPT You need to answer the user question based on the data available to You by the websearch. Focus on factchecking and always admit if You didn't find the answer. For the user question, You can use the following format: 'User: What"
    prompt = prompt.partial(instructions=prefix)
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    AGENT_CHAIN = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
    )
    return AGENT_CHAIN

def get_llm_response(user_text, chain):
    text = chain.run(input=user_text)
    return text

@app.route("/get")
def get_bot_response():
    global AGENT_CHAIN
    user_text = request.args.get('msg')
    if not AGENT_CHAIN:        
        AGENT_CHAIN = get_llm()
    llm_response = get_llm_response(user_text, AGENT_CHAIN)
    return jsonify({"text":llm_response})

@app.route("/")
def home():
    flask.session.pop('agent_chain', None)

    return render_template("chatbot.html")

app.run(debug = True)