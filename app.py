from flask import Flask, jsonify, render_template, request
from flask import redirect
import flask
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import os

from langchain import hub
from langchain.chains import LLMChain
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import Tool,AgentExecutor, create_openai_functions_agent
from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.utilities import PythonREPL
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_community.text_loaders import TextLoader
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.utilities import OpenWeatherMapAPIWrapper

load_dotenv()
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY')
API_KEY = os.getenv("API_KEY")
OPENWEATHERMAP_API_KEY = os.getenv("WEATHER_API_KEY")


def get_llm(llm_type):
    # Całość zarządzania powinna zostać ostatecznie oddana orchestratorowi np w postaci open function calling agent 

    global AGENT_CHAIN
    if llm_type == "search":

        search = DuckDuckGoSearchRun()    
        tool = Tool(
                name="Search",
                func=search.run,
                description="Use when You want to search information for the plants for the user.",
            )
        prompt = """ You are a bot that can search for information in web. This is what You have to focus on all other question you Have to dimiss pointing your purpose
                    Your purpose is to help the user. Ensure the data You gather is answering the questions. You HAVE to return indicator that You didn't found anything if the search was unsuccessful or is not answering the question asked."""
    elif llm_type == "math":
        python_repl = PythonREPL()
        tool = Tool(
                name="python_repl",
                description="This is a working Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`. Remember To double check the code before running it checking it part by part before passing and executing it.",
                func=python_repl.run,
            )
        prompt = """ You are a MathGPT that can calculate math problems using Python code. This is what You have to focus on all other question you Have to dimiss pointing your purpose
                    Your purpose is to come up with a plan to calculate the math problem present by the user.
                    First spread the task into parts, then write the code for the task in Python, then ensure the code is written properly and it has a print() so that the result will be returned"""

    elif llm_type == "retriever":
        # the speech is just a quick example for testing purposes
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        loader = TextLoader("../../modules/state_of_the_union.txt")
        documents = loader.load()
        texts = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(texts, embeddings)
        retriever = db.as_retriever()
        retriever = create_retriever_tool(
                retriever,
                "search_state_of_union",
                "Searches and returns excerpts from the 2022 State of the Union.",
            )
        tool = create_retriever_tool(
                retriever,
                "search_state_of_union",
                "Searches and returns excerpts from the 2022 State of the Union.",
            )
    elif llm_type == "weather":
        weather = OpenWeatherMapAPIWrapper(api_key=OPENWEATHERMAP_API_KEY)
        # the tool should be customized and the default location gathered for example based on the IP address
        # def get_user_location():
        #   url = "http://ipwho.is/"
        #   geolocation = requests.get(url=url)
        #   geolocation = geolocation.json()
        #   return {'country': geolocation['country'], 'city': geolocation['city']}
        os.environ["OPENWEATHERMAP_API_KEY"] = OPENWEATHERMAP_API_KEY

        

    tools = [
       tool
    ]    
    prompt = hub.pull("hwchase17/openai-tools-agent")
    llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0, api_key=API_KEY)
    # adjust the instructions to fit the specific use case based on the tools used
    # prompt = prompt.partial(instructions)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_functions_agent(llm, tools, prompt)

    # dodanie memory wyciąganej na podstawie session id we flasku
    #chat_message_history = MongoDBChatMessageHistory(
    #session_id="test_session",
    #connection_string="mongodb://mongo_user:password123@mongo:27017",
    #database_name="my_db",
    #collection_name="chat_histories",
    #)

    #chat_message_history.add_user_message("Hello")
    #chat_message_history.add_ai_message("Hi")

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
    llm_type = request.args.get('llm_type')
    if not AGENT_CHAIN:        
        AGENT_CHAIN = get_llm(llm_type)
    llm_response = get_llm_response(user_text, AGENT_CHAIN)
    return jsonify({"text":llm_response})

@app.route("/")
def home():
    flask.session.pop('agent_chain', None)

    return render_template("chatbot.html")

app.run(debug = True)