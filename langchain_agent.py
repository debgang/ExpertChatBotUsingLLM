from langchain_openai import AzureChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
import os, json, ast
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
import matplotlib.pyplot as plt
from langchain.tools import ShellTool

## Helper function to pretty print the response
def pretty_print_response(question, response):
    print(f"Question: {question}")
    print("\n" + "=" * 80 + "\n")
    print(response["output"])  # Extract the 'output' key for pretty printing
    print("\n" + "=" * 80 + "\n")

## Initialize the LLM
def initialize_llm():
    deployment = os.environ.get("AZURE_DEPLOYMENT_NAME")
    api_version = os.environ.get("AZURE_API_VERSION")   
    llm = AzureChatOpenAI(
    azure_deployment= deployment,  # or your deployment
    api_version=api_version,  # or your api version
    temperature=0,
    max_tokens=None,
    timeout=None,
        max_retries=2
    )
    return llm

## Creates a pandas agent
def get_pandas_agent(filename):
    llm = initialize_llm()  
    df = pd.read_csv(filename)
    agent = create_pandas_dataframe_agent(llm=llm, df=df, verbose=True,  
    agent_type="tool-calling", allow_dangerous_code=True,  early_stopping_method="force", max_iterations=5)
    return agent

## Helper function to get the response summary for each response from agent
def get_pandas_agent_response(agent, query):
    response = agent.invoke(query)
    return response
'''
def get_shell_agent_response(query):
    llm = initialize_llm()
    shell_tool = ShellTool()
    tools = [shell_tool]
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    response = agent.run(query)
    return response
'''    

#get_shell_agent_response("Run the command using Git sh '/c/workspace/dganguly_cai/deployments/docker/icai_dev/start-cai.sh`")
     
    
