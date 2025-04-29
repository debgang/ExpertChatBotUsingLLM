from langchain_agent import initialize_llm
from langchain.prompts import ChatPromptTemplate

def get_chat_respose(domain, query):
    """
    Print response of LLM for the given query and response.
    """
    llm = initialize_llm()
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in {domain}"),
        ("human", "{query}")
    ])

    messages = chat_template.format_messages(query=query, domain=domain)
    response = llm.invoke(messages)
    return response.content

print(get_chat_respose("Cricket", "What is the average of Sachin Tendulkar."))