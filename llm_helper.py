from langchain_agent import initialize_llm
from langchain.prompts import ChatPromptTemplate

## Helper function to get the response summary for each response from agent
def get_response_summary(query, response):
    llm = initialize_llm()
    chat_template = ChatPromptTemplate.from_messages([
    ("system", """You are an expert summarizer of Agent responses on data analysis queries"""),
    ("human", """Provide a brief one line summary of  the response {response} for the query {query}.Summary should be helpful for data analysis tasks. """)
    ])

    messages = chat_template.format_messages(query=query, response=response)
    response =  llm.invoke(messages)
    return response.content

## Helper function to get the final report from the dictionary of responses
def get_final_report(dict):
    llm = initialize_llm()
    chat_template = ChatPromptTemplate.from_messages([
    ("system", """You are an expert in generating a detailed report for data analysis task.You will be provided with a python dictionary containing a pair of (query,response). 
     You need to generate a detailed report which provides real insights into the data"""),
    ("human", "Create a detailed summary for {dict}. The report should be in form of numbered lists""")
    ])
    messages = chat_template.format_messages(dict=dict)
    response =  llm.invoke(messages)
    return response.content

def intent_classifier(query, dataset_description):
    llm = initialize_llm()
    chat_template = ChatPromptTemplate.from_messages([
    ("system", """You are an expert in intent classification of user queries"""),
    ("human", """Classify the intent of the query {query}. The intent should be one of the following: data analysis, data visualization, data cleaning, data preprocessing, data modeling, data interpretation and data summarization"
               . The intent should be based on the dataset description {dataset_description}
              For example, if the query is 'What is the average age of the customers in the dataset?' and the dataset description is 'customer', the intent should be 'data analysis'
    If the query is 'Plot a bar chart for the age distribution of customers' and the dataset description is 'customer', the intent should be 'data visualization')
    If the query is 'Remove the missing values from the dataset' and the dataset description is 'customer', the intent should be 'data cleaning')
    If the intent cann be classified into any of the above categories the respond as 'Not Applicable' """
     )])

    messages = chat_template.format_messages(query=query, dataset_description=dataset_description)
    response =  llm.invoke(messages)
    return response.content