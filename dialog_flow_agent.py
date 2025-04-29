from langchain_agent import get_pandas_agent_response,get_pandas_agent,pretty_print_response
from llm_helper import get_final_report,get_response_summary,intent_classifier
import os

def dialogue_mgmt_system(conversations):
   # print("Welcome to your data analysis assistant!!")
    #print("You can simulate a conversation with your data analysis assistant")
    #print("You can ask questions about the dataset and get the analysis done")
    dataset_description = conversations["input_queue"].get(30)
    #print("==================================================================")
    conversations["output_queue"].put("Enter the filepath(csv) for the dataset:")
    filename = conversations["input_queue"].get(30)
    if not os.path.exists(filename):
        conversations["output_queue"].put("File not found. Please enter the correct file path")
        return
    agent = get_pandas_agent(filename)
    conversations["output_queue"].put("Agent created successfully for dataset.Enter your query or type 'exit' to end the conversation and generate the final report:")
    dict = {}
    while(True):
        user_input = conversations["input_queue"].get(30)
        if (user_input == 'exit'):
            final_message = "Thank you for using the Data Analysis Assistant!"
            if dict:
                 final_message += "\n" + get_final_report(dict)
            conversations["output_queue"].put(final_message)     
            break
        intent = intent_classifier(user_input, dataset_description)
        ##print("Intent of the query:",intent)
        if intent == "Not Applicable":
            conversations["output_queue"].put("The intent of the query could not be classified. Please ask a valid question")
            continue
        query = "You are a {intent} expert.Use all rows in the provided dataframe for the given question {user_input}"
        query = query.format(intent=intent,user_input=user_input)
        response = get_pandas_agent_response(agent, user_input)
        #pretty_print_response(user_input, response)
        #print("Brief summary of the response:")
        conversations["output_queue"].put(get_response_summary(user_input, response["output"]))
        dict[user_input] = response["output"]
