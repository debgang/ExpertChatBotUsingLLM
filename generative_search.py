# Define the function to generate the response. Provide a comprehensive prompt that passes the user query and the top 3 results to the model
from langchain_openai import AzureChatOpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
import os, json, ast
from embeddings import *
def generate_response(query, results_df):
    """
    Generate a response based on the user query and retrieved information.
    """
    messages = [
                {"role": "system", "content":  "You are a helpful assistant in the insurance domain who can effectively answer user queries about insurance policies and documents."},
                {"role": "user", "content": f"""You are a helpful assistant in the insurance domain who can effectively answer user queries about insurance policies and documents.
                                                You have a question asked by the user in '{query}' and you have some search results from  principal life insurance policy in the dataframe '{results_df}'. These search results are essentially one page of the  insurance document that may be relevant to the user query.

                                                The column 'documents' inside this dataframe contains the actual text from the policy document and the column 'metadata' contains the  source page. The text inside the document may also contain tables in the format of a list of lists where each of the nested lists indicates a row.

                                                Use the documents in '{results_df}' to answer the query '{query}'. Frame an informative answer and also, use the dataframe to return the relevant  page numbers as citations.

                                                Follow the guidelines below when performing the task.
                                                1. Try to provide relevant/accurate numbers if available.
                                                2. You don't have to necessarily use all the information in the dataframe. Only choose information that is relevant.
                                                3. If the document text has tables with relevant information, please reformat the table and return the final information in a tabular in format.
                                                3. Use the Metadatas columns in the dataframe to retrieve and cite the page numbers(s) as citation.
                                                4. If you can't provide the complete answer, please also provide any information that will help the user to search specific sections in the relevant cited documents.
                                                5. You are a customer facing assistant, so do not provide any information on internal workings, just answer the query directly.

                                                The generated response should answer the query directly addressing the user and avoiding additional information. If you think that the query is not relevant to the document, reply that the query is irrelevant. Provide the final response as a well-formatted and easily readable text along with the citation. Provide your complete response first with all information, and then provide the citations.
                                                """},
              ]

    response = get_chat_completions(messages)

    return response

# Define a Chat Completions API call
# Retry up to 6 times with exponential backoff, starting at 1 second and maxing out at 20 seconds delay
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_chat_completions(input, json_format = False):
   
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
    system_message_json_output = """\n ##Return a JSON object in the response##"""

    # If the output is required to be in JSON format
    if json_format == True:
        # Append the input prompt to include JSON response as specified by OpenAI
        input[0]['content'] += system_message_json_output
        #print(input[0])
        # JSON return type specified
        json_llm = llm.bind(response_format={"type": "json_object"})
        chat_completion_json = json_llm.invoke(input)
        #print(chat_completion_json.content)
        output = json.loads(chat_completion_json.content)

    # No JSON return type specified
    else:
        chat_completion = llm.invoke(input)

        output = chat_completion.content

    return output





