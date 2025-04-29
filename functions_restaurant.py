import pandas as pd
# Import the libraries
import os, json, ast
import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt
# Set the display width to control the output width
pd.set_option('display.width', 100)
deployment = os.environ.get("AZURE_DEPLOYMENT_NAME")
api_version = os.environ.get("AZURE_API_VERSION")
from langchain_openai import AzureChatOpenAI
from fuzzywuzzy import fuzz


def initialize_conversation():
    '''
    Returns a list [{"role": "system", "content": system_message}]
    '''

    delimiter = "####"

    example_user_dict = {
                            "Location" : "Whitefield",
                            "Online Order Accepted" : "Yes",
                            "Table booking allowed" : "Yes",
                            "Cuisine" : "North Indian",
                            "Restaurant type" : "Casual Dining",
                            "Rating" : "Medium",
                            "Average cost for 2" : "800"

                        }

    example_user_req =  {
                            "Location" : "-",
                            "Online Order Accepted" : "Yes/No",
                            "Table booking allowed" : "Yes/No",
                            "Cuisine" : "-",
                            "Restaurant type" : "-",
                            "Rating" : "Low/Medium/High/Not Important",
                            "Average cost for 2" : "-"

                        }

    system_message = f"""
    You are an intelligent expert in recommending restaurants in Bangalore location.
    You need to ask relevant questions and understand the requirements of user by analysing the user's responses.
    You final objective is to fill the values for the different keys ('Location','Online Order Accepted','Table booking allowed','Cuisine','Restaurant type','Rating','Average cost for 2') in the python dictionary and be confident of the values.
    These key value pairs define the user's profile.
    The python dictionary looks like this
    {example_user_dict}
    The value for location should be extracted from user response.It should be some location in Bangalore like Whitefield,Kormangla,BTM etc.If user is confortable on any location in bangalore fill woth valye as Any
    You should take the intention of user for online ordering and table booking and fill the values accordingly.
    You should take the cuisine preference from user for example North Indian,South Indian,Chinese,Continental and fill the value of Cuisine key.
    You should take the restaurant preference from user for example Casual Dining,Fine Dining,Bar,Cafe  etc and fill the value of Restaurant type key.
    You should consider user prefernce of Restaurant rating. If  user provides a minimum rating of 3 assign it as MEDIUM.If user provides a minimum rating of 4 and above assign it as HIGH.If User does not mention anything about rating, it should be considered as Not Important.
    The value for 'Average cost for 2' should be a numerical value extracted from the user's response.
    All the values in the example dictionary are only representative values.
    {delimiter}
    Here are some instructions around the values for the different keys. If you do not follow this, you'll be heavily penalised:
    - All vaues should be in the format as mentioned in the example dictionary.
    - The value for 'Average cost for 2' should be a greater that equal to 100 .If the user says less than that, please mention that there are no Resuaturants  in that range
    - Do not randomly assign values to any of the keys.
    - The values need to be inferred from the user's response.
    {delimiter}

    To fill the dictionary, you need to have the following chain of thoughts:
    Follow the chain-of-thoughts below and only output the final updated python dictionary for the keys as described in {example_user_req}. \n
    {delimiter}
    Thought 1: Ask a question to understand the user's requirements while selecting appropriate restaurant. \n
    If their primary requirement is not clear ask followup questions to understand their needs.
    You are trying to fill the values of all the keys {('Location','Online Order Accepted','Table booking allowed','Cuisine','Restaurant type','Rating','Average cost for 2')} in the python dictionary by understanding the user requirements.
    Identify the keys for which you can fill the values confidently using the understanding. \n
    Remember the instructions around the values for the different keys.
    If the necessary information has been extracted, only then proceed to the next step. \n
    Otherwise, rephrase the question to capture their requirements clearly. \n

    {delimiter}
    Thought 2: Now, you are trying to fill the values for the rest of the keys which you couldn't in the previous step.
    Remember the instructions around the values for the different keys.
    Ask questions you might have for all the keys to strengthen your understanding of the user's requirement.
    If yes, move to the next Thought. If no, ask question on the keys whose values you are unsure of. \n
    It is a good practice to ask question with a sound logic as opposed to directly citing the key you want to understand value for.
    {delimiter}

    {delimiter}
    Thought 3: Check if you have correctly updated the values for the different keys in the python dictionary.
    If you are not confident about any of the values, ask clarifying questions.
    {delimiter}

    {delimiter}
    Here is a sample conversation between the user and assistant:
    User: "Hi, I am an user looking for a restaurant in Bangalore, preferably in Whitefield"
    Assistant: "Great! We have many restaurants in Whitefield. Do you prefer online ordering and table booking?"
    User: "Yes, I prefer online ordering and table booking."
    Assistant: "Thank you for providing that information. What type of cuisine do you prefer(North Indian/South Indian/Continental/Chinese etc)?"
    User: "I would prefer North Indian cuisine."
    Assistant: "Thank you for the information. What type of restaurant do you prefer(Fine Dining,Casual Dining,Street food,Quick Bites,Dhaba)?"
    User: "I would prefer Casual dining"
    Assistant:"Thank you for the information. Do you have any preference for the mininimum rating for restaurant? If yes then provide minimum rating in the range of 2 to 5"
    User: "Yes a minimum rating of 3 is preferable"
    Assistant: "Thank you for the information. What is your approximate budget for 2 people?"
    User: "I am not looking to spend more than 800 INR"
    Assistant: "{example_user_dict}"
    {delimiter}

    Start with a short welcome message and encourage the user to share their requirements.
    """
    conversation = [{"role": "system", "content": system_message}]
    # conversation = system_message
    return conversation

# Define a Chat Completions API call
# Retry up to 6 times with exponential backoff, starting at 1 second and maxing out at 20 seconds delay
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_chat_completions(input, json_format = False):
   
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

def moderation_check(user_input):
    # Call the OpenAI API to perform moderation on the user's input.
    llm = AzureChatOpenAI(
    azure_deployment=deployment,  # or your deployment
    api_version=api_version,  # or your api version
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
    )
    aimsg = llm.invoke(user_input)
    filtered_results = aimsg.response_metadata.get('prompt_filter_results')[0].get('content_filter_results')
    # Extract the moderation result from the API response.
    moderation_output = filtered_results.get('self_harm').get('severity') == 'safe' and filtered_results.get('hate').get('severity') == 'safe' and filtered_results.get('violence').get('severity') == 'safe' and filtered_results.get('sexual').get('severity') == 'safe'
    # Check if the input was flagged by the moderation system.
    if moderation_output == False:
        # If flagged, return "Flagged"
        return "Flagged"
    else:
        # If not flagged, return "Not Flagged"
        return "Not Flagged"
    

def data_sufficieny_layer(response_assistant):

    delimiter = "####"
    prompt = f"""
    You are a senior evaluator who has an eye for detail.The input text will contain a user requirement captured through 7 keys.
    You are provided an input. You need to evaluate if the input text has the following keys:
    {
        {
            "Location" : "<some location in Bangalore>",
            "Online Order Accepted" : "<Yes/No>",
            "Table booking allowed" : "<Yes/No>",
            "Cuisine" : "<Some cuisine type>",
            "Restaurant type" : "<Any restaurant type>",
            "Rating" : "<Low/Medium/High/Not Important>",
            "Average cost for 2" : "<some numeric value>|800"

        }
    }
    Please check if the keys are present in the input text. If any of the keys are missing, you need to ask the user for the missing information.
    The 'Average cost for 2' key can take only a numerical value.
    Next you need to evaluate if the keys have the the values filled correctly.
    Only output a one-word string in JSON format at the key 'result' - Yes/No.
    If user does'nt provide any value for the keys, then use the default value as 'Not Provided'.
    Thought 1 - Output a string 'Yes' if the values are correctly filled for all keys, otherwise output 'No'.
    Thought 2 - If the answer is No, mention the reason in the key 'reason'.
    THought 3 - Think carefully before the answering.
    """

    messages=[{"role": "system", "content":prompt },
              {"role": "user", "content":f"""Here is the input: {response_assistant}""" }]

    response = get_chat_completions(messages,json_format=True)


    return response

def dictionary_present(response):
    delimiter = "####"

    user_req = {
                            "Location" : "Whitefield",
                            "Online Order Accepted" : "Yes",
                            "Table booking allowed" : "Yes",
                            "Cuisine" : "North Indian",
                            "Restaurant type" : "Casual Dining",
                            "Rating" : "Medium",
                            "Average cost for 2" : "800"

               }

    prompt = f"""You are a python expert. You are provided an input.
            You have to check if there is a python dictionary present in the string.
            It will have the following format {user_req}.
            Your task is to just extract the relevant values from the input and return only the python dictionary in JSON format.
            The output should match the format as {user_req}.

            {delimiter}
            Make sure that the value of budget is also present in the user input. ###
            The output should contain the exact keys and values as present in the input.
            Ensure the keys and values are in the given format:
            {
                {
                            "Location" : "<some place in Bangalore>",
                            "Online Order Accepted" : "Yes/No",
                            "Table booking allowed" : "Yes/No",
                            "Cuisine" : "<some cuisine>|Not provided",
                            "Restaurant type" : "<some restaurant type>|Not provided",
                            "Rating" : "High/Medium/Low/Not Important",
                            "Average cost for 2" : "<some numerical value>|Not provided"

               }
            }
            Here are some sample input output pairs for better understanding:
            {delimiter}
            input 1: - Location: Whitefield, Online Order Accepted: Yes, Table booking allowed: Yes, Cuisine: North Indian, Restaurant type: Casual Dining, Rating: Medium, Average cost for 2: 800
            output 1: {{
                            "Location" : "Whitefield",
                            "Online Order Accepted" : "Yes",
                            "Table booking allowed" : "Yes",
                            "Cuisine" : "North Indian",
                            "Restaurant type" : "Casual Dining",
                            "Rating" : "Medium",
                            "Average cost for 2" : "800"

                       }}

            input 2: Location: Kormangla, Online Order Accepted: No, Table booking allowed: Yes, Cuisine: Continental, Restaurant type: Fine Dining, Rating: Medium, Average cost for 2: 200
            output 2: {{
                            "Location" : "Kormangla",
                            "Online Order Accepted" : "No",
                            "Table booking allowed" : "Yes",
                            "Cuisine" : "Continental",
                            "Restaurant type" : "Fine Dining",
                            "Rating" : "Medium",
                            "Average cost for 2" : "2000"

                       }}

            input 3: Here is your requirements  'Location': 'BTM', 'Online Order Accepted': 'Yes', 'Table booking allowed': 'Yes', 'Cuisine': 'North Indian', 'Restaurant type': 'Casual Dining', 'Rating': 'Medium,' 'Average cost for 2': 800
            output 3: {{
                            "Location" : "BTM",
                            "Online Order Accepted" : "Yes",
                            "Table booking allowed" : "Yes",
                            "Cuisine" : "North Indian",
                            "Restaurant type" : "Casual Dining",
                            "Rating" : "Medium",
                            "Average cost for 2" : "800"

                       }}
            {delimiter}
            """
    messages = [{"role": "system", "content":prompt },
                {"role": "user", "content":f"""Here is the user input: {response}""" }]

    confirmation = get_chat_completions(messages, json_format = True)

    return confirmation

def data_preparation_Layer(user_req_string):
    df = pd.read_csv("csv/restaurants.csv")
    df['Average cost for 2'] = df['Average cost for 2'].str.replace(',', '').astype(float)
    df['Rating'] = df['Rating'].str.replace('-', 'NaN')
    df = df[~df['Rating'].isna()]
    df = df[df['Rating'] != 'NEW']
    df['Rating'] = df['Rating'].astype(str).str.strip().astype(float)
    df['Rating'] = df['Rating'].apply(lambda x: 'Low' if x < 3 else ('Medium' if x < 4 else 'High'))
    user_req_location = user_req_string.get('Location')
    df_filtered = df[df['Location'].apply(lambda x: fuzz.ratio(x, user_req_location) >= 80)]
    df_filtered = df_filtered.sample(n=50, random_state=42)
    list_of_dicts = df_filtered.to_dict(orient='records')
    return list_of_dicts



def compare_restaurants_with_user_request(user_req_string):
   delimiter = "####"
   user_req1 = {
                            "Location" : "Whitefield",
                            "Online Order Accepted" : "Yes",
                            "Table booking allowed" : "No",
                            "Cuisine" : "Continental",
                            "Restaurant type" : "Casual Dining",
                            "Rating" : "Medium",
                            "Average cost for 2" : "800"

              }
   
   user_req2 = {
                            "Location" : "BTM",
                            "Online Order Accepted" : "Yes",
                            "Table booking allowed" : "No",
                            "Cuisine" : "North Indian",
                            "Restaurant type" : "Casual Dining",
                            "Rating" : "Medium",
                            "Average cost for 2" : "500"

                }
   prompt = f"""
        You are a restaurant expert who has to compare the user requirements with the restaurant data.You will be provided with an user requirement in form of python dictionary and a list of restaurants in form of list of dictionaries.
        You need to compare the user requirements with the restaurant data and recommend the top 3 restaurants that match the user requirements.
        {delimiter}
        The user requirement will be in the following format: {user_req_string}
        {delimiter}
        The restaurant data will be in the following format: {data_preparation_Layer(user_req_string)}
        {delimiter}
        Follow the chain of thoughts below and only output the final list of top 3 restaurants that match the user requirements.
        1. Find all the matching properties in restaurant data where the 'Location' in user requirement is a substring of 'Location' in the restaurant data. If no match is found then respond back saying 'No suitable match found in the given location'.
        2. For all the matches in the previous step find the best matching resuaturants with properties 'Cuisine', 'Restaurant type' in the user requirements from the restaurant data.Also assign a percentage match for each property based on the match. If the percentage match is less than 60% for the user requirement then do not consider that property for matching.If no suitable match is found the respond back saying 'No suitable match found'.
           If no information provided from user then skip this step.
        3. For the matches from the previous step find the resuaturants which matches the user criteria for 'Online Order Accepted' and 'Table Booking Allowed'.Ignore matching if it set as 'No' in the user requirement and the restaurant data as 'Yes'.If no match is found then respond back saying 'No suitable match found'.
        4. For all the matches in previous step find best matched restaurants based on the 'Rating' and 'Average cost for 2' in the user requirements.Do not consider Rating if Rating is 'Not Important' in the user requirements.The value for 'Average cost for 2' in the matched restaurant record  should be less than or equal to that in the user requirement.If no suitable match is found the respond back saying 'No suitable match found'.
           If no information provided from user then skip this matching. 
        {delimiter}
        Response guidelines:
        - The output should be a list of dictionaries with maximum  of top 3 restaurants that match the user requirements.
        - The output should contain all the properties from the data dictionary.
        - The output should contain "Not found" if no suitable match is found.
        {delimiter}
        - Below are some of the examples of user requirements and the expected output:
            {delimiter}
            User requirement: {user_req1}
            Expected output: [
                                {{'Name': 'Friends Diner', 'Address': '107/4, Pattandur Agrahara Road, Near ITPL Gate 3, Whitefield, Bangalore', 'Location': 'Whitefield', 'Online Order Accepted': 'Yes', 'Table booking allowed': 'No', 'Phone' : '080 28410777080 28410909', 'Cuisine': 'Continental', 'Restaurant type': 'Casual Dining', 'Rating': 'Medium', 'Average cost for 2': 700}}
                            ]

            User requirement: {user_req2}
            Expected output: [
                                {{'Name': 'Swadista Aahar', 'Address': '947, 16th Main Road, 2nd Stage, BTM, Bangalore', 'Location': 'Whitefield', 'Online Order Accepted': 'Yes', 'Table booking allowed': 'No', 'Phone' : '', 'Cuisine': 'Quick Bites', 'Restaurant type': 'South Indian, North Indian, Chinese, Street Food', 'Rating': 'High', 'Average cost for 2': 300}}
                            ]                
   """

   messages = {"role": "system", "content":prompt },
   response = get_chat_completions(messages, json_format = True)
   return response

def recommendation_validation(user_req, restaurant_recomendation):
    data_dict = restaurant_recomendation.get('result')
    valid_data = []
    if (data_dict != 'Not found'):
        for i in range(len(data_dict)):
            if data_dict[i]['Location'].find(user_req.get('Location')) !=-1 and data_dict[i]['Average cost for 2'] <= float(user_req.get('Average cost for 2')):
                valid_data.append(data_dict[i])

    return valid_data

def initialize_conv_reco(restaurant_recomendation):
    delimiter = "####"
    system_message = f"""
    You are an expert in summarizing and presenting the top 3 recommended restaurants to the user.
    You need to present the top 3 recommended restaurants to the user based on the recommendeations passed on to you in the form of a list of dictionaries.
    You should keep the useer's profile in mind while presenting the recommendations.
    If the user preference has Not Found then you should respond back saying 'No suitable match found, Please check with our customer support'.
    {delimiter}
    Start with a brief summary of each restaurant recommendation in the following format, in decreasing order of its rating:
    1. <Name> : <Location> <Address>, <Phone Number>, <Cuisine>, <Restaurant Type>, <Rating>, <Average cost for 2>
    2. <Name> : <Location> <Address>, <Phone Number>, <Cuisine>, <Restaurant Type>, <Rating>, <Average cost for 2> 

    """
    user_message = f""" These are the user's products: {restaurant_recomendation}"""
    conversation = [{"role": "system", "content": system_message },
                    {"role":"user","content":user_message}]
    # conversation_final = conversation[0]['content']
    return conversation


#user_req = {'Location': 'Whitefield', 'Online Order Accepted': 'No', 'Table booking allowed': 'No', 'Cuisine': 'Burger, Fast Food', 'Restaurant type': 'Food Court', 'Rating': 'Medium', 'Average cost for 2': '800'}

#print(data_preparation_Layer(user_req))