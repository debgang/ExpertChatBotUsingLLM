# ExpertChatBotUsingLLM
Flask based chatbot developed for all LLM flows

## Chatbot for Expert Data Analysis - Using Agentic Flow
- This chatbot is based on agent which runs queries on Pandas dataframe
- User is required to provide the path to a csv or excel file.
- User can ask any questions for data analysis,data visualization,data summarization on the dataset.
- The Agent will execute tools to run dataframe queries on the datarame and return relevent results.
- The input before passing on to agent is passed through intent classifier to understant the viability of the request.
- One user wishes to exit from the conversation the expert agent will generate a full summary report from all the previous conversations

## Chatbot for Restaurant reccomendation - Using LLM calls
- This chatbot is based on simple LLM calls used to recommend restaurants in Bangalore based on user queries
- The restaurant dataset is obtained from Kaggle https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants
- The inputs and outputs are moderator checked and passed on through various layers of LLM calls before making the final recommendations

## Chatbot for Insurance Assistance - Using RAG
- This chatbot refers to an insurance document which user uploads and answers to the queries of users
- chromadb is used as vector store to store the embeddings of the doument provided.
- The document is splitted per page and then embedded using sentence-transformers/all-MiniLM-L6-v2
- Relevent pages are obtained from the vector store based on similarity search and passed on to reranker to get the reranked documents.
- Finally the result of reranker is passed on to LLM to generate the final response of the query


## Setup instructions 
- Setup pyton environment. have used python3 for development
- From the wokspace folder run pip install -e .
- Once all the requirements are installed run start-flask-app to start the server.
