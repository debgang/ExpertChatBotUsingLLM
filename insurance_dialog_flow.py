from pdf_processor import *
from insurance_dialog_flow import *
from embeddings import *
from generative_search import *

def insurance_dialog_flow(conversations):
    pdf_path = conversations["input_queue"].get(30)
    chunking_strategy = 'page'
    chunk_size = 512
    lines = 20
    insurannce_df = convert_pdf_to_dataframe(pdf_path, chunking_strategy, chunk_size, lines)
    ##Add embeddings
    model = 'sentence-transformers/all-MiniLM-L6-v2'     
    insurance_collection = addEmbeddings(insurannce_df, model)
    conversations["output_queue"].put("Successfully analysed the PDF. Enter your query or type 'exit' to end the conversation")
    user_input = ''
    while(True):
        user_input = conversations["input_queue"].get(30)
        if (user_input == 'exit'):
            conversations["output_queue"].put("Thank you for using the Insurance Assistant!")
            break
        results_df = searchEmbeddingsWithCache(user_input, insurance_collection, model)
        print("RAG results are:", results_df)
        reranker_results = reranker(results_df, user_input)
        print("Reranked results", reranker_results)
        response = generate_response(user_input, reranker_results)
        conversations["output_queue"].put("Response:" + response)
      
    