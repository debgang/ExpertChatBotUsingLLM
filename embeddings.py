import chromadb
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from sentence_transformers import CrossEncoder, util


def addEmbeddings(insurance_pdfs_data, embeddingModel='sentence-transformers/all-MiniLM-L6-v2'):
    client = chromadb.PersistentClient()
    model = SentenceTransformer(embeddingModel)
    insurance_collection = client.get_or_create_collection(name='RAG_on_Insurance')
    documents_list = insurance_pdfs_data["Page_Text"].tolist()
    metadata = [item for item in insurance_pdfs_data['Metadata'].tolist()]
    embeddings = model.encode(documents_list)
    insurance_collection.add(documents=documents_list,ids = [str(i) for i in range(0, len(documents_list))], metadatas=metadata, embeddings=embeddings)
    return insurance_collection

def searchEmbeddingsWithCache(query, insurance_collection, embeddingModel):
    client = chromadb.PersistentClient()
    cache_collection = client.get_or_create_collection(name='Insurance_Cache')
    model = SentenceTransformer(embeddingModel)
    query_embedding = model.encode(query)
    # Implementing Cache in Semantic Search
    cache_results = cache_collection.query(query_embeddings=query_embedding, n_results=1)

    # Set a threshold for cache search
    threshold = 0.2

    ids = []
    documents = []
    distances = []
    metadatas = []
    results_df = pd.DataFrame()


    # If the distance is greater than the threshold, then return the results from the main collection.

    if cache_results['distances'][0] == [] or cache_results['distances'][0][0] > threshold:
      # Query the collection against the user query and return the top 10 results
      results = insurance_collection.query(
      query_embeddings=query_embedding,
      n_results=10
      )

      # Store the query in cache_collection as document w.r.t to ChromaDB so that it can be embedded and searched against later
      # Store retrieved text, ids, distances and metadatas in cache_collection as metadatas, so that they can be fetched easily if a query indeed matches to a query in cache
      Keys = []
      Values = []

      for key, val in results.items():
        if val is None:
          continue
        for i in range(9):
          Keys.append(str(key)+str(i))
          Values.append(str(val[0][i]))


      cache_collection.add(
          documents= [query],
          ids = [query],  # Or if you want to assign integers as IDs 0,1,2,.., then you can use "len(cache_results['documents'])" as will return the no. of queries currently in the cache and assign the next digit to the new query."
          metadatas = dict(zip(Keys, Values))
      )


      result_dict = {'Metadatas': results['metadatas'][0], 'Documents': results['documents'][0], 'Distances': results['distances'][0], "IDs":results["ids"][0]}
      results_df = pd.DataFrame.from_dict(result_dict)
      return results_df


    # If the distance is, however, less than the threshold, you can return the results from cache

    elif cache_results['distances'][0][0] <= threshold:
      cache_result_dict = cache_results['metadatas'][0][0]

      # Loop through each inner list and then through the dictionary
      for key, value in cache_result_dict.items():
          if 'ids' in key:
              ids.append(value)
          elif 'documents' in key:
              documents.append(value)
          elif 'distances' in key:
              distances.append(value)
          elif 'metadatas' in key:
              metadatas.append(value)


      # Create a DataFrame
      results_df = pd.DataFrame({
        'IDs': ids,
        'Documents': documents,
        'Distances': distances,
        'Metadatas': metadatas
      })

    return results_df

def reranker(results_df, query):
   cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
   cross_inputs = [[query, response] for response in results_df['Documents']]
   cross_scores = cross_encoder.predict(cross_inputs)
   results_df['Cross_Scores'] = cross_scores
   results_df = results_df.sort_values(by='Cross_Scores', ascending=False)
   return results_df[["Documents", "Metadatas"]][:3] 

