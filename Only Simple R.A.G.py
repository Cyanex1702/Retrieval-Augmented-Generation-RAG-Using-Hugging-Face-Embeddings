import chromadb as chdb
from chromadb.utils import embedding_functions
chroma = chdb.PersistentClient(path="chroma_db_files")
collection_name = "Rag_using_HuggingFace_Embeddings"
collection = chroma.create_collection(name=collection_name)

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=20):
  chunks = []
  start = 0
  while start < len(text):
    end = start + chunk_size
    chunks.append(text[start:end])
    start = end
  return chunks

import os
directory = '/content/evaluating-student-writing/training-corpus.txt'
with open(directory, 'r') as file:
    text = file.read()

chunked_data = split_text_into_chunks(text)
print(len(chunked_data))

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

#embeddings = embed_model.get_text_embedding(chunked_data)
embeddings = embed_model.get_text_embedding_batch(chunked_data)
print(len(embeddings))
print(embeddings[:5])

ids = [f"chunk-{i}" for i in range(len(chunked_data))]
collection.upsert(
    documents=chunked_data,
    metadatas=None,
    embeddings=embeddings,
    ids=ids  # Use the generated IDs
)

from multiprocessing import context
def query_collection(query_text, n_results=2, top_k=3):
    query_embedding = embed_model.get_text_embedding(query_text)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results

query_collection(" Even though the planet is very similar to Earth there are challenges to get accurate data on the planet because of the many spacecrafts that were unable to withstand the harshness of the planet. As technology advanced, the author claims that more and more missions to Venus occured, and even around the time of World War II there was a spacecraft that survived in Venus conditions for about three weeks." , n_results=1)
