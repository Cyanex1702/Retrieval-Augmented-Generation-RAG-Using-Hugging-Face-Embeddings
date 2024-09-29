# Retrieval-Augmented Generation (RAG) Using Hugging Face Embeddings

This project demonstrates how to implement a **Retrieval-Augmented Generation (RAG)** pipeline using **Hugging Face embeddings** and **ChromaDB** for efficient semantic search. The solution reads, processes, and embeds textual data, enabling a user to perform accurate and fast queries on the data.

## Features
- **Dataset Integration**: Load and process datasets from Hugging Face.
- **Text Chunking**: Split large text into manageable chunks for embedding.
- **Embeddings Generation**: Utilize Hugging Face embeddings (`BAAI/bge-base-en-v1.5`) to convert text chunks into vector representations.
- **ChromaDB Storage**: Store embeddings in ChromaDB for easy retrieval.
- **Semantic Search**: Query the stored data for relevant text based on a provided prompt using semantic similarity.

## Installation

Before running the notebook, ensure the necessary libraries are installed:

```bash
pip install chromadb
pip install llama-index
```

You also need to clone the required datasets from Hugging Face:

```bash
git clone https://huggingface.co/datasets/NahedAbdelgaber/evaluating-student-writing
git clone https://huggingface.co/datasets/transformersbook/emotion-train-split
```

## How It Works

1. **Load Datasets**: 
   - The notebook loads the "Evaluating Student Writing" dataset and splits the text into chunks for embedding.

2. **Embedding Creation**: 
   - Using the `BAAI/bge-base-en-v1.5` model, text chunks are converted into vector embeddings.

3. **ChromaDB Integration**: 
   - The generated embeddings, along with their corresponding text chunks, are stored in ChromaDB for persistence and later querying.

4. **Semantic Search**:
   - A query function is provided to search the vector database using a given input query. The relevant chunks are returned based on similarity to the query.

## Usage

To use the code, simply run the notebook after installing the dependencies and cloning the required datasets. The following command can be used to query the stored embeddings:

```python
query_collection("Your search query here", n_results=1)
```

This will return the most relevant text chunk based on the provided query.

## Example

```python
query_collection(
  "Even though the planet is very similar to Earth, there are challenges to get accurate data because of the harsh conditions on the planet.", 
  n_results=1
)
```

## Dependencies

- [ChromaDB](https://docs.trychroma.com/)
- [Hugging Face Embeddings](https://huggingface.co/models)
- [llama-index](https://github.com/jerryjliu/llama_index)

## Future Enhancements

- Improve the chunking mechanism for more flexible handling of overlapping sentences.
- Fine-tune the embedding model for more specific domain applications.
- Add support for multiple datasets.

## License

This repository is licensed under the MIT License.
