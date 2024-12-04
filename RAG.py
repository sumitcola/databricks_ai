# Databricks notebook source
# MAGIC %md
# MAGIC #### Note please uplaod a pdf with name "my_file" in same directory as notebook then only this notebook will run and will answer your question out of pdf.

# COMMAND ----------

pip install faiss-cpu PyPDF2 sentence-transformers transformers numpy

# COMMAND ----------

import os

# Define the directory where the file should be uploaded
upload_directory = '/Workspace/Users/sumitraja2016@gmail.com/Assignment/RAG assignment/'

# Specify the file name you are expecting
expected_file = 'my_file.pdf'

# Wait until the user uploads the file
print(f"Please upload the file '{expected_file}' to the directory {upload_directory} and press Enter to continue.")

# Wait for user input to start
input("Press Enter after uploading the file...")

# Check if the file exists in the specified directory
file_path = os.path.join(upload_directory, expected_file)

while not os.path.exists(file_path):
    print(file_path)
    print(f"File '{expected_file}' not found. Please upload the file to {upload_directory}.")
    input("Press Enter after uploading the file...")
    file_path = os.path.join(upload_directory, expected_file)

print(f"File '{expected_file}' has been uploaded successfully. Proceeding with the next steps...")


# COMMAND ----------

# Import necessary libraries
import os
import shutil
import PyPDF2
import numpy as np
import faiss
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# pdf_path = "/Workspace/Users/sumitraja2016@gmail.com/Assignment/RAG assignment/p165.pdf"
# Load the embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Function to parse PDF and split text into smaller chunks
def extract_text_from_pdf(pdf_file_path):
    reader = PyPDF2.PdfReader(pdf_file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    overlap = 800  # Overlapping characters
    chunk_size = 1200  # Size of each chunk
    chunks = []
    if len(text) < chunk_size:
        chunks.append(text)
    else:
        chunks = [text[i:i + chunk_size] for i in range(0, len(text) - chunk_size + 1, chunk_size - overlap)]
    return chunks

# Function to generate embeddings from document chunks
def generate_embeddings(chunks):
    embeddings = embedder.encode(chunks)
    return embeddings

# Function to create FAISS index and add embeddings
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

# Function to perform similarity search using FAISS
def perform_similarity_search(index, user_question):
    query_embedding = embedder.encode([user_question])
    query_embedding = np.array(query_embedding).reshape(1, -1)
    k = 2  # Number of nearest neighbors to retrieve
    distances, indices = index.search(query_embedding, k=k)
    return indices

# Function to summarize the selected chunk of text
def summarize_text(chunk):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(chunk, max_length=1000, min_length=50, do_sample=False)
    return summary[0]['summary_text']

# Main workflow
def main():
    # Step 1: Extract text from PDF
    chunks = extract_text_from_pdf(file_path)
    print(f"Extracted {len(chunks)} chunks from the PDF.")

    # Step 2: Generate embeddings
    embeddings = generate_embeddings(chunks)

    # Step 3: Create FAISS index
    index = create_faiss_index(embeddings)

    while True:
        # Get the user question
        user_question = input("Ask a question from uploaded files (type 'STOP' to exit): ")
        if user_question.lower() == 'stop':
            print("Session ended.")
            break

        # Step 4: Perform similarity search
        indices = perform_similarity_search(index, user_question)

        # Step 5: Retrieve the top chunk
        top_chunk_index = indices[0][0]
        top_chunk_text = chunks[top_chunk_index]

        # Step 6: Summarize the top chunk
        summary = summarize_text(top_chunk_text)
        print(f"Answer: \n\n\n\n\n\n\n\n\n{summary}\n\n\n\n\n\n\n\n\n")

# Run the main function
main()


# COMMAND ----------


