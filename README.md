#PDF Q&A Chatbot using Groq, Langchain, and Streamlit
This project implements a powerful PDF question-answering chatbot, utilizing Streamlit for the user interface, Langchain for document retrieval and processing, and Groq's LLM for natural language understanding. It allows users to upload PDF files, split them into smaller chunks, index them using FAISS vector store, and then ask questions based on the PDF contents. The chatbot provides precise answers by leveraging a retrieval-augmented generation (RAG) approach, combined with contextual chat history for more accurate responses.

#Features:
PDF Upload: Users can upload multiple PDF files, which are then processed and split into chunks.
FAISS Vector Store: The documents are indexed and stored in a FAISS vector store for fast similarity search.
Chat History Integration: The chatbot uses contextual chat history to enhance the accuracy of answers based on previous user interactions.
Groq LLM: Uses Groq's powerful LLM to generate context-aware responses from the PDFs.
Interactive Chat Interface: Streamlit interface for seamless user interaction, allowing real-time Q&A with the uploaded PDF.
Requirements:
Python Libraries:
streamlit
langchain
langchain_community
langchain_core
langchain_groq
langchain_huggingface
langchain_text_splitters
langchain_community.document_loaders
langchain.vectorstores
dotenv
faiss-cpu
Groq API Key: You need a Groq API key to interact with Groq's LLM.
Setup:
Install the necessary libraries by running pip install -r requirements.txt.
Create a .env file in the root directory with your Groq API key.
Run the Streamlit app with streamlit run app.py.
Upload your PDF files and ask questions related to the content.
How it works:
Upload PDFs: Users upload PDFs via Streamlit's file uploader.
Document Splitting & Indexing: PDFs are split into smaller chunks and indexed using the FAISS vector store for efficient search and retrieval.
Query Handling: When a user asks a question, the chatbot utilizes a contextualization system to reframe or adapt the question, ensuring it is well-understood. The query is then passed through the RAG chain for document retrieval and answer generation.
Response Generation: Groq's LLM generates answers based on the retrieved context from the uploaded documents, maintaining a conversational flow with chat history integration.
Customization:
You can modify the system prompts for both contextualization and answer generation to fit your specific needs.
The session ID allows for separate conversations, making the chatbot interactive and adaptable to different users.
Example Usage:
Upload a PDF document (e.g., research paper, user manual, etc.).
Ask questions about the document.
The chatbot returns concise and relevant answers based on the uploaded document’s content.
This project provides a highly customizable, easy-to-use chatbot that can be deployed for document-based question answering, ideal for use cases like academic research, technical support, and more.
