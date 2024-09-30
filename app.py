import os
import streamlit as st
import warnings
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING
from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pinecone import Pinecone, ServerlessSpec
from langchain.chains import create_history_aware_retriever, create_retrieval_chain


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from groq import Groq
from langchain_groq import ChatGroq

import joblib
import os
import nest_asyncio  # noqa: E402
nest_asyncio.apply()

import pinecone
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyMuPDFLoader, DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers



chat_message_history = MongoDBChatMessageHistory(
        session_id="test_session",
        connection_string="mongodb+srv://amiteshpatrabtech2021:WHcFjhQUYwVb4KSX@aibf.snwxj.mongodb.net/",
        database_name="amiteshpatrabtech2021",
        collection_name="chat_histories",
    )

# MongoDB connection setup
def get_mongo_client():
    connection_string = "mongodb+srv://amiteshpatrabtech2021:WHcFjhQUYwVb4KSX@aibf.snwxj.mongodb.net/"

    client = MongoClient(connection_string)
    return client

# Create TTL index to expire chat history after 7 days
def create_ttl_index():
    client = get_mongo_client()
    db = client['amiteshpatrabtech2021']
    collection = db['chat_histories']

    # Create the TTL index on 'createdAt' field
    collection.create_index([("createdAt", ASCENDING)], expireAfterSeconds=604800)

# Function to display chat history from MongoDB
def see_chat_history():
    create_ttl_index()  # Ensure the TTL index is created

    chat_message_history = MongoDBChatMessageHistory(
        session_id="test_session",
        connection_string="mongodb+srv://amiteshpatrabtech2021:WHcFjhQUYwVb4KSX@aibf.snwxj.mongodb.net/",
        database_name="amiteshpatrabtech2021",
        collection_name="chat_histories",
    )

    if not chat_message_history.messages:
        return []

    history = []
    for message in chat_message_history.messages:
        if isinstance(message, AIMessage):
            history.append(("AI", message.content))
        elif isinstance(message, HumanMessage):
            history.append(("human", message.content))
    return history

def save_chat_message(message_content, role="human"):
    client = get_mongo_client()
    db = client['amiteshpatrabtech2021']
    collection = db['chat_histories']

    message_data = {
        "content": message_content,
        "role": role,
        "createdAt": datetime.utcnow()
    }
    collection.insert_one(message_data)

# Function to delete all chat history
def delete_chat_history():
    client = get_mongo_client()
    db = client['amiteshpatrabtech2021']
    collection = db['chat_histories']

    result = collection.delete_many({})
    st.sidebar.write(f"Deleted {result.deleted_count} messages from the chat history.")

# Sidebar for deleting chat history
st.sidebar.title("Manage Chat History")
if st.sidebar.button("Delete Chat History"):
    delete_chat_history()
    st.sidebar.success("Chat history has been deleted.")

# Set up the Pinecone vector store and embeddings
PINECONE_API_KEY = "de16b7ed-6489-4517-9fcb-7baae413eddb"
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "mf-rag"
index = pc.Index(index_name)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Define retriever with similarity score threshold
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 1, "score_threshold": 0.5},
)

# Define custom prompt template
prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

# Set up ChatGroq model
chat_model = ChatGroq(temperature=0.2, model_name="mixtral-8x7b-32768", api_key="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")


def get_context_retriever_chain(vector_store):
    llm = ChatGroq(model="mixtral-8x7b-32768",api_key="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")
    # llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 2, "score_threshold": 0.6},
    )
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user","{input}"),
        ("user","Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

# Define the QA chain using the ChatGroq model and retriever
qa = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

# Function to get chatbot response using RetrievalQA
def get_response(user_input):
    # Query the RetrievalQA model
    response = qa.invoke({"query": user_input})

    # Extract the helpful answer
    ai_response = response["result"]

    return ai_response

# Streamlit UI setup
st.title("AIBF Chatbot")

# Display chat history
# st.subheader("Chat History")
# chat_history = see_chat_history()
# for role, message in chat_history:
#     if role == "human":
#         st.write(f"**Human**: {message}")
#     else:
#         st.write(f"**AI**: {message}")

# User input field

user_input = st.chat_input("Type your message here...")

if user_input:
    # Save user message to MongoDB chat history
    save_chat_message(user_input, "human")

    # Get RAG-based response using the updated QA chain
    response = get_response(user_input)

    # Display AI response
    # st.write(f"**AI**: {response}")

    # Save AI response to MongoDB chat history
    save_chat_message(response, "ai")

# Chat interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello I am your Mutual fund advisor . How can I help you?")
    ]

if "vector_store" not in st.session_state:
    st.session_state.vector_store = vector_store

retriever_chain = get_context_retriever_chain(vector_store)

if user_input:
    response = get_response(user_input)
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=response))
    retrieved_documents = retriever_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })

# conversation
for index, message in enumerate(st.session_state.chat_history):
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            chat_message_history.add_ai_message(message.content)
            st.write(message.content)
            
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            chat_message_history.add_user_message(message.content)
            st.write(message.content)
with st.sidebar:
    st.header("AIBF Chatbot")
    st.subheader("Amitesh Patra")