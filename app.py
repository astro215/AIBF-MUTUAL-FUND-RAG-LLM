import streamlit as st
from mftool import Mftool
import pandas as pd
import time
import plotly.express as px

# Initialize the Mftool object
mf = Mftool()
# Custom CSS to style the app with a modern blue theme

# Create tabs
tabs = st.tabs(["Project Description" , "Mutual Fund Tools","Finnace Chat Bot"])

with tabs[1]:
    # Custom CSS to style the app with a modern black theme
    st.markdown("""
        <style>
        /* Background color for the app */
        .stApp {
            background-color: #121212;
            color: #ffffff;
        }

        /* Sidebar styling */
        .sidebar .sidebar-content {
            background-color: #1e1e1e;
            color: white;
        }

        /* Title styling */
        .st-title {
            color: #ffffff;
            font-size: 40px;
            font-weight: bold;
        }

        /* Progress bar color */
        .stProgress > div > div {
            background-color: #ff4081;
        }

        /* DataFrame container styling */
        .stDataFrame {
            border-radius: 10px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.3);
        }

        /* Input text and button styling */
        .stTextInput > div > div input {
            border: 2px solid #ff4081;
            border-radius: 8px;
            padding: 10px;
            background-color: #1e1e1e;
            color: #ffffff;
        }

        .stButton > button {
            background-color: #ff4081;
            color: white;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            transition: background-color 0.3s;
        }

        .stButton > button:hover {
            background-color: #d5006d;
        }

        /* Style for the radio buttons */
        .stRadio > label > div > div {
            color: #ff4081;
        }

        /* Styling for json and error messages */
        .stJson {
            background-color: #e8f0fe;
            border-radius: 8px;
            padding: 10px;
            color: #000000;
        }

        .stError {
            color: #ff4081;
        }

        /* Chart styling */
        .plotly {
            background-color: #121212;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title
    st.markdown("<div class='st-title'>Mutual Fund Tools App</div>", unsafe_allow_html=True)

    # Sidebar with selectbox to choose an action
    st.sidebar.title("Menu")
    menu_option = st.sidebar.selectbox("Choose an Action", [
        "Get Available Schemes under AMC",
        "Get Scheme Quote (Current NAV)",
        "Get Scheme Details",
        "Get Historical NAV Data"
    ])

    # Initialize empty containers for updating dynamically
    data_placeholder = st.empty()
    progress_placeholder = st.empty()

    # 1. Get Available Schemes under AMC
    if menu_option == "Get Available Schemes under AMC":
        amc_name = st.text_input("Enter AMC Name (e.g., 'ICICI')", value="HDFC")
        
        # Use radio button to allow user to select a filter for schemes
        filter_choice = st.radio("Filter Options", ("Show All Schemes", "Show only Equity Schemes", "Show only Debt Schemes"))
        
        # Nested filter choice based on user's selection
        additional_filter = None
        if filter_choice != "Show All Schemes":
            additional_filter = st.selectbox("Select additional filter", ["Filter 1", "Filter 2"])

        if st.button("Get Schemes"):
            try:
                progress = progress_placeholder.progress(0)  # Show progress bar
                schemes = mf.get_available_schemes(amc_name)
                if schemes:
                    # Filter based on user selection
                    if filter_choice == "Show only Equity Schemes":
                        schemes = {k: v for k, v in schemes.items() if "Equity" in v}
                    elif filter_choice == "Show only Debt Schemes":
                        schemes = {k: v for k, v in schemes.items() if "Debt" in v}
                    
                    # Convert the schemes to DataFrame
                    df = pd.DataFrame(list(schemes.items()), columns=["Scheme Code", "Scheme Name"])
                    
                    # Update the progress bar
                    progress.progress(50)
                    time.sleep(1)  # Simulate a delay

                    # Display the DataFrame using st.dataframe with advanced features
                    data_placeholder.dataframe(df.style.set_properties(**{
                        'background-color': '#1e1e1e',
                        'color': '#ff4081',
                        'border': '1px solid #ff4081',
                        'border-radius': '10px',
                        'font-family': 'Arial',
                        'font-size': '14px'
                    }), use_container_width=True)

                    # Full progress completion
                    progress.progress(100)
                    progress_placeholder.empty()  # Clear the progress bar
                else:
                    st.error("No schemes found for this AMC.")
            except Exception as e:
                st.error(f"Error: {e}")
                progress_placeholder.empty()  # Clear the progress bar

    # 2. Get Scheme Quote (Current NAV)
    elif menu_option == "Get Scheme Quote (Current NAV)":
        scheme_code = st.text_input("Enter Scheme Code", value="151729")
        
        # Radio button for different data view filters
        view_type = st.radio("View Options", ("Basic Info", "Full Info"))

        if st.button("Get Scheme Quote"):
            try:
                quote = mf.get_scheme_quote(scheme_code)
                if quote:
                    st.write(f"Quote for Scheme {scheme_code}:")
                    if view_type == "Basic Info":
                        st.json({
                            "scheme_code": quote["scheme_code"],
                            "scheme_name": quote["scheme_name"],
                            "nav": quote["nav"]
                        })
                    else:
                        st.json(quote)  # Show full information
                else:
                    st.error("No quote found for this scheme code.")
            except Exception as e:
                st.error(f"Error: {e}")

    # 3. Get Scheme Details
    elif menu_option == "Get Scheme Details":
        scheme_code = st.text_input("Enter Scheme Code for Details", value="151729")
        
        # Radio button for detail level
        detail_level = st.radio("Detail Level", ("Basic", "Extended"))
        
        if st.button("Get Scheme Details"):
            try:
                scheme_details = mf.get_scheme_details(scheme_code)
                if scheme_details:
                    st.write(f"Details for Scheme {scheme_code}:")
                    if detail_level == "Basic":
                        st.json({
                            "scheme_code": scheme_details["scheme_code"],
                            "scheme_name": scheme_details["scheme_name"],
                            "fund_house": scheme_details["fund_house"],
                        })
                    else:
                        st.json(scheme_details)  # Show full details
                else:
                    st.error("No details found for this scheme code.")
            except Exception as e:
                st.error(f"Error: {e}")

    # 4. Get Historical NAV Data
     # Get Historical NAV Data along with Scheme Details
    # Get Historical NAV Data along with Scheme Details
    elif menu_option == "Get Historical NAV Data":
        scheme_code = st.text_input("Enter Scheme Code for Historical NAV", value="151729")
        
        # Radio button to choose the output format
        output_format = st.radio("Output Format", ("Table", "JSON"))

        if st.button("Get Historical NAV"):  # Only execute the following after the button is pressed
            try:
                # Fetch scheme details first
                scheme_details = mf.get_scheme_details(scheme_code)
                
                if scheme_details:
                    st.write(f"Details for Scheme Code {scheme_code}:")
                    # Display basic scheme information above the graph
                    st.json({
                        "Scheme Name": scheme_details.get("scheme_name", "N/A"),
                        "Fund House": scheme_details.get("fund_house", "N/A"),
                        "Scheme Type": scheme_details.get("scheme_type", "N/A"),
                        "Scheme Category": scheme_details.get("scheme_category", "N/A")
                    })
                else:
                    st.error("No scheme details found for this scheme code.")
                
                # Fetch historical NAV data
                historical_nav = mf.get_scheme_historical_nav(scheme_code)
                
                # Extracting date and nav data into a DataFrame
                data = historical_nav['data']
                historical_nav_df = pd.DataFrame(data)

                # Convert 'date' to datetime format and 'nav' to float
                historical_nav_df['date'] = pd.to_datetime(historical_nav_df['date'], format='%d-%m-%Y')
                historical_nav_df['nav'] = historical_nav_df['nav'].astype(float)

                # Check if the DataFrame is not empty
                if not historical_nav_df.empty:
                    # Display DataFrame only after button is clicked
                    data_placeholder.dataframe(historical_nav_df.style.set_properties(**{
                        'background-color': '#1e1e1e',
                        'color': '#ff4081',
                        'border': '1px solid #ff4081',
                        'border-radius': '10px',
                        'font-family': 'Arial',
                        'font-size': '14px'
                    }), use_container_width=True)

                    # Plotting the line chart
                    fig = px.line(historical_nav_df, x='date', y='nav', 
                                title="NAV Over Time", 
                                labels={"date": "Date", "nav": "NAV"},
                                hover_data={"date": "|%d-%m-%Y", "nav": ":.2f"})
                    
                    # Customize the layout
                    fig.update_traces(line_color="#ff4081")
                    fig.update_layout(hovermode="x unified", xaxis_title="Date", yaxis_title="NAV",
                                    plot_bgcolor='#121212', paper_bgcolor='#121212',
                                    font=dict(color="#ffffff"))

                    # Show the plot
                    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
                else:
                    st.error("No historical NAV data found.")
            except Exception as e:
                st.error(f"Error: {e}")


with tabs[0]:
    # Project description content
    st.markdown("""
    # Project Description
    This application provides various tools for exploring and analyzing mutual fund data using the Mftool library. 
    It includes features such as:

    - Viewing available schemes under a specific Asset Management Company (AMC).
    - Obtaining the current Net Asset Value (NAV) quote for a scheme.
    - Viewing detailed information about a particular mutual fund scheme.
    - Displaying historical NAV data with visualization for better analysis.

    ## Features:
    - Interactive user interface with a modern black theme.
    - Filtering options to view equity or debt schemes.
    - Visualization using Plotly for historical NAV trends.

    This tool is particularly useful for investors looking to track and analyze mutual fund performance efficiently.
    """)

with tabs[2]:
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
    import joblib
    import os
    import nest_asyncio  # noqa: E402
    nest_asyncio.apply()
    from dotenv import load_dotenv
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.document_loaders import PyMuPDFLoader, DirectoryLoader, PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.prompts import PromptTemplate
    from langchain.llms import CTransformers
    load_dotenv()
    PINECONE_API_KEY = os.environ.get('PINECONE_API')
    connection_string1 = os.environ.get("MONGODB_CONNECTION_STRING")
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")


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
    chat_model = ChatGroq(temperature=0.2, model_name="mixtral-8x7b-32768", api_key=GROQ_API_KEY)


    def get_context_retriever_chain(vector_store):
        llm = ChatGroq(model="mixtral-8x7b-32768",api_key=GROQ_API_KEY)
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

    

    user_input = st.chat_input("Type your message here...")

    if user_input:
        # Save user message to MongoDB chat history
        save_chat_message(user_input, "human")

        # Get RAG-based response using the updated QA chain
        response = get_response(user_input)

        
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
        st.subheader("Jainil Patel")