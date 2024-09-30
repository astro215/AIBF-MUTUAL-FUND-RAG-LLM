
# Mutual Fund Advisory Chatbot Application

**Developers**: Amitesh Patra, Jainil Patel  
**Deployment**: [Streamlit](https://aibf-mutual-fund-rag-llm.streamlit.app/?embed_options=dark_theme)  <br>
**GitHub Repository**: [Link to Repository](https://github.com/amitesh30/AIBF-MUTUAL-FUND-RAG-LLM/)

## Objective
An interactive chatbot providing mutual fund advice with real-time data, personalized responses, and visualizations.
## Tech stack
<p align="center">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Pinecone-1B5E20?logo=pinecone&logoColor=white" alt="Pinecone" />
  <img src="https://img.shields.io/badge/Langchain-FFD700?logo=langchain&logoColor=black" alt="Langchain" />
  <img src="https://img.shields.io/badge/ChatGroq-3C873A?logo=groq&logoColor=white" alt="ChatGroq" />
  <img src="https://img.shields.io/badge/MongoDB-47A248?logo=mongodb&logoColor=white" alt="MongoDB" />
  <img src="https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white" alt="Plotly" />
  <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white" alt="Python" />
</p>

---

## Key Features

1. **Real-Time Mutual Fund Data Retrieval**:  
   Using `Mftool`, the app fetches updated and accurate information on mutual funds, including NAVs, scheme details, and fund performance.

2. **Personalized Guidance with Conversational AI**:  
   Powered by a Retrieval-Augmented Generation (RAG) model using `Langchain` for data retrieval and `ChatGroq` for generating context-aware responses.

3. **Vector Storage and Retrieval**:  
   `Pinecone` stores data embeddings, allowing for efficient similarity searches and quick retrieval of relevant information.

4. **Contextual Conversations**:  
   User chat history is stored using `MongoDB` to maintain context for personalized responses, with TTL indexing for automatic data expiry.

5. **Data Visualization**:  
   Users can view historical NAV data through interactive visualizations using `Plotly`.

---
## Architecture

```mermaid
graph TD;
    A[User Interaction] --> B[Data Retrieval];
    B --> C{Data Analysis and Visualization};
    C --> D[Historical NAV];
    C --> E[Chatbot Interaction];
    E --> F[Langchain Retrieval];
    F --> G[ChatGroq Response];
    E --> H[Pinecone Data Retrieval];
    H --> G;
    E --> I[MongoDB Storage];
    I --> E;

``` 



![Screenshot 2024-09-30 175656](https://github.com/user-attachments/assets/fc8ce360-90c3-472c-b7c7-276b9085e9d9)
![Screenshot 2024-09-30 175247](https://github.com/user-attachments/assets/018534a9-5fc8-44a3-90ec-d2dd0ca0d19b)
![Screenshot 2024-09-30 175227](https://github.com/user-attachments/assets/c215eb9c-529a-4c81-828c-20486772f423)
![Screenshot 2024-09-30 175205](https://github.com/user-attachments/assets/8cf94156-2bd5-457d-8ac1-84a64694ad4a)
![Screenshot 2024-09-30 174802](https://github.com/user-attachments/assets/e12a0284-2133-4940-86fc-29594aa16f20)
![Screenshot 2024-09-30 174750](https://github.com/user-attachments/assets/528444d4-d987-4db7-b80d-4f29bc8795f4)




