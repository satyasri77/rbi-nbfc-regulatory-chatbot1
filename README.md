# RBI NBFC Regulatory Chatbot
A lightweight, RAG based chatbot is developed to answer questions strictly based on RBI regulations governing Non-Banking Financial Companies (NBFCs).

⚠️ The application does not store user data, chat logs, timestamps, or personal information and processes all interactions in memory.

## ✨ Key Features


- RBI NBFC–focused regulatory chatbot
- Strict scope limitation to RBI guidelines
- No user authentication
- No chat log storage
- No timestamp or user data persistence
- Clean, RBI-themed chat UI
- Suitable for demos, prototypes, and compliance-safe experimentation

## 🧠 Architecture Overview (RAG Pipeline)


This chatbot follows a **Retrieval-Augmented Generation (RAG)** approach:


1. **User Query**
- User submits a regulatory question via the chat UI.


2. **Context Retrieval**
- Relevant RBI NBFC regulatory content is retrieved from a predefined knowledge base (documents / embeddings).
- Only the retrieved context is used for response generation.


3. **Controlled Generation**
- The language model generates responses strictly limited to the provided RBI context.
- If information is not explicitly available, the system responds accordingly.


4. **Response Delivery**
- The answer is displayed to the user.
- No data is stored after the response is delivered.


> ⚠️ The system does not rely on external knowledge beyond the supplied RBI regulatory context.

