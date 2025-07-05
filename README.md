M.Tech Prospectus Chatbot
  
This project utilizes the Kerala University M.Tech Prospectus—available for download from the official university website—as its primary knowledge source, encompassing comprehensive information on application procedures, eligibility requirements, program offerings, and course descriptions.
This is a modular Retrieval-Augmented Generation (RAG) chatbot integrated with RASA for natural language understanding and dialogue management.  
The project is split into four independent components:
1.data_preprocessing — Text extraction,chunking,embedding and vectore store
2.retrieval — Embedding + Vector Store
3.generation — LLM Answer Generation
4.rasa_layer — Rasa Bot Integration

Project Structure

MTECH_KU
├── chatbot.html
├── config.py
├── data_preprocessing
│   ├── chromadb_manager.py
│   ├── embedding_generator.py
│   ├── init.py
│   ├── pdf_extractor.py
│   ├── .pdf
│   └── text_chunker.py
├── generation
│   ├── init.py
│   ├── llm.py
|   ├── prompt_template.txt
│   ├── prompt_utils.py
│   └── rag_core.py
├── rasa_layer
│   ├── actions
│   │   ├── actions.py
│   │   ├── init.py
│   │   ├── mtech_chroma_data
│   ├── config.yml
│   ├── credentials.yml
│   ├── data
│   │   ├── nlu.yml
│   │   ├── rules.yml
│   │   └── stories.yml
│   ├── domain.yml
│   ├── endpoints.yml
│   ├── init.py
│   ├── models
│   │   └── 20250616-093921-violent-shell.tar.gz
│   └── tests
│       └── test_stories.yml
├── README.md
├── requirements.txt
└── retrieval
    ├── chroma_vectorstore.py
    ├── embedding.py
    ├── init.py


Setup Environment

cd /file_name
conda create -n chatbot_env python=3.9.21
conda activate chatbot_env
pip install -r requirements.txt

Update IP Addresses
In the chatbot.html, you must manually update the server IP address 


Run the Full System (3 Terminals)

Terminal 1: RASA Core

cd /file_name/rasa_layer
conda activate chatbot_env
rasa run  --cors "*" 

Terminal 2: RASA Action Server  

cd /file_name/rasa_layer/actions
conda activate chatbot_env
rasa run actions 

Terminal 3: Serve HTML Interface

python3 -m http.server 8003  --bind 0.0.0.0

Now open your browser and visit: http://localhost:8003/chatbot.html

Notes:
The RAG backend uses ChromaDB with custom chunking and Jina Embeddings v3.Vector data is stored in rasa_layer/actions/mtech_chroma_data/ — this includes, chroma.sqlite3 and the vectorstore data.
