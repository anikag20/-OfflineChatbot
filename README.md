# 💬 Offline GenAI Assistant

A fully offline, document-aware chatbot designed for intelligent summarization, interactive Q&A, and logic-driven learning modes—packaged with a clean lavender-themed UI.

---

## 🛠️ Setup Instructions
1. clone the repo:-

git clone https://github.com/anikag20/-OfflineChatbot.git
cd OfflineChatbot

### 🔧 Requirements

- Python 3.11.3
- `pip` for dependency installation
- Internet for first-time setup only (for model download, if needed)

2. ### 📦 Install Dependencies

# bash
pip install -r requirements.txt

📚 3. Install Dependencies
Install all required packages using the provided requirements.txt:
pip install -r requirements.txt

🚀 4. Run the Assistant
Launch the Streamlit app:
streamlit run app.py
The app will open at http://localhost:8501 in your default browser.

🌈 4. Customize the Look
The lavender-themed UI is configured via:
.streamlit/config.toml


## Chatbot
This app will help you to gain valuable insights about your pdfs,it is an aesthetic app that uses logic to give accurate answers.There are two modes
.Ask me anything - This allows user to ask questions to chatbot related to pdf and bot will deliver the answers accordingly with the source and justification.There is an option of chat history also.

.Challenge me - This will ask 3 questions from the user and user have to answer it,accordingly the bot mark them and evaluates and gives the correct answers back.We can also generate new set of questions again.


## Launch The App
streamlit run app.py


## Architecture &Reasoning flow:-
                     ┌──────────────────┐
           │ Upload Document  │
           └────────┬─────────┘
                    ▼
           ┌──────────────────────┐
           │ Chunking & Cleaning  │
           │  (utils.py)          │
           └────────┬─────────────┘
                    ▼
           ┌──────────────────────┐
           │ Embedding with        │
           │ SentenceTransformers │
           └────────┬─────────────┘
                    ▼
               FAISS Vector Store
              (artifacts/faiss_store.pkl)

                    ▼
           ┌────────────────────────┐
           │ Streamlit Mode Switch │
           └──────┬────────┬───────┘
                  ▼        ▼
            Ask Mode    Summary Mode
            (LangChain   (Transformer
             Retrieval     Models +
             + Prompt)     Prompting)

                    ▼
           ┌────────────────────────┐
           │ Challenge Me Mode      │
           │ (Logic QGen + Answer   │
           │  Validator)            │
           └────────────────────────┘


Ask Anything: Embeds user query, retrieves relevant chunks from FAISS, and answers using LLM + custom prompt templates.

Summary: Generates concise, context-aware summaries with citation tracking.

Challenge Me: Uses logic-based question generation + answer validation for active recall.



## screenshots

### 🧠 Ask Anything Mode

![ask_anything png](https://github.com/user-attachments/assets/8b8b5166-ac4b-4dc1-af01-b5e227c2e232)

### 🎯 Challenge Me Mode

![challenge_me png](https://github.com/user-attachments/assets/d788383b-981a-41c7-a75a-
9f130503264f)

### 📊 Overview 

![overview png](https://github.com/user-attachments/assets/ba0e71e0-7f0d-48d1-b4c4-462fc4bd2151)

### ✨ Summary 

![summary png](https://github.com/user-attachments/assets/feae90b6-6a2c-4ac8-bcd2-7571b64efdf7)

📂Organized Source Code:-

chatbot/
├── .streamlit/
│   └── config.toml                # Streamlit UI configuration (theme, layout, etc.)
├── artifacts/
│   └── faiss_store.pkl            # Serialized vector store for offline retrieval
├── assets/
│   ├── ask_anything.png           # UI screenshot - Ask Mode
│   ├── challenge_me.png           # UI screenshot - Challenge Mode
│   ├── overview.png               # UI screenshot - Overview Mode
│   └── summary.png                # UI screenshot - Summary Mode
├── app.py                         # Streamlit app entry point
├── test_instructor.py            # Optional test module or scratchpad logic
├── requirements.txt              # Python dependencies
├── .gitignore                    # Files/folders excluded from version control
└── README.md                     # Project overview, setup, architecture


🌟 Highlights:-

.Runs fully offline once models are downloaded
.Grounded answers with citations
.Auto-generated summaries with high clarity
.Lavender aesthetic for calm, focused UI
.Designed for extensibility and modular upgrades

👩‍💻 Author
Built with care by Anika Goel 💜 "Built for clarity. Designed for intuition. Grounded in logic."
