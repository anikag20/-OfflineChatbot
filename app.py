# 🔧 Full Offline QA-Enabled Streamlit App with Local BERT

import streamlit as st
import pdfplumber
import faiss
import numpy as np
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# ----------------------------
# 🌸 Streamlit Configuration
# ----------------------------
st.set_page_config(page_title="🧠 Offline Smart Research Chatbot", layout="wide")

# ----------------------------
# 🎨 Custom Styling (Optional)
# ----------------------------
def local_css():
    st.markdown("""
        <style>
        body, .stApp { background-color: #E9E4F0; color: #2A1A40; font-family: 'Segoe UI', sans-serif; }
        h1, h2, h3 { color: #8E6DEB; }
        section[data-testid="stSidebar"] { background-color: #DCD2EE !important; padding: 1.5rem; }
        .stMarkdown > div { background-color: #F3E9FE; border-left: 4px solid #C299FC; padding: 12px 18px; border-radius: 8px; margin-top: 10px; }
        </style>
    """, unsafe_allow_html=True)

local_css()

# ----------------------------
# 🔍 Load Models
# ----------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# ----------------------------
# 🧠 Build FAISS Index
# ----------------------------
@st.cache_resource
def build_index(_embedder, chunks):
    embeddings = _embedder.encode(chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

# ----------------------------
# ✂️ Text Chunking
# ----------------------------
@st.cache_data
def extract_text_chunks(files, max_pages=5):
    chunks = []
    for file in files:
        if file.name.endswith(".pdf"):
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages[:max_pages]:
                    text = page.extract_text() or ""
                    text = text.replace("\n", " ").strip()
                    if len(text) > 100:
                        chunks.append(text)
        elif file.name.endswith(".txt"):
            text = file.read().decode("utf-8")
            for para in text.split("\n\n"):
                para = para.strip().replace("\n", " ")
                if len(para) > 100:
                    chunks.append(para)
    return chunks

# ----------------------------
# 🧪 Local QA over top chunks
# ----------------------------
def answer_question_from_chunks(question, chunks, qa_model, embedder, top_k=3):
    q_vec = embedder.encode([question], convert_to_numpy=True)
    doc_vecs = embedder.encode(chunks, convert_to_numpy=True)
    sims = cosine_similarity(q_vec, doc_vecs)[0]
    top_idxs = sims.argsort()[-top_k:][::-1]

    answers = []
    for idx in top_idxs:
        context = chunks[idx]
        try:
            result = qa_model(question=question, context=context)
            answers.append((result['answer'], result['score'], context))
        except:
            continue

    if not answers:
        return "Sorry, I couldn't find an answer.", None

    best_answer = max(answers, key=lambda x: x[1])
    return best_answer[0], best_answer[2]

# ----------------------------
# 📌 Summarizer (Simple)
# ----------------------------
def summarize_chunks(chunks, embedder, max_words=150):
    vecs = embedder.encode(chunks, convert_to_numpy=True)
    avg_vec = np.mean(vecs, axis=0)
    scores = cosine_similarity([avg_vec], vecs)[0]
    top_idxs = np.argsort(scores)[-5:][::-1]
    selected = " ".join([chunks[i] for i in top_idxs])
    return " ".join(selected.split()[:max_words]) + "..."

# ----------------------------
# ❓ Question Generator (for quiz mode)
# ----------------------------
def generate_questions(chunks, n=3):
    sents = []
    for ch in chunks:
        sents += [s.strip() for s in ch.split(".") if len(s.strip().split()) > 5]
    picked = random.sample(sents, min(n, len(sents)))
    return [{"question": f"What is meant by: \"{s[:100]}...?\"", "answer": s, "source": s[:200] + "..."} for s in picked]

def similarity_score(a, b, embedder):
    vecs = embedder.encode([a, b], convert_to_numpy=True)
    return cosine_similarity([vecs[0]], [vecs[1]])[0][0]

def is_answer_correct(user_answer, correct_answer, embedder, threshold=0.7):
    score = similarity_score(user_answer, correct_answer, embedder)
    return score >= threshold, score

# ----------------------------
# 🚀 Streamlit App
# ----------------------------

          
def main():
    st.markdown("<h1 style='font-size:2.8rem;'>🧠 Offline Smart Research Chatbot</h1>", unsafe_allow_html=True)

    embedder = load_embedder()
    qa_model = load_qa_model()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.header("📂 Upload Files")
        files = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"], accept_multiple_files=True)

        if files and st.button("📄 Process Document"):
            with st.spinner("Processing..."):
                chunks = extract_text_chunks(files)

                if not chunks:
                    st.warning("⚠️ No extractable content found in the files.")
                    return

                index, _ = build_index(embedder, chunks)  # ✅ Unpacking only 2 values now
                st.session_state.chunks = chunks
                st.session_state.index = index
                st.session_state.questions = generate_questions(chunks)
                st.session_state.summary = summarize_chunks(chunks, embedder)
                st.session_state.chat_history.clear()
            st.success(f"✅ {len(chunks)} chunks processed.")

    if "chunks" not in st.session_state or not st.session_state.chunks:
        st.info("Please upload and process a document to begin.")
        return

    st.subheader("📌 Summary")
    st.markdown(st.session_state.summary)

    mode = st.selectbox("Select Mode", ["Ask Anything", "Challenge Me"])

    if mode == "Ask Anything":
        q = st.text_input("Ask a question from the document:")
        if st.button("Get Answer") and q.strip():
            answer, context = answer_question_from_chunks(q, st.session_state.chunks, qa_model, embedder)

            st.success("✍️ Answer:")
            st.write(answer)
            if context:
                st.caption("Justified by:")
                st.markdown(context[:300] + "...")

            st.session_state.chat_history.append({"q": q, "a": answer})

        if st.session_state.chat_history:
            with st.expander("🧠 Previous Q&A"):
                for i, turn in enumerate(st.session_state.chat_history[-5:], 1):
                    st.markdown(f"**Q{i}:** {turn['q']}")
                    st.markdown(f"**A{i}:** {turn['a']}")

    elif mode == "Challenge Me":
        with st.form("quiz"):
            answers = {}
            for i, q in enumerate(st.session_state.questions, 1):
                st.markdown(f"**Q{i}:** {q['question']}")
                answers[i] = st.text_input("Your Answer:", key=f"ans_{i}")
            submitted = st.form_submit_button("Submit Answers")

            if submitted:
                score = 0
                for i, q in enumerate(st.session_state.questions, 1):
                    ua = answers[i].strip()
                    correct = q["answer"]
                    passed, sim = is_answer_correct(ua, correct, embedder)
                    if passed:
                        st.success(f"✅ Q{i}: Correct ({sim:.2f})")
                        score += 1
                    else:
                        st.error(f"❌ Q{i}: Incorrect ({sim:.2f})")
                    st.markdown(f"**Answer:** {correct}")
                    st.caption(f"Justified by: {q['source']}")
                st.info(f"🏁 Score: {score}/{len(st.session_state.questions)}")
                if score == len(st.session_state.questions):
                    st.balloons()

        if st.button("🎲 New Challenge"):
            import time
            random.seed(time.time())
            new_qs = generate_questions(st.session_state.chunks)
            while new_qs == st.session_state.questions:
                new_qs = generate_questions(st.session_state.chunks)
            st.session_state.questions = new_qs
            st.rerun()


# Run the app
if __name__ == "__main__":
    main()

