# ================== IMPORTS ==================
import streamlit as st
import json, os, re, time
from datetime import datetime
from dotenv import load_dotenv

# ---- LangChain (stable imports) ----
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---- Optional Speech (safe) ----
try:
    from streamlit_mic_recorder import mic_recorder
    from google.cloud import speech
    SPEECH_ENABLED = True
except Exception:
    SPEECH_ENABLED = False

# ================== ENV ==================
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# ================== HELPERS ==================
def extract_candidate_info(text):
    info = {"name": "", "experience": 0, "skills": [], "projects": []}
    name = re.search(r'name\s*[:\-]\s*(.+)', text, re.I)
    if name:
        info["name"] = name.group(1).strip()
    return info


def load_resumes(folder="resumes"):
    if not os.path.exists(folder):
        os.makedirs(folder)
        return []

    docs = []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        try:
            loader = PyPDFLoader(path) if file.endswith(".pdf") else TextLoader(path)
            pages = loader.load()
            full_text = "\n".join(p.page_content for p in pages)
            info = extract_candidate_info(full_text)

            for p in pages:
                p.metadata["candidate_name"] = info["name"] or file.replace("_", " ")
                docs.append(p)
        except Exception as e:
            print("Resume load error:", e)
    return docs


def build_resume_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    splits = splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_documents(splits, embeddings)


def create_resume_qa_chain(vs):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.6)

    prompt = PromptTemplate(
        template="""
You are an HR assistant.
Use ONLY the resume context.

Context:
{context}

Question: {question}

Rules:
- Always mention candidate name
- If info missing, say clearly
- Use bullet points
""",
        input_variables=["context", "question"],
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vs.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt},
    )

# ================== RESUME INIT ==================
resume_docs = load_resumes()
resume_vectorstore = build_resume_vectorstore(resume_docs) if resume_docs else None

# ================== STREAMLIT UI ==================
st.set_page_config("Event Bot AI", "🤖", layout="wide")

st.title("🤖 Event Bot AI – Hackathon Assistant")

st.set_page_config(
    page_title="AI Resume Assistant",
    page_icon="🤖",
    layout="wide"
)

load_css("style.css")

# ================== AUTH ==================
confirmed = [" ", "Jane Smith", "Alex Johnson"]

spoken_name = ""

if SPEECH_ENABLED:
    audio = mic_recorder(start_prompt="🎤 Speak", stop_prompt="🛑 Stop", key="mic")
    if audio:
        try:
            client = speech.SpeechClient()
            audio_data = speech.RecognitionAudio(content=audio["bytes"])
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=audio["sample_rate"],
                language_code="en-US",
            )
            res = client.recognize(config=config, audio=audio_data)
            spoken_name = res.results[0].alternatives[0].transcript
        except Exception:
            st.warning("Speech recognition unavailable")

typed_name = st.text_input("Type your full name")
user_name = spoken_name or typed_name

# ================== ACCESS ==================
if user_name:
    if user_name.lower() in [x.lower() for x in confirmed]:
        st.success(f"Welcome {user_name} 🎉")

        tab1, tab2, tab3 = st.tabs(["📅 Agenda", "💬 Ask", "🔎 Resume Search"])

        with tab1:
            st.json({"09:00": "Registration", "10:00": "AI Workshop"})

        with tab2:
            q = st.chat_input("Ask anything about event")
            if q:
                st.write("Gemini:", q)

        with tab3:
            if resume_vectorstore:
                query = st.chat_input("Ask about candidates")
                if query:
                    qa = create_resume_qa_chain(resume_vectorstore)
                    st.write(qa.run(query))
            else:
                st.info("No resumes uploaded")

    else:
        st.error("Name not found in confirmed list")