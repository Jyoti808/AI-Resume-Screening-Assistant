"""AI Resume Screening Assistant - LangChain + RAG + Streamlit."""
import os
import tempfile
from typing import List

import pandas as pd
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from pathlib import Path


def load_env():
    """Read GOOGLE_API_KEY from the .env file that sits next to app.py."""
    f = Path(__file__).parent / ".env"
    if f.exists():
        for line in f.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")


load_env()


# ---------- Structured output schema ----------
class Evaluation(BaseModel):
    match_score: int = Field(description="Overall fit with the JD, 0-100")
    candidate_summary: str = Field(description="2-3 sentence summary of the candidate")
    matching_skills: List[str] = Field(description="JD skills found in the resume")
    missing_skills: List[str] = Field(description="JD skills NOT found in the resume")
    strengths: List[str]
    weaknesses: List[str]
    recommendation: str = Field(description="One of: Strong Hire, Hire, Maybe, No Hire")
    justification: str = Field(description="Short reasoning grounded in the resume")


parser = PydanticOutputParser(pydantic_object=Evaluation)

EVAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert technical recruiter. Evaluate the candidate ONLY using the "
     "resume excerpts provided. If something is not in the excerpts, treat it as "
     "missing - never invent experience.\n{format_instructions}"),
    ("human",
     "JOB DESCRIPTION:\n{jd}\n\nRESUME EXCERPTS ({candidate}):\n{context}\n\n"
     "Evaluate this candidate against the job description."),
])

COMPARE_PROMPT = ChatPromptTemplate.from_template(
    "You are a hiring manager. Using ONLY the evaluations below, recommend the best "
    "candidate for the job and justify it in one short paragraph, mentioning the "
    "runner-up.\n\nJOB DESCRIPTION:\n{jd}\n\nEVALUATIONS:\n{evals}"
)

QA_PROMPT = ChatPromptTemplate.from_template(
    "Answer the recruiter's question ONLY from the resume excerpts below. Mention "
    "the candidate name for each fact. If the answer is not in the excerpts, say "
    "\"Not found in the uploaded resumes.\"\n\nEXCERPTS:\n{context}\n\nQUESTION: {question}"
)


# ---------- RAG pipeline ----------
def build_vectorstore(files, api_key):
    """Load PDFs -> split -> embed -> FAISS. Each chunk is tagged with its candidate."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = []
    for f in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.getvalue())
            path = tmp.name
        pages = PyPDFLoader(path).load()
        os.unlink(path)
        for c in splitter.split_documents(pages):
            c.metadata["candidate"] = f.name
            chunks.append(c)
    if not chunks:
        raise ValueError("No text could be extracted (scanned PDFs need OCR).")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
    return FAISS.from_documents(chunks, embeddings)


def format_docs(docs):
    return "\n---\n".join(d.page_content for d in docs)


def evaluate_candidate(vs, llm, candidate, jd):
    retriever = vs.as_retriever(
        search_kwargs={"k": 6, "fetch_k": 60, "filter": {"candidate": candidate}}
    )
    context = format_docs(retriever.invoke(jd))
    chain = EVAL_PROMPT | llm | parser
    return chain.invoke({
        "jd": jd, "candidate": candidate, "context": context,
        "format_instructions": parser.get_format_instructions(),
    })


# ---------- UI ----------
st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")
st.title("📄 AI Resume Screening Assistant")

api_key = os.getenv("GOOGLE_API_KEY", "")

with st.sidebar:
    model = st.selectbox("LLM", ["gemini-2.5-flash", "gemini-2.5-flash-lite"])

files = st.file_uploader("Upload resumes (PDF)", type="pdf", accept_multiple_files=True)
jd = st.text_area("Job Description", height=200,
                  placeholder="Paste the job description here...")

if st.button("Evaluate", type="primary"):
    if not (api_key and files and jd.strip()):
        st.error("Make sure GOOGLE_API_KEY is in your .env file, and add at least one resume, and a job description.")
    else:
        try:
            llm = ChatGoogleGenerativeAI(model=model, temperature=0, google_api_key=api_key)
            with st.spinner("Indexing resumes..."):
                st.session_state.vs = build_vectorstore(files, api_key)
            results = {}
            bar = st.progress(0.0)
            for i, f in enumerate(files):
                with st.spinner(f"Evaluating {f.name}..."):
                    results[f.name] = evaluate_candidate(st.session_state.vs, llm, f.name, jd)
                bar.progress((i + 1) / len(files))
            st.session_state.results = results
            st.session_state.jd = jd
            st.session_state.llm_model = model
            st.session_state.pop("best", None)
        except Exception as e:
            st.error(f"Something went wrong: {e}")

results = st.session_state.get("results")
if results:
    st.header("Results")
    ranked = sorted(results.items(), key=lambda kv: kv[1].match_score, reverse=True)

    df = pd.DataFrame([
        {"Candidate": n, "Match Score": e.match_score, "Recommendation": e.recommendation}
        for n, e in ranked
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)

    for name, e in ranked:
        with st.expander(f"{name} - {e.match_score}/100 - {e.recommendation}"):
            st.progress(min(max(e.match_score, 0), 100) / 100)
            st.markdown(f"**Summary:** {e.candidate_summary}")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**✅ Matching skills**")
                st.write(", ".join(e.matching_skills) or "None")
                st.markdown("**💪 Strengths**")
                for s in e.strengths:
                    st.markdown(f"- {s}")
            with c2:
                st.markdown("**❌ Missing skills**")
                st.write(", ".join(e.missing_skills) or "None")
                st.markdown("**⚠️ Weaknesses**")
                for w in e.weaknesses:
                    st.markdown(f"- {w}")
            st.markdown(f"**Justification:** {e.justification}")

    if len(results) > 1 and st.button("Recommend best candidate"):
        llm = ChatGoogleGenerativeAI(model=st.session_state.llm_model, temperature=0, google_api_key=api_key)
        evals = "\n\n".join(f"{n}: {e.model_dump_json()}" for n, e in ranked)
        st.session_state.best = (COMPARE_PROMPT | llm | StrOutputParser()).invoke(
            {"jd": st.session_state.jd, "evals": evals})
    if st.session_state.get("best"):
        st.success(st.session_state.best)

if "vs" in st.session_state:
    st.header("Ask about the resumes")
    q = st.text_input("e.g. Who has experience with PyTorch? Compare A and B on Python.")
    if q and api_key:
        llm = ChatGoogleGenerativeAI(model=model, temperature=0, google_api_key=api_key)
        docs = st.session_state.vs.similarity_search(q, k=8)
        context = "\n---\n".join(f"[{d.metadata['candidate']}] {d.page_content}" for d in docs)
        st.write((QA_PROMPT | llm | StrOutputParser()).invoke(
            {"context": context, "question": q}))
