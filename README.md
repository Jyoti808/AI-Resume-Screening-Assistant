# 📄 AI Resume Screening Assistant

An AI tool that reads resumes (PDF) and compares them with a job description.
For every candidate it gives a **match score (0-100)**, matching skills, missing skills,
strengths, weaknesses and a **hire / no-hire recommendation**. It can also recommend the
best candidate and answer questions about the uploaded resumes.


👉 [Try the Sarcasm Detection App](https://ai-resume-screening-assistant-dtbbreaqiq9kptbm2swumf.streamlit.app/)

## Features
- Upload one or many resumes (PDF) and paste any job description
- Match score and ranking table for all candidates
- Matching skills, missing skills, strengths and weaknesses for each candidate
- "Recommend best candidate" button that compares everyone
- Question box: ask things like "Who knows Python?" and get answers only from the resumes

## How it works (RAG pipeline)
1. Resume PDFs are loaded and split into small chunks (LangChain).
2. Chunks are converted to embeddings (Google Gemini) and stored in a FAISS vector database.
3. For each candidate, the most relevant chunks are retrieved and sent to Gemini together with the job description.
4. The answer comes back in a fixed format (Pydantic output parser) and is shown on a Streamlit web page.

## Tech used
Python, LangChain, Google Gemini, FAISS, Streamlit

## Project structure
```
app.py                      Streamlit web app
AI_Resume_Screening.ipynb   Notebook with step-by-step code and test cases
requirements.txt            Libraries needed
resumes/                    Sample resumes (PDF)
job_descriptions/           Sample job description
images/                     Screenshots
```

## How to run locally
1. Clone or download this repository.
2. Create a `.env` file in the project folder containing one line:
   ```
   GOOGLE_API_KEY=your-gemini-key
   ```
3. Install the libraries:
   ```
   python -m pip install -r requirements.txt
   ```
4. Start the app:
   ```
   python -m streamlit run app.py
   ```
5. Open http://localhost:8501, upload resumes, paste a job description and click **Evaluate**.

## Test cases (see the notebook)
1. Evaluate Resume A for a Data Scientist role
2. Compare Resume A and Resume B
3. Find the missing skills in Resume C
4. Recommend the best candidate among all resumes


## Note
The API key is kept in `.env` (local) or Streamlit Secrets (deployed) and is never uploaded to GitHub.
