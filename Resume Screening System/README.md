# AI Resume Screening System (SBERT + Cosine Similarity)

An intelligent, NLP-driven Resume Screening web application built with Flask, PyPDF2, Sentence-BERT (`all-MiniLM-L6-v2`), and Cosine Similarity.

## Features
- **Semantic Matching**: Uses Sentence-BERT (SBERT) embeddings to evaluate candidate fit beyond simple keyword matches.
- **Cosine Similarity Scoring**: Computes match percentage against the Job Description.
- **Automated Skill Detection**: Detects key technical and soft skills present and missing.
- **Candidate Ranking**: Ranks multiple uploaded PDF resumes automatically with visual badges and progress bars.

## Local Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application
py app.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## Free Cloud Deployment on Netlify (100% Free, No RAM Limits)

1. Push this folder to a GitHub repository.
2. Go to [Netlify](https://app.netlify.com/) and log in with GitHub.
3. Click **Add new site** → **Import an existing project** → select **GitHub**.
4. Choose your repository.
5. Netlify will automatically detect [`netlify.toml`](file:///c:/Users/MANEESH/Desktop/Resume%20Screening%20System/netlify.toml):
   - **Build command**: `pip install -r requirements.txt`
   - **Functions directory**: `netlify/functions`
   - **Publish directory**: `static`
6. *(Optional for unlimited HF SBERT API)*: Go to **Site settings** → **Environment variables** → add `HF_TOKEN` with your free token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
7. Click **Deploy Site**. Your site will be live instantly with a free `.netlify.app` domain!


