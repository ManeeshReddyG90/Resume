import os
import re
import math
import json
import urllib.request
from collections import Counter
from flask import Flask, render_template, request
import PyPDF2

# Configure absolute paths for templates and static folders for Netlify serverless Lambda compatibility
base_dir = os.path.abspath(os.path.dirname(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(base_dir, 'templates'),
    static_folder=os.path.join(base_dir, 'static')
)

# Optional local SBERT loader (if sentence_transformers is installed locally)
local_sbert = None
try:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    import torch
    torch.set_num_threads(1)
    from sentence_transformers import SentenceTransformer
    local_sbert = SentenceTransformer('all-MiniLM-L6-v2')
    print("Local SBERT model loaded successfully.")
except Exception:
    print("Running in serverless/lightweight mode (local PyTorch skipped).")

# List of key technical and soft skills to look for in resumes
SKILLS_LIST = [
    'python', 'java', 'sql', 'html', 'css',
    'javascript', 'machine learning', 'communication',
    'git', 'data analysis', 'problem solving'
]


# Function 1: Read and extract text from an uploaded PDF resume file
def read_pdf(file_storage):
    extracted_text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(file_storage)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                extracted_text = extracted_text + " " + page_text
    except Exception:
        extracted_text = ""
    return extracted_text


# Function 2: Clean the raw text by making it lowercase and removing symbols
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    words = text.split()
    return " ".join(words)


# Pure Python Cosine Similarity (Zero heavy library dependencies)
def calculate_cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    if mag1 == 0.0 or mag2 == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot_product / (mag1 * mag2)))


# SBERT Embeddings Generator (supports local SBERT & free Hugging Face API)
def get_sbert_embedding(text):
    if not text.strip():
        return None

    # Option A: Local SBERT if available (local execution)
    if local_sbert is not None:
        try:
            return local_sbert.encode(text, show_progress_bar=False).tolist()
        except Exception:
            pass

    # Option B: Hugging Face Serverless API (when HF_TOKEN is configured in Netlify environment variables)
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if hf_token:
        try:
            req_data = json.dumps({"inputs": text}).encode('utf-8')
            req = urllib.request.Request(
                "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2",
                data=req_data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {hf_token}",
                    "User-Agent": "ResumeScreeningApp"
                }
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                if isinstance(data, list):
                    if len(data) > 0 and isinstance(data[0], list):
                        # Mean pooling over token embeddings
                        return [sum(dim) / len(data) for dim in zip(*data)]
                    return data
        except Exception as e:
            print(f"HF API notice: {e}")

    return None


# Fallback Term-Frequency Vector Cosine Similarity (guarantees 100% uptime on Netlify if offline)
def get_tf_cosine_similarity(text1, text2):
    words1 = text1.lower().split()
    words2 = text2.lower().split()
    vocab = list(set(words1 + words2))
    if not vocab:
        return 0.0
    c1 = Counter(words1)
    c2 = Counter(words2)
    v1 = [c1[w] for w in vocab]
    v2 = [c2[w] for w in vocab]
    return calculate_cosine_similarity(v1, v2)


# Function 3: Calculate similarity score between Job Description and Resume using SBERT + Cosine Similarity (0 to 100)
def calculate_score(job_desc, resume_text, job_embedding=None):
    if not resume_text.strip():
        return 0.0
    if job_embedding is None and not job_desc.strip():
        return 0.0

    # Retrieve job embedding
    if job_embedding is None:
        job_emb = get_sbert_embedding(job_desc)
    else:
        job_emb = job_embedding

    resume_emb = get_sbert_embedding(resume_text)

    # If SBERT embeddings are available (local or via HF API), calculate SBERT Cosine Similarity
    if job_emb is not None and resume_emb is not None:
        sim = calculate_cosine_similarity(job_emb, resume_emb)
        return round(sim * 100, 1)

    # Seamless fallback to TF vector Cosine Similarity
    sim = get_tf_cosine_similarity(job_desc, resume_text)
    return round(sim * 100, 1)


# Function 4: Check which important skills are present and which are missing
def find_skills(resume_text):
    skills_found = []
    skills_missing = []
    resume_lower = resume_text.lower()
    for skill in SKILLS_LIST:
        if skill in resume_lower:
            skills_found.append(skill.upper())
        else:
            skills_missing.append(skill.upper())
    return skills_found, skills_missing


def get_candidate_score(candidate):
    return candidate['score']


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/screen', methods=['POST'])
def screen_resumes():
    job_desc = request.form.get('job_description', '')
    uploaded_files = request.files.getlist('resumes')

    if not job_desc.strip():
        return render_template('index.html', error_message="Please paste a Job Description before proceeding.")

    if not uploaded_files or uploaded_files[0].filename == '':
        return render_template('index.html', error_message="Please upload at least one PDF resume.")

    cleaned_job_desc = clean_text(job_desc)
    job_embedding = get_sbert_embedding(cleaned_job_desc)

    results = []
    for file in uploaded_files:
        if file.filename.lower().endswith('.pdf'):
            raw_resume_text = read_pdf(file)
            cleaned_resume_text = clean_text(raw_resume_text)
            score = calculate_score(cleaned_job_desc, cleaned_resume_text, job_embedding=job_embedding)
            skills_found, skills_missing = find_skills(cleaned_resume_text)

            candidate_data = {
                'filename': file.filename,
                'score': score,
                'skills_found': skills_found,
                'skills_missing': skills_missing
            }
            results.append(candidate_data)

    if len(results) == 0:
        return render_template('index.html', error_message="No valid PDF files could be read. Please upload readable PDF files.")

    results.sort(key=get_candidate_score, reverse=True)
    current_rank = 1
    for candidate in results:
        candidate['rank'] = current_rank
        current_rank += 1

    return render_template('results.html', candidates=results, total=len(results))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
