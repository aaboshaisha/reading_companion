from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
import os, uuid
from starlette.middleware.sessions import SessionMiddleware
import os, re, json, pickle, numpy as np
from collections import namedtuple
from sentence_transformers import SentenceTransformer

load_dotenv()

Document = namedtuple('Document', 'chunks pages embs')
client = OpenAI(api_key=os.getenv('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")
model = SentenceTransformer('all-MiniLM-L6-v2')
corpus_cache = {} # temporary in-memory save

def chunk_document(doc, size=10, overlap=2):
    """Yields (text, page) tuples using a sliding window over sentences."""
    for d in doc:
        sents = re.split(r"(?<=[.!?])\s+", d['text'])
        for i in range(0, len(sents) - size + 1, size - overlap):
            yield ' '.join(sents[i: i+size]), d['page']

def load_or_create_corpus(path:str) -> Document:
    """Returns cached Document or computes and saves new embeddings."""
    cache_path = path.replace('.json', '.pkl')

    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        with open(cache_path, 'rb') as f: return pickle.load(f)

    with open(path) as f: doc_data = json.load(f)

    chunks, pages = zip(*chunk_document(doc_data))
    embeddings = model.encode(chunks)
    corpus = Document(chunks, pages, embeddings)

    with open(cache_path, 'wb') as f: pickle.dump(corpus, f)
    return corpus

def cosine_similarity(query_emb, chunks_emb):
    query_norm = query_emb / np.linalg.norm(query_emb)
    chunks_norm = chunks_emb / np.linalg.norm(chunks_emb, axis=1, keepdims=True)
    return np.dot(chunks_norm, query_norm)

def retrieve(query, corpus, k=5):
    """Finds top k chunks using vectorized cosine similarity."""
    query_emb = model.encode(query)
    scores = cosine_similarity(query_emb, corpus.embs)
    top_indices = np.argsort(scores)[-k:][::-1]
    return [{'text': ''.join(corpus.chunks[i]), 'page':corpus.pages[i]} for i in top_indices]

sessions = {} # storing memory for each user
max_messages = 11 # system + 10

def get_response(query, memory, model:str='deepseek-chat'):
    response = client.chat.completions.create(model=model, messages=memory, max_tokens=1024, temperature=0.0, stream=False)
    return response.choices[0].message.content

class Query(BaseModel):
    query: str

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=os.getenv('SECRET_KEY'))

@app.get('/')
def index(): return FileResponse('index.html')


@app.post('/upload-pdf-text')
async def upload_pdf_text(data:dict, request: Request):
    fname = data['filename'].replace('.pdf', '.json')
    fpath = f"temp/{fname}"

    # Save JSON temporarily
    os.makedirs('temp', exist_ok=True)
    with open(fpath, 'w') as f:
        json.dump(data['data'], f)
    
    corpus = load_or_create_corpus(fpath)
    corpus_cache[fname] = corpus
        
    request.session['current_pdf'] = fname
    
    return {"status": "success", "chunks": len(corpus.chunks)}

system_prompt = "Provide direct, 2-4 sentence answers. Always cite sources using [Source X] format when referencing information."


def make_excerpt(text, max_chars=200):
    text = text.strip().replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def clean_markdown(text: str) -> str:
    text = text.replace("\r\n", "\n").strip() # Normalize line endings
    text = re.sub(r"<[^>]+>", "", text) # Prevent raw HTML injection from the model
    return text


@app.post('/ask')
def ask(q: Query, request: Request):
    corpus = corpus_cache[request.session['current_pdf']]
    context = retrieve(q.query, corpus)
    
    if 'session_id' not in request.session:
        request.session['session_id'] = str(uuid.uuid4())
    
    sid = request.session['session_id']
    
    if sid not in sessions:
        sessions[sid] = [{'role': 'system', 'content': system_prompt}]
    
    sessions[sid].append({'role': 'user', 'content': q.query})
    
    ctx = "\n\n".join([f"[Source {i+1}, Page {d['page']}]\n{d['text']}" 
                       for i, d in enumerate(context)])
    prompt = f"Answer using this PDF context:\n\n{ctx}\n\nQuestion: {q.query}"
    
    response = get_response(prompt, sessions[sid])
    
    sessions[sid].append({'role': 'assistant', 'content': response})
    
    # Trim memory
    if len(sessions[sid]) > max_messages:
        sessions[sid] = [sessions[sid][0]] + sessions[sid][-(max_messages-1):]

    sources = [{"page": d["page"], "excerpt": make_excerpt(d["text"])} for d in context]
    return { "response": clean_markdown(response), "sources": sources }

