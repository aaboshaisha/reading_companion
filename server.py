from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
from starlette.middleware.sessions import SessionMiddleware
import json, math, os, re, string, json, pickle, uuid, numpy as np
from collections import namedtuple, defaultdict, Counter
from sentence_transformers import SentenceTransformer
from nltk.stem import PorterStemmer

load_dotenv()

deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
base_url = "https://api.deepseek.com"

Document = namedtuple('Document', 'chunks pages embs index docmap doc_lens')
client = OpenAI(api_key=deepseek_api_key, base_url=base_url)
model = SentenceTransformer('all-MiniLM-L6-v2')
stemmer = PorterStemmer()

corpus_cache = {} # temporary in-memory save

with open('stopwords.txt') as f:
    stopwords = set(f.read().splitlines()) # O(1) lookup

# --------------------------------------------------------
def chunk_document(doc, size=10, overlap=2):
    """Yields (text, page) tuples using a sliding window over sentences."""
    for d in doc:
        sents = re.split(r"(?<=[.!?])\s+", d['text'])
        for i in range(0, len(sents) - size + 1, size - overlap):
            yield ' '.join(sents[i: i+size]), d['page']

def tokenize(text:str) -> list[str]:
    """Tokenize, lowercase, stem, remove punctuation and stopwords from text."""
    table = str.maketrans('', '', string.punctuation)
    words = text.lower().translate(table).split()
    return [stemmer.stem(word) for word in words if word not in stopwords]

def load_or_build_index(path:str) -> Document:
    """Build search index: chunks, embeddings, and BM25 data."""
    cache_path = path.replace('.json', '.pkl')
    
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
            
    with open(path) as f:
        doc_data = json.load(f)

    chunks, pages, index, docmap, doc_lens = [], [], defaultdict(dict), {}, []
    
    for cid, (txt, pg) in enumerate(chunk_document(doc_data)):
        chunks.append(txt); pages.append(pg)
        docmap[cid] = {'text':txt, 'page':pg}
        tokens = tokenize(txt)
        doc_lens.append(len(tokens))
        
        for token, tf in Counter(tokens).items():
            index[token][cid] = tf 
    embeddings = model.encode(chunks)
    return Document(chunks, pages, embeddings, index, docmap, doc_lens)

def cosine_sim(query_emb, chunks_emb):
    query_norm = query_emb / np.linalg.norm(query_emb)
    chunks_norm = chunks_emb / np.linalg.norm(chunks_emb, axis=1, keepdims=True)
    return np.dot(chunks_norm, query_norm)

def bm25_score(query_tokens, doc_id, index, doc_lens, k1=1.5, b=0.75):
    N = len(doc_lens) # total documents
    avgdl = sum(doc_lens) / N # average doc length
    score = 0 # accumulates scores of each token in that doc/chunk

    for token in query_tokens:
        if token not in index: continue

        df = len(index[token]) # document frequency
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        tf = index[token].get(doc_id, 0) # get tf for this doc (if token in it)
        dl = doc_lens[doc_id] # document length
        score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
    return score

def scores_to_ranks(cid_score_pairs: list[tuple]) -> dict:
    sorted_pairs = sorted(cid_score_pairs, key=lambda x: x[1], reverse=True)
    return {cid: rank for rank, (cid, score) in enumerate(sorted_pairs)}

def rrf_score(rank, k=60):
    return 1 / (k + rank)


def retrieve(query:str, index:Document, k:int=5) -> list[dict]:
    """Retrieve top-k chunks using RRF fusion of semantic and BM25 rankings."""
    query_tokens = tokenize(query)
    bm25_scores= [(cid, bm25_score(query_tokens, cid, index.index, index.doc_lens)) for cid in index.docmap.keys()]

    query_emb = model.encode(query)
    semantic_scores = cosine_sim(query_emb, index.embs)
    semantic_scores = [(cid, score) for cid, score in enumerate(semantic_scores.tolist())]

    semantic_ranks, bm25_ranks = scores_to_ranks(semantic_scores), scores_to_ranks(bm25_scores)

    combined_scores = sorted(((cid, rrf_score(semantic_ranks[cid]) + rrf_score(bm25_ranks[cid])) for cid in index.docmap.keys()), key=lambda x:x[1], reverse=True)

    context = [index.docmap[i] for i, _ in  combined_scores[:k]]
    return context    

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
    
    corpus = load_or_build_index(fpath)
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

def generate_prompt(query, context):
    return f"""Answer the question or provide information based on the provided documents.

    If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.
    
    Query: {query}
    
    Documents:
    {context}
    
    Instructions:
    - Provide a comprehensive answer that addresses the query
    - Cite sources using [1], [2], etc. format when referencing information
    - If sources disagree, mention the different viewpoints
    - If the answer isn't in the documents, say "I don't have enough information"
    - Be direct and informative
    
    Answer:""" 


@app.post('/ask')
def ask(q: Query, request: Request):
    corpus = corpus_cache[request.session['current_pdf']]
    context = retrieve(q.query, corpus)
    
    if 'session_id' not in request.session:
        request.session['session_id'] = str(uuid.uuid4())
    
    sid = request.session['session_id']
    
    if sid not in sessions:
        sessions[sid] = []
    
    # Build context string
    ctx = "\n\n".join([f"[Source {i+1}, Page {d['page']}]\n{d['text']}" for i, d in enumerate(context)])
    
    # Create messages: history + current query with context
    messages = sessions[sid] + [{'role': 'user', 'content': generate_prompt(q.query, ctx)}]
    
    resp = client.chat.completions.create(model="deepseek-chat", messages=messages, temperature=0)
    response = resp.choices[0].message.content
    
    # Store only the query and response in history (not the full prompt with context)
    sessions[sid].append({'role': 'user', 'content': q.query})
    sessions[sid].append({'role': 'assistant', 'content': response})
    
    # Trim memory
    if len(sessions[sid]) > max_messages:
        sessions[sid] = sessions[sid][-max_messages:]
    
    sources = [{"page": d["page"], "excerpt": make_excerpt(d["text"])} for d in context]
    return {"response": clean_markdown(response), "sources": sources}
