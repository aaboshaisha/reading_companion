from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
import os, uuid
from starlette.middleware.sessions import SessionMiddleware

load_dotenv()

client = OpenAI(api_key=os.getenv('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")

system_instructions = """
You are answering questions about a PDF document the user is reading.
Answer clearly and directly in 2-4 sentences.
If the question requires detailed explanation, provide step-by-step reasoning.
If you're uncertain about something, say so.
"""

sessions = {} # storing memory for each user
max_messages = 11 # system + 10

def get_response(query, memory, model:str='deepseek-chat'):
    response = client.chat.completions.create(
        model=model,
        messages=memory,
        max_tokens=1024, temperature=0.0, stream=False)
    return response.choices[0].message.content

class Query(BaseModel):
    query: str

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=os.getenv('SECRET_KEY'))

@app.get('/')
def index(): return FileResponse('index.html')

@app.post('/ask')
def ask(q: Query, request:Request):
    if 'session_id' not in request.session:
        request.session['session_id'] = str(uuid.uuid4()) # generate uuid

    memory_id = request.session['session_id'] # get session id: just created or already exists
    
    if memory_id not in sessions: # is this first message? create conversation hx
        sessions[memory_id] = [{'role':'system', 'content': system_instructions}]
    
    sessions[memory_id].append({'role':'user', 'content':q.query})
    response = get_response(q.query, sessions[memory_id])
    sessions[memory_id].append({'role':'assistant', 'content':response})

    # trim memory
    if len(sessions[memory_id]) > max_messages:
        sessions[memory_id] = [sessions[memory_id][0]] + sessions[memory_id][-(max_messages-1):] # system + 10 messages

    return {'response': response} 

