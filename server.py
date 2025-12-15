from fastapi import FastAPI
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI(api_key="sk-b73395fb85304e7f8abd390a5c20f145", base_url="https://api.deepseek.com")

system_instructions = """
Always be brief, use the fewest sentences possible, do not add extra context, examples, or elaboration, do not restate the question, explain ideas by directly linking cause and effect, and stop once the core logic is stated.
"""

def get_response(query, model:str='deepseek-chat'):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": query},
        ],
        max_tokens=1024, temperature=0.0, stream=False)
    return response.choices[0].message.content

class Query(BaseModel):
    query: str

app = FastAPI()

@app.get('/')
def index(): return FileResponse('index.html')

@app.post('/ask')
def ask(q: Query):
    response = get_response(q.query)
    return {'response': response} 

