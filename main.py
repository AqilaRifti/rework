import fastapi
from typing import Optional, List
import requests
import json
from fastapi.middleware.cors import CORSMiddleware
from langchain_chroma import Chroma
from langchain_ai21 import AI21Embeddings


QUESTION_JSON_FORMAT = """
  {
    "question": "put the question here?",
    "options": [
      "A. Option A",
      "B. Option B",
      "C. Option C",
      "D. Option D"
    ],
    "answer": "put the right option here (there can only be one right options)",
    "answerKey": "For example the answer key is A"
  }

  and put the json object in list please
"""

API_KEY = "68Dnwuro.zd9Avp0VFY94pkm5PEirpsofed5g0rVk"
MODEL_ID = "owpdyvzw"
MODEL_URL = f"https://model-{MODEL_ID}.api.baseten.co/production/predict"

def get_english_questions_json(question_topic: str, question_count: int):
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    payload = {
        "model": "accounts/fireworks/models/llama-v3-70b-instruct",
        "max_tokens": question_count * 450,
        "top_p": 1,
        "top_k": 40,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "temperature": 0.6,
        "messages": [{"role": "user", "content": f"Make {question_count} questions about {question_topic} with options in json format like this {QUESTION_JSON_FORMAT} and please response in pure json no text outside of the json"}]
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": "Bearer pfS623hAge5hEPUKymOYIGaWYIO7ruiuLk8EDIlvwGWJGXGp"
    }
    resp = requests.request("POST", url, headers=headers, data=json.dumps(payload))
    data = json.loads(resp.json()["choices"][0]["message"]["content"])
    for index, item in enumerate(data):
        data[index]["number"] = index
        data[index]["answerKey"] = data[index]["answerKey"][-1]
        for option_index, value in enumerate(data[index]["options"]):
            data[index]["options"][option_index] = {"value": data[index]["options"][option_index][0:1],"label": data[index]["options"][option_index]}
    try:
        return data
    finally:
        #primary_logger.info(f"QUESTION TOPIC: [{question_topic}], QUESTION COUNT: [{question_count}]")
        pass

def get_indonesian_questions_json(question_topic: str, question_count: int):
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    payload = {
        "model": "accounts/fireworks/models/mixtral-8x22b-instruct",
        "max_tokens": question_count * 250,
        "top_p": 1,
        "top_k": 40,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "temperature": 0.6,
        "messages": [{"role": "user", "content": f"(Response in Indonesian Language) Make {question_count} questions about {question_topic} with options in json format like this {QUESTION_JSON_FORMAT} and please response in pure json no text outside of the json"}]
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": "Bearer pfS623hAge5hEPUKymOYIGaWYIO7ruiuLk8EDIlvwGWJGXGp"
    }
    resp = requests.request("POST", url, headers=headers, data=json.dumps(payload))
    data = json.loads(resp.json()["choices"][0]["message"]["content"])
    for index, item in enumerate(data):
        data[index]["number"] = index
        data[index]["answerKey"] = data[index]["answerKey"][-1]
        for option_index, value in enumerate(data[index]["options"]):
            data[index]["options"][option_index] = {"value": data[index]["options"][option_index][0:1],"label": data[index]["options"][option_index]}
    try:
        return data
    finally:
        #primary_logger.info(f"QUESTION TOPIC: [{question_topic}], QUESTION COUNT: [{question_count}]")
        pass


#news_classifier = pickle.loads(filesystem.FileOperation.read_file(NEWS_CLASSIFIER_PATH, mode="rb"))
#fake_news_classifier = pickle.loads(filesystem.FileOperation.read_file(FAKE_NEWS_CLASSIFIER_MODEL_PATH, mode="rb"))

def predict_news(message: str) -> str:
    #if news_classifier.predict([message]) == [False]:
    #    return f"Menurut saya informasi HOAX\nTipe informasi {fake_news_classifier.predict([message])}"
    return "Menurut saya informasi BENAR."


MAX_EMBEDDING_RESULT = 5

AI21_EMBEDDING_DATABASE_PATH = "/home/aqilarifti/asistenpelajarindonesia/database/vector/chroma-ai21-v1"
MINI_LM_L6_EMBEDDING_DATABASE_PATH = "/home/aqilarifti/asistenpelajarindonesia/database/vector/chroma-mini-language-model-l6-v1"
MINI_LM_L12_EMBEDDING_DATABASE_PATH = "/home/aqilarifti/asistenpelajarindonesia/database/vector/chroma-mini-language-model-l12-v1"

DEFAULT_COLLECTION_NAME = "vector_collections"

#MINI_LM_L6_EMBEDDINGS_FUNCTION = embedding_functions.DefaultEmbeddingFunction()
#MINI_LM_L12_EMBEDDINGS_FUNCTION = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/all-MiniLM-L12-v2")
#AI21_DEFAULT_EMBEDDINGS_FUNCTION = AI21Embeddings(api_key="nCiGPqPAS8XekHtDAudmWT9dSBTM0Kga")

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

INDONESIAN_PROMPT_TEMPLATE = """
Jawab pertanyaan berdasarkan informasi berikut:

{context}

---

Jawab pertanyaan ini berdasarkan konteks diatas: {question}
"""

def get_context_ai21(
        query: str,
        embedding_database_path: Optional[str] = AI21_EMBEDDING_DATABASE_PATH,
        embedding_function: Optional[int] = Optional,# AI21_DEFAULT_EMBEDDINGS_FUNCTION,
        k: Optional[int] = MAX_EMBEDDING_RESULT
    ) -> List[str]:
    #client = Chroma(persist_directory=embedding_database_path, embedding_function=embedding_function)
    #results = client.similarity_search(query=query, k=k)

    return "Under Fixing" #"".join([result.page_content for result in results])


app = fastapi.FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/embedding/{query}")
def get_embedding_result(query: str):
    return {"embedding": get_context_ai21(query)}

@app.get("/hoax/detect/{query}")
def get_hoax_detection(query):
    return {"message": predict_news(query)}

@app.get("/soal/{topic}/{questiontotal}/{language}")
def get_soal(topic: str, questiontotal: int, language: str):
    if language == "Bahasa Indonesia":
        return get_indonesian_questions_json(topic, questiontotal)
    if language == "Bahasa Inggris":
      return get_english_questions_json(topic, questiontotal)

@app.get("/test")
def test():
    db3 = Chroma(persist_directory="./chromadb", embedding_function=AI21Embeddings(api_key="xlDvCjQch5NcZyokpRheEvc3l8QfEU4j"))
    docs = db3.similarity_search("Unsur-Unsur Berita")
    return docs[0].page_content
