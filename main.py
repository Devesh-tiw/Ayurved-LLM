from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import time

# --- AI & RAG IMPORTS ---
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions

# 1. Load Environment Variables (Your hidden API Key)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    print("CRITICAL ERROR: GEMINI_API_KEY not found in .env file!")
    exit()

# 2. Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# 3. Connect to your Local ChromaDB (The Memory)
# We use get_or_create_collection so it never crashes even if the folder shifts!
chroma_client = chromadb.PersistentClient(path="./ayurveda_vector_db")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chroma_client.get_or_create_collection(
    name="ayurveda_herbs_en",
    embedding_function=sentence_transformer_ef
)

# 4. Initialize the blazing fast API
app = FastAPI()

# Setup folders for HTML and Audio
templates = Jinja2Templates(directory="templates")
os.makedirs('static/audio', exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Data model for incoming requests
class ChatRequest(BaseModel):
    message: str
    language: str = "en"

# --- ROUTES ---

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/ask")
async def ask_ai(chat_request: ChatRequest):
    user_message = chat_request.message
    user_lang = chat_request.language
    
    try:
        # STEP A: RETRIEVAL (Search the Vector Database)
        results = collection.query(
            query_texts=[user_message],
            n_results=3
        )
        
        # Combine the retrieved data into a single string
        retrieved_context = ""
        if results and results['documents'] and len(results['documents']) > 0:
            retrieved_context = "\n".join(results['documents'][0])
        
        # STEP B: PROMPT ENGINEERING
        system_prompt = f"""
        You are an empathetic, expert Ayurvedic practitioner. A patient has come to you and said: "{user_message}".
        
        Based ONLY on the following data retrieved from the Bhavaprakash Nighantu database:
        ---
        {retrieved_context}
        ---
        
        Provide a concise, clear, and empathetic response to the patient. Do not make up any information. If the data does not provide an answer, politely inform the patient that you cannot provide advice based on the available information.
        """
        
        # Enforce Language
        if user_lang == "hi":
            system_prompt += "\n\nCRITICAL: You MUST write your entire response in Hindi (हिंदी)."
        else:
            system_prompt += "\n\nCRITICAL: You MUST write your entire response in English."

        # STEP C: GENERATION (Ask Gemini)
        response = model.generate_content(system_prompt)
        answer_text = response.text
        
    except Exception as e:
        print(f"Error during RAG pipeline: {e}")
        if user_lang == "hi":
            answer_text = "क्षमा करें, मैं अभी डेटाबेस से कनेक्ट नहीं हो पा रहा हूँ।"
        else:
            answer_text = "Sorry, I am having trouble connecting to my database right now."

    # STEP D: RETURN TO FRONTEND
    return JSONResponse(content={
        "answer": answer_text,
        "audio_url": ""
    })
