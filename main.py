import os                                                                   #file paths, directory creation
import shutil                                                               #copying uploaded files
from fastapi import FastAPI, UploadFile, File                               #webFramework, handling file uploads
from fastapi.middleware.cors import CORSMiddleware                          #allows frontend---->backend
from fastapi.staticfiles import StaticFiles                                 #serve frontend files
from fastapi.responses import FileResponse                                  #return HTML page
from injest import build_index                                              #PDF--->FAISS pipeline
from query import answer                                                    #RAG query function

app = FastAPI()                                                             #create FastAPI app

app.add_middleware(                                                         
    CORSMiddleware,
    allow_origins=["*"],                                                    #enabling CORS
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="Frontend"), name="static")      #makes frontend folder accessible

UPLOAD_DIR = "uploaded_docs"                                                #creating upload dir
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")                                                               #homepage
def serve_frontend():
    return FileResponse("Frontend/index.html")

@app.post("/upload")                                                        #page where files are uploaded
async def upload_files(files: list[UploadFile] = File(...)):
    saved_paths = []

    for file in files:
        if not file.filename.endswith(".pdf"):
            continue
        filepath = os.path.join(UPLOAD_DIR, file.filename)
        with open(filepath, "wb") as f:                                     #Take uploaded file → copy all its contents → save to disk
            shutil.copyfileobj(file.file, f)
        saved_paths.append(filepath)

    build_index(saved_paths)

    return {"message": f"{len(saved_paths)} file(s) uploaded and indexed successfully."}    #JSON response to frontend

@app.post("/ask")                                                           #asks for query
async def ask_question(payload: dict):
    question = payload.get("question", "").strip()

    if not question:
        return {"error": "Question cannot be empty."}

    if not os.path.exists("faiss.index"):
        return {"error": "No documents indexed yet. Please upload files first."}

    response = answer(question)
    return {"answer": response}