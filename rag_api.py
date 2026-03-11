from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import shutil

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    question: str


# -------------------------
# Models
# -------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

llm = Ollama(model="mistral")


# -------------------------
# Prompt
# -------------------------

prompt_template = """
You are an assistant answering questions about a document.

Use ONLY the provided context.

If the answer is not found in the context say:
"I cannot find this information in the document."

Context:
{context}

Question:
{question}

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)


# -------------------------
# Global RAG state
# -------------------------

vectorstore = None
retriever = None
qa_chain = None

doc_title = None
doc_author = None
doc_filename = None


# -------------------------
# Upload PDF
# -------------------------

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):

    global vectorstore, retriever, qa_chain
    global doc_title, doc_author, doc_filename

    file_path = f"temp_{file.filename}"
    doc_filename = file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print("Loaded document:", file.filename)

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # simple title guess from first page
    if documents:
        first_page = documents[0].page_content.split("\n")
        if len(first_page) > 0:
            doc_title = first_page[0]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1400,
        chunk_overlap=250,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    docs = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(docs, embeddings)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 15
        }
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT}
    )

    return {"message": "PDF uploaded and indexed successfully"}


# -------------------------
# Ask Question
# -------------------------

@app.post("/ask")
def ask(query: Query):

    global qa_chain, doc_title, doc_author, doc_filename

    if qa_chain is None:
        return {"answer": "Upload a PDF first before asking questions."}

    q = query.question.lower()

    # ---- Author detection ----
    if "author" in q or "who wrote" in q:
        if doc_author:
            return {"answer": f"The author is {doc_author}."}

    # ---- Title detection ----
    if "title" in q or "name of the document" in q:
        if doc_title:
            return {"answer": f"The document title appears to be '{doc_title}'."}
        else:
            return {"answer": f"The file name is '{doc_filename}'."}

    print("Question:", query.question)

    result = qa_chain.invoke(query.question)

    return {"answer": result["result"]}