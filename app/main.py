import zipfile
import os
import gradio as gr
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader, TextLoader
from langchain.schema import Document
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

UPLOAD_FOLDER = "uploaded"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def unzip_file(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(UPLOAD_FOLDER)
    return os.listdir(UPLOAD_FOLDER)

def process_file(filename):
    path = os.path.join(UPLOAD_FOLDER, filename)

    documents = []

    if filename.endswith(".csv"):
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="ISO-8859-1")

        for i, row in df.iterrows():
            content = "\n".join(f"{col}: {row[col]}" for col in df.columns)
            documents.append(Document(page_content=content))
    else:
        loader = TextLoader(path, encoding="utf-8")
        documents = loader.load()

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    qa_chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=0),
        retriever=vectorstore.as_retriever()
    )
    return qa_chain


state = {"chain": None}

def upload_and_select(zip_file):
    if not zip_file:
        return "Nenhum arquivo zip enviado.", gr.Dropdown.update(choices=[], value=None)

    paths = unzip_file(zip_file.name)
    return f"{len(paths)} arquivos extraídos.", gr.update(choices=paths, value=paths[0] if paths else None)



def carregar_arquivo(nome_arquivo):
    state["chain"] = process_file(nome_arquivo)
    return f"Arquivo {nome_arquivo} indexado com sucesso."

def perguntar(pergunta, chat_history=[]):
    if not state["chain"]:
        return "Nenhum arquivo carregado.", chat_history
    resposta = state["chain"]({"question": pergunta, "chat_history": chat_history})
    chat_history.append((pergunta, resposta["answer"]))
    return resposta["answer"], chat_history

with gr.Blocks() as demo:
    zip_input = gr.File(label="Envie seu arquivo .zip")
    status = gr.Textbox(label="Status")
    file_list = gr.Dropdown(label="Arquivos extraídos")
    pergunta = gr.Textbox(label="Pergunta")
    resposta = gr.Textbox(label="Resposta")
    chatbox = gr.Chatbot()

    zip_input.change(upload_and_select, inputs=zip_input, outputs=[status, file_list])
    file_list.change(carregar_arquivo, inputs=file_list, outputs=status)
    pergunta.submit(perguntar, inputs=[pergunta, chatbox], outputs=[resposta, chatbox])

demo.launch()
