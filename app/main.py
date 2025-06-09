import zipfile
import os
import gradio as gr
import pandas as pd
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.schema import Document, HumanMessage, AIMessage
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter

UPLOAD_FOLDER = "uploaded"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
state = {"chain": None}

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

        for _, row in df.iterrows():
            content = "\n".join(f"{col}: {row[col]}" for col in df.columns)
            documents.append(Document(page_content=content))
    else:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        documents.append(Document(page_content=text))

    # Segmentar documento
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(documents)

    # Embeddings reais
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    # LLM local
    llm = OllamaLLM(model="mistral")

    # QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
        return_source_documents=True
    )
    return qa_chain


def upload_and_select(zip_file):
    if not zip_file:
        return "Nenhum arquivo zip enviado.", gr.Dropdown.update(choices=[], value=None)

    paths = unzip_file(zip_file.name)
    return f"{len(paths)} arquivos extraídos.", gr.update(choices=paths, value=paths[0] if paths else None)



def carregar_arquivo(nome_arquivo):
    state["chain"] = process_file(nome_arquivo)
    return f"Arquivo {nome_arquivo} indexado com sucesso."

def perguntar(pergunta, chat_log=[]):
    if not state["chain"]:
        return "Nenhum arquivo carregado.", chat_log

    chat_history_lc = []
    for msg in chat_log:
        if msg["role"] == "user":
            chat_history_lc.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            chat_history_lc.append(AIMessage(content=msg["content"]))

    resposta = state["chain"].invoke({"query": pergunta})

    chat_log.append({"role": "user", "content": pergunta})
    chat_log.append({"role": "assistant", "content": resposta["result"]})

    return resposta["result"], chat_log

with gr.Blocks() as demo:
    zip_input = gr.File(label="Envie seu arquivo .zip")
    status = gr.Textbox(label="Status")
    file_list = gr.Dropdown(label="Arquivos extraídos")
    pergunta = gr.Textbox(label="Pergunta")
    resposta = gr.Textbox(label="Resposta")
    chatbox = gr.Chatbot(type="messages")

    zip_input.change(upload_and_select, inputs=zip_input, outputs=[status, file_list])
    file_list.change(carregar_arquivo, inputs=file_list, outputs=status)
    pergunta.submit(perguntar, inputs=[pergunta, chatbox], outputs=[resposta, chatbox])

demo.launch()