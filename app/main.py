import os
import zipfile
import gradio as gr
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains.mapreduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.reduce import ReduceDocumentsChain
from langchain.schema import Document
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

# Pasta onde os arquivos extraídos serão armazenados
UPLOAD_FOLDER = "uploaded"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Estado compartilhado para armazenar instâncias de cadeias, LLMs, etc.
state = {"chain": None}

def unzip_file(zip_path):
    # """Extrai os arquivos do zip enviado para a pasta UPLOAD_FOLDER"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(UPLOAD_FOLDER)
    return os.listdir(UPLOAD_FOLDER)

def process_file(filename):
    # """Processa um arquivo CSV ou TXT para gerar vetores e cadeia QA"""

    path = os.path.join(UPLOAD_FOLDER, filename)
    documents = []

    # Leitura e transformação de CSV para Documentos de texto
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

    # Divide os documentos em trechos menores
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\\n\\n", "\\n", ",", " "]
    )
    split_docs = splitter.split_documents(documents)

    # Geração de embeddings e base vetorial
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    # Inicializa a LLM local
    llm = OllamaLLM(model="mistral")

    # Criação da cadeia de perguntas e respostas com base em vetores
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 15}),
        return_source_documents=True
    )

    # Armazena estado para uso posterior
    state["docs"] = split_docs
    state["llm"] = llm
    return qa_chain

def upload_and_select(zip_file):
    # """Faz upload do ZIP e retorna lista de arquivos extraídos"""
    if not zip_file:
        return "Nenhum arquivo zip enviado.", gr.Dropdown.update(choices=[], value=None)

    paths = unzip_file(zip_file.name)
    return f"{len(paths)} arquivos extraídos.", gr.update(choices=paths, value=paths[0] if paths else None)

def carregar_arquivo(nome_arquivo):
    # """Indexa o arquivo selecionado para QA"""
    state["chain"] = process_file(nome_arquivo)
    return f"Arquivo {nome_arquivo} indexado com sucesso."

def perguntar(pergunta, chat_log=[], modo_completo=False, modo_mapreduce=True):
    # """Realiza a pergunta à cadeia de QA com opção de MapReduce"""

    if not state["chain"]:
        return "Nenhum arquivo carregado.", chat_log

    llm = OllamaLLM(model="mistral")

    if modo_completo:
        # Junta todo o conteúdo para uma resposta completa (não recomendável para arquivos grandes)
        conteudo = "\n".join(d.page_content for d in state["docs"])
        prompt = f"Baseado no seguinte conteúdo de arquivo, responda à pergunta:\n\n{conteudo}\n\nPergunta: {pergunta}"
        resposta = llm.invoke(prompt)
        chat_log.append({"role": "user", "content": pergunta})
        chat_log.append({"role": "assistant", "content": resposta})
        return resposta, chat_log

    if modo_mapreduce:
        # Define template para perguntas no formato MapReduce
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="Use o seguinte contexto para responder à pergunta:\n\n{context}\n\nPergunta: {question}\nResposta:"
        )
        map_chain = LLMChain(llm=llm, prompt=prompt_template)
        reduce_chain = LLMChain(llm=llm, prompt=prompt_template)
        reduce_documents_chain = ReduceDocumentsChain(combine_documents_chain=reduce_chain)

        map_reduce_chain = MapReduceDocumentsChain(
            llm_chain=map_chain,
            reduce_documents_chain=reduce_documents_chain,
            document_variable_name="context"
        )

        docs = state["chain"].retriever.get_relevant_documents(pergunta)
        resposta = map_reduce_chain.invoke({"input": pergunta, "context": docs})
        chat_log.append({"role": "user", "content": pergunta})
        chat_log.append({"role": "assistant", "content": resposta})
        return resposta, chat_log

    # Caminho padrão com cadeia simples baseada em vetor
    resposta = state["chain"].invoke({"query": pergunta})
    chat_log.append({"role": "user", "content": pergunta})
    chat_log.append({"role": "assistant", "content": resposta["result"]})
    return resposta["result"], chat_log

# Interface Gradio
with gr.Blocks() as demo:
    gr.Markdown("# Agente IA Local com Arquivo ZIP")
    zip_input = gr.File(label="Envie seu arquivo .zip")
    status = gr.Textbox(label="Status")
    file_list = gr.Dropdown(label="Arquivos extraídos")
    modo_completo = gr.Checkbox(label="Forçar leitura completa do arquivo?")
    pergunta = gr.Textbox(label="Pergunta")
    resposta = gr.Textbox(label="Resposta")
    chatbox = gr.Chatbot(label="Histórico", type="messages")

    zip_input.change(upload_and_select, inputs=zip_input, outputs=[status, file_list])
    file_list.change(carregar_arquivo, inputs=file_list, outputs=status)
    pergunta.submit(perguntar, inputs=[pergunta, chatbox, modo_completo], outputs=[resposta, chatbox])

demo.launch()
