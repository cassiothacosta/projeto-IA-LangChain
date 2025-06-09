# 🧠 Agente IA Local com LangChain e Ollama

Este projeto utiliza a biblioteca [LangChain](https://github.com/langchain-ai/langchain) com LLM local (Ollama) para responder perguntas com base em documentos compactados em um `.zip` contendo arquivos `.csv` ou `.txt`, como notas fiscais. A interface é construída com [Gradio](https://gradio.app/), permitindo fácil interação via navegador.

---

## 🚀 Funcionalidades

- Upload de arquivos `.zip` com múltiplos arquivos `.csv` e `.txt`
- Leitura e segmentação automática dos documentos
- Criação de embeddings locais com HuggingFace e FAISS
- LLM local com suporte ao modelo `mistral` via Ollama
- Modos de resposta via busca vetorial (`RetrievalQA`) ou `MapReduce`
- Interface interativa e leve usando Gradio

---

## ⚙️ Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
```

### 2. Instale o Python (versão 3.10+ recomendada)

[Download do Python](https://www.python.org/downloads/)

### 3. Instale as dependências

#### Com `requirements.txt`:

```bash
pip install -r requirements.txt
```

#### Ou diretamente:

```bash
pip install gradio pandas langchain langchain-community langchain-ollama langchain-huggingface faiss-cpu
```

> ℹ️ No Linux, pode ser necessário:
> ```bash
> sudo apt-get install libopenblas-dev
> ```

---

## 🐘 Instalação do Ollama

### Windows

1. Baixe e instale o Ollama: https://ollama.com/download
2. Após a instalação, execute no terminal:

```bash
ollama run mistral
```

### Linux/macOS

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama run mistral
```

Isso irá baixar e iniciar o modelo local `mistral`.

---

## ▶️ Execução do Projeto

```bash
python main.py
```

O Gradio abrirá automaticamente uma interface no navegador.

---

## 🌐 Rodar sem Ollama (opcional)

Se estiver rodando em ambientes como Hugging Face Spaces:

- Substitua `OllamaLLM` por `OpenAI`, `HuggingFaceHub`, etc.
- Remova dependências relacionadas ao `ollama`.

---

## 📁 Estrutura esperada do `.zip`

Seu arquivo `.zip` deve conter arquivos `.csv` ou `.txt`. Exemplo:

```
dados.zip
├── cabecalho.csv
├── itens.csv
└── observacoes.txt
```

---

## 🧪 Exemplos de Perguntas (Notas Fiscais)

1. **Quais os produtos mais vendidos neste mês?**
2. **Qual foi o valor total de notas fiscais emitidas?**
3. **Quais clientes compraram mais de R$ 10.000?**
4. **Qual o item mais vendido por quantidade?**
5. **Houve algum produto com devolução?**
6. **Em quais datas houve maior volume de vendas?**

---
