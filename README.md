# Sistema de Consulta Jurídica Baseado em PDFs

Este projeto Python permite processar documentos PDF do ramo do direito, construir um modelo de linguagem baseado nas informações extraídas, salvá-lo, atualizá-lo incrementalmente com novos PDFs e interagir com ele através de um terminal ou de uma interface web Streamlit.

## Funcionalidades

- **Extração de Texto de PDFs:** Converte o conteúdo de PDFs em texto puro.
- **Pré-processamento de Texto:** Limpa e normaliza o texto extraído para otimizar o treinamento.
- **Geração de Embeddings:** Utiliza modelos de linguagem (Sentence Transformers) para transformar texto em vetores numéricos (embeddings).
- **Indexação FAISS:** Cria um índice eficiente para busca de similaridade entre os embeddings, permitindo encontrar rapidamente documentos relevantes.
- **Persistência do Modelo:** Salva e carrega o modelo de embeddings, o índice FAISS e os metadados dos documentos, permitindo continuar o treinamento ou usar o modelo a qualquer momento.
- **Atualização Incremental:** Adiciona novos PDFs ao modelo existente sem a necessidade de retreinar tudo do zero.
- **Interface de Conversa (Terminal):** Permite interagir com o modelo via terminal, fazendo perguntas e recebendo trechos de documentos relevantes como resposta.
- **Interface Web (Streamlit):** Uma interface gráfica amigável para upload de PDFs, treinamento do modelo e chat interativo.

## Estrutura do Projeto

```
. 
├── data/                 # PDFs de entrada para treinamento
├── models/               # Modelos treinados e índices salvos
│   ├── sentence_transformer_model/ # Modelo Sentence Transformer
│   ├── faiss_index.bin           # Índice FAISS
│   └── documents_metadata.json   # Metadados dos documentos processados
│   └── processed_pdfs.log        # Log de PDFs já processados
├── src/                  # Código-fonte do projeto
│   ├── __init__.py
│   ├── pdf_processor.py  # Módulo para extração e pré-processamento de PDFs
│   ├── model_trainer.py  # Módulo para treinamento e gerenciamento do modelo/índice
│   ├── main.py           # Script principal para interação via terminal
│   └── app.py            # Script principal para a aplicação Streamlit
├── requirements.txt      # Dependências do Python
└── README.md             # Este arquivo de documentação
```

## Requisitos

- Python 3.8 ou superior
- `pip` (gerenciador de pacotes Python)

## Instalação

1.  **Clone o repositório (ou crie a estrutura de diretórios manualmente):**

    ```bash
    # Se você tiver um repositório git
    # git clone <url_do_seu_repositorio>
    # cd <nome_do_repositorio>

    # Se não, crie as pastas manualmente
    mkdir -p data models src
    ```

2.  **Navegue até o diretório raiz do projeto:**

    ```bash
    cd /caminho/para/o/seu/projeto
    ```

3.  **Instale as dependências:**

    ```bash
    pip install -r requirements.txt
    ```

## Uso

### Via Terminal

1.  **Coloque seus PDFs:**

    Mova os arquivos PDF do ramo do direito que você deseja usar para treinar o modelo para a pasta `data/`.

2.  **Execute o script principal:**

    ```bash
    python3 src/main.py
    ```

    O script apresentará um menu interativo:

    ```
    Bem-vindo ao Sistema de Consulta Jurídica!
    Certifique-se de colocar seus PDFs na pasta 'data/'.

    Escolha uma opção:
    1. Treinar/Atualizar modelo com PDFs
    2. Conversar com o modelo
    3. Sair
    Opção: 
    ```

    - **Opção 1: Treinar/Atualizar modelo com PDFs:** Inicia o processo de treinamento. Na primeira execução, o modelo `paraphrase-multilingual-MiniLM-L12-v2` será baixado (requer conexão com a internet). O sistema irá extrair texto de todos os PDFs na pasta `data/` que ainda não foram processados, pré-processar o texto, gerar embeddings, adicionar ao índice FAISS e salvar o modelo, o índice e os metadados dos documentos na pasta `models/`.
    - **Opção 2: Conversar com o modelo:** Após o modelo ser treinado (ou carregado se já existir), você pode usar esta opção para fazer perguntas no terminal. O sistema buscará os documentos mais similares e apresentará os trechos mais relevantes. Digite `sair` a qualquer momento para encerrar o modo de conversa.
    - **Opção 3: Sair:** Encerra o programa.

### Via Interface Web (Streamlit)

1.  **Coloque seus PDFs:**

    Embora a interface Streamlit permita o upload, é recomendado colocar os PDFs diretamente na pasta `data/` para um processamento mais robusto, especialmente para grandes volumes.

2.  **Execute a aplicação Streamlit:**

    Navegue até o diretório raiz do projeto e execute:

    ```bash
    streamlit run src/app.py
    ```

    Isso abrirá a aplicação Streamlit no seu navegador padrão (geralmente `http://localhost:8501`).

    A interface Streamlit possui duas abas:

    - **Treinar/Atualizar Modelo:** Permite fazer upload de PDFs e iniciar o processo de treinamento/atualização do modelo. Você verá o progresso e as mensagens de sucesso/erro diretamente na interface.
    - **Conversar com o Modelo:** Uma interface de chat interativa onde você pode digitar suas perguntas e ver as respostas baseadas nos documentos treinados. As respostas incluirão a origem do documento e um trecho relevante.

## Detalhes do Código

### `requirements.txt`

```
PyPDF2
sentence-transformers
faiss-cpu
torch
scipy
scikit-learn
huggingface-hub
transformers
streamlit
```

### `src/pdf_processor.py`

```python
import os
from PyPDF2 import PdfReader
import re

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extrai texto de um arquivo PDF.

    Args:
        pdf_path (str): O caminho completo para o arquivo PDF.

    Returns:
        str: O texto extraído do PDF, ou uma string vazia se ocorrer um erro.
    """
    text = ""
    if not os.path.exists(pdf_path):
        print(f"Erro: O arquivo PDF não foi encontrado em {pdf_path}")
        return text
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            for page in reader.pages:
                # Usar .extract_text() e garantir que não é None
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Erro ao extrair texto do PDF {pdf_path}: {e}")
    return text

def preprocess_text(text: str) -> str:
    """
    Pré-processa o texto extraído para limpeza e normalização.

    Args:
        text (str): O texto bruto a ser pré-processado.

    Returns:
        str: O texto pré-processado.
    """
    # Converter para minúsculas
    text = text.lower()
    # Remover múltiplos espaços em branco e quebras de linha excessivas
    text = re.sub(r'\s+', ' ', text)
    # Remover caracteres especiais, mantendo letras, números e pontuação básica para o português
    # Adicionado \' para apóstrofos comuns em textos jurídicos
    text = re.sub(r'[^a-z0-9áéíóúçãõâêôàèìòùäëïöüñ.,;?!\s\[\]\(\)\{\}\'\"\-]', '', text)
    # Remover espaços em branco no início e fim
    text = text.strip()
    return text

if __name__ == '__main__':
    # Exemplo de uso e teste do módulo
    # Para testar, crie um arquivo 'data/exemplo.pdf' com algum texto.
    # Certifique-se de que a pasta 'data' existe.
    sample_pdf_path = 'data/exemplo.pdf'
    print(f"Tentando extrair texto de: {sample_pdf_path}")
    extracted_text = extract_text_from_pdf(sample_pdf_path)
    if extracted_text:
        print("\n--- Texto Extraído (primeiras 500 chars): ---")
        print(extracted_text[:500])
        processed_text = preprocess_text(extracted_text)
        print("\n--- Texto Pré-processado (primeiras 500 chars): ---")
        print(processed_text[:500])
    else:
        print("\nNenhum texto extraído. Certifique-se de que 'data/exemplo.pdf' existe e contém texto.")
```

### `src/model_trainer.py`

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import json

class LegalModelTrainer:
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        """
        Inicializa o treinador do modelo.

        Args:
            model_name (str): Nome do modelo Sentence Transformer a ser usado.
        """
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Erro ao carregar o modelo SentenceTransformer {model_name}: {e}")
            print("Verifique sua conexão com a internet ou se o nome do modelo está correto.")
            self.model = None

        self.index = None
        self.documents = []
        self.document_paths = [] # Para rastrear de quais PDFs os documentos vieram

    def create_embeddings(self, texts: list[str]) -> np.ndarray:
        """
        Gera embeddings para uma lista de textos.

        Args:
            texts (list[str]): Lista de textos para gerar embeddings.

        Returns:
            np.ndarray: Array NumPy de embeddings.
        """
        if self.model is None:
            print("Modelo SentenceTransformer não carregado. Não é possível criar embeddings.")
            return np.array([])
        print(f"Gerando embeddings para {len(texts)} textos...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return np.array(embeddings).astype('float32')

    def build_index(self, embeddings: np.ndarray):
        """
        Constrói um índice FAISS a partir dos embeddings.

        Args:
            embeddings (np.ndarray): Embeddings para indexar.
        """
        if embeddings.size == 0:
            print("Nenhum embedding para construir o índice.")
            return
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search
        self.index.add(embeddings)
        print(f"Índice FAISS construído com {self.index.ntotal} vetores.")

    def train_and_index(self, new_texts: list[str], new_paths: list[str]):
        """
        Treina ou atualiza o modelo e o índice com novos textos.

        Args:
            new_texts (list[str]): Novos textos pré-processados a serem adicionados.
            new_paths (list[str]): Caminhos dos PDFs correspondentes aos novos textos.
        """
        if not new_texts:
            print("Nenhum novo texto para treinar ou indexar.")
            return

        self.documents.extend(new_texts)
        self.document_paths.extend(new_paths)

        embeddings = self.create_embeddings(new_texts)
        if embeddings.size == 0:
            return

        if self.index is None:
            self.build_index(embeddings)
        else:
            self.index.add(embeddings)
            print(f"Novos vetores adicionados ao índice. Total: {self.index.ntotal}")

    def save_model_and_index(self, model_dir: str = 'models'):
        """
        Salva o modelo Sentence Transformer, o índice FAISS e os documentos.

        Args:
            model_dir (str): Diretório onde os arquivos serão salvos.
        """
        if self.model is None or self.index is None:
            print("Modelo ou índice não inicializado. Não é possível salvar.")
            return

        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'sentence_transformer_model')
        index_path = os.path.join(model_dir, 'faiss_index.bin')
        docs_metadata_path = os.path.join(model_dir, 'documents_metadata.json')

        try:
            self.model.save(model_path)
            faiss.write_index(self.index, index_path)
            
            # Salvar documentos e seus caminhos
            metadata = [{'text': text, 'path': path} for text, path in zip(self.documents, self.document_paths)]
            with open(docs_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)

            print(f"Modelo, índice e metadados dos documentos salvos em {model_dir}")
        except Exception as e:
            print(f"Erro ao salvar modelo/índice/documentos: {e}")

    @classmethod
    def load_model_and_index(cls, model_dir: str = 'models'):
        """
        Carrega o modelo Sentence Transformer, o índice FAISS e os documentos.

        Args:
            model_dir (str): Diretório de onde os arquivos serão carregados.

        Returns:
            LegalModelTrainer or None: Uma instância de LegalModelTrainer carregada, ou None se falhar.
        """
        model_path = os.path.join(model_dir, 'sentence_transformer_model')
        index_path = os.path.join(model_dir, 'faiss_index.bin')
        docs_metadata_path = os.path.join(model_dir, 'documents_metadata.json')

        if not os.path.exists(model_path) or not os.path.exists(index_path) or not os.path.exists(docs_metadata_path):
            print("Arquivos de modelo, índice ou metadados dos documentos não encontrados. Retornando None.")
            return None

        trainer = cls()
        try:
            trainer.model = SentenceTransformer(model_path)
            trainer.index = faiss.read_index(index_path)
            with open(docs_metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                trainer.documents = [item['text'] for item in metadata]
                trainer.document_paths = [item['path'] for item in metadata]
            print(f"Modelo, índice e metadados dos documentos carregados de {model_dir}")
            return trainer
        except Exception as e:
            print(f"Erro ao carregar modelo/índice/documentos: {e}")
            return None

    def search_similar_documents(self, query_text: str, k: int = 5) -> list[dict]:
        """
        Busca documentos similares a uma consulta.

        Args:
            query_text (str): O texto da consulta.
            k (int): O número de documentos mais similares a retornar.

        Returns:
            list[dict]: Uma lista de dicionários contendo o documento, a distância e o caminho do PDF de origem.
        """
        if self.model is None or self.index is None or self.index.ntotal == 0:
            print("Modelo não treinado ou índice vazio. Não é possível realizar a busca.")
            return []

        query_embedding = self.model.encode([query_text]).astype('float32')
        
        # Garante que k não exceda o número total de documentos no índice
        k = min(k, self.index.ntotal)
        if k == 0:
            return []

        distances, indices = self.index.search(query_embedding, k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents): # Garante que o índice é válido
                results.append({
                    'document': self.documents[idx],
                    'distance': float(distances[0][i]),
                    'source_path': self.document_paths[idx]
                })
        return results

if __name__ == '__main__':
    # Exemplo de uso e teste do módulo
    MODEL_TEST_DIR = 'models_test'
    os.makedirs(MODEL_TEST_DIR, exist_ok=True)

    trainer = LegalModelTrainer()
    sample_texts = [
        "O direito civil regula as relações entre particulares.",
        "A constituição federal é a lei máxima do país.",
        "Contratos são acordos de vontades que geram obrigações.",
        "O código penal define os crimes e suas respectivas penas.",
        "A doutrina jurídica é fundamental para a interpretação das leis."
    ]
    sample_paths = [
        "data/doc1.pdf",
        "data/doc2.pdf",
        "data/doc3.pdf",
        "data/doc4.pdf",
        "data/doc5.pdf"
    ]

    print("\n--- Teste de Treinamento Inicial ---")
    trainer.train_and_index(sample_texts, sample_paths)
    trainer.save_model_and_index(MODEL_TEST_DIR)

    print("\n--- Teste de Carregamento e Busca ---")
    loaded_trainer = LegalModelTrainer.load_model_and_index(MODEL_TEST_DIR)
    if loaded_trainer:
        query = "qual a importância da constituição?"
        results = loaded_trainer.search_similar_documents(query, k=2)
        print(f"\nResultados da busca para '{query}':")
        for res in results:
            print(f"  Documento: {res['document']}\n  Distância: {res['distance']}\n  Origem: {res['source_path']}\n")

        print("\n--- Teste de Atualização Incremental ---")
        new_sample_texts = [
            "A jurisprudência é a interpretação das leis pelos tribunais.",
            "O processo civil trata das ações judiciais não criminais."
        ]
        new_sample_paths = [
            "data/doc6.pdf",
            "data/doc7.pdf"
        ]
        loaded_trainer.train_and_index(new_sample_texts, new_sample_paths)
        loaded_trainer.save_model_and_index(MODEL_TEST_DIR)

        print("\n--- Teste de Busca Após Atualização ---")
        query_updated = "o que é jurisprudência?"
        results_updated = loaded_trainer.search_similar_documents(query_updated, k=2)
        print(f"\nResultados da busca para '{query_updated}':")
        for res in results_updated:
            print(f"  Documento: {res['document']}\n  Distância: {res['distance']}\n  Origem: {res['source_path']}\n")

    # Limpeza dos arquivos de teste
    import shutil
    if os.path.exists(MODEL_TEST_DIR):
        shutil.rmtree(MODEL_TEST_DIR)
        print(f"\nDiretório de teste {MODEL_TEST_DIR} removido.")
```

### `src/main.py`

```python
import os
import glob
import sys
from src.pdf_processor import extract_text_from_pdf, preprocess_text
from src.model_trainer import LegalModelTrainer

MODEL_DIR = "models"
PDF_DIR = "data"
PROCESSED_PDFS_LOG = os.path.join(MODEL_DIR, "processed_pdfs.log")

def get_processed_pdfs() -> set:
    """
    Carrega a lista de PDFs já processados de um arquivo de log.
    """
    if not os.path.exists(PROCESSED_PDFS_LOG):
        return set()
    with open(PROCESSED_PDFS_LOG, "r", encoding="utf-8") as f:
        return set(f.read().splitlines())

def add_processed_pdf(pdf_path: str):
    """
    Adiciona um PDF à lista de PDFs processados no arquivo de log.
    """
    with open(PROCESSED_PDFS_LOG, "a", encoding="utf-8") as f:
        f.write(pdf_path + "\n")

def train_or_update_model():
    """
    Função para treinar ou atualizar o modelo com PDFs da pasta "data".
    """
    print("\n--- Treinamento/Atualização do Modelo ---")
    trainer = LegalModelTrainer.load_model_and_index(MODEL_DIR)
    if trainer is None:
        trainer = LegalModelTrainer()
        print("Nenhum modelo existente encontrado. Iniciando novo treinamento.")
    else:
        print("Modelo existente carregado. Verificando novos PDFs para atualização.")

    processed_pdfs = get_processed_pdfs()
    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    new_texts = []
    new_paths = []
    processed_count = 0

    if not pdf_files:
        print(f"Nenhum arquivo PDF encontrado na pasta {PDF_DIR}. Adicione PDFs para treinar o modelo.")
        return trainer

    for pdf_file in pdf_files:
        if pdf_file in processed_pdfs:
            print(f"PDF {pdf_file} já processado. Pulando.")
            continue

        print(f"Processando PDF: {pdf_file}")
        raw_text = extract_text_from_pdf(pdf_file)
        if raw_text:
            processed_text = preprocess_text(raw_text)
            if processed_text:
                new_texts.append(processed_text)
                new_paths.append(pdf_file)
                processed_count += 1
                add_processed_pdf(pdf_file) # Marca como processado
            else:
                print(f"Texto pré-processado vazio para {pdf_file}. Ignorando.")
        else:
            print(f"Nenhum texto extraído de {pdf_file}. Ignorando.")

    if new_texts:
        print(f"Processando {processed_count} novos PDFs...")
        trainer.train_and_index(new_texts, new_paths)
        trainer.save_model_and_index(MODEL_DIR)
        print("Modelo atualizado com sucesso!")
    else:
        print("Nenhum novo PDF encontrado para processar ou todos já foram processados. Modelo não atualizado.")
    return trainer

def chat_with_model(trainer: LegalModelTrainer):
    """
    Inicia o modo de conversa com o modelo treinado.

    Args:
        trainer (LegalModelTrainer): Instância do treinador do modelo carregado.
    """
    print("\n--- Modo de Conversa (digite \"sair\" para encerrar) ---")
    if trainer.index is None or trainer.index.ntotal == 0:
        print("Erro: Modelo não treinado ou índice vazio. Por favor, treine o modelo primeiro (Opção 1).")
        return

    while True:
        try:
            query = input("Você: ")
            if query.lower() == "sair":
                break

            results = trainer.search_similar_documents(query, k=3) # Busca os 3 documentos mais relevantes

            if results:
                print("\nModelo (informações relevantes encontradas):\n")
                for i, res in enumerate(results):
                    print(f"  Documento {i+1} (Distância: {res['distance']:.4f}):")
                    print(f"    Origem: {res['source_path']}")
                    # Limita a exibição para não sobrecarregar o terminal
                    print(f"    Conteúdo: {res['document'][:500]}...\n") 
                print("\n(Nota: A resposta é baseada nos documentos encontrados. Para uma resposta mais elaborada, seria necessário integrar um modelo de linguagem maior para sumarização ou geração de texto.)\n")
            else:
                print("Nenhuma informação relevante encontrada nos documentos.")
        except Exception as e:
            print(f"Ocorreu um erro durante a conversa: {e}")
            print("Por favor, tente novamente ou digite \"sair\" para encerrar.")

def main():
    """
    Função principal para executar o sistema.
    """
    # Garante que os diretórios existam
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)

    print("Bem-vindo ao Sistema de Consulta Jurídica!")
    print("Certifique-se de colocar seus PDFs na pasta \"data/\".")

    trainer = None
    # Tenta carregar o modelo ao iniciar para que a opção 2 funcione sem treinar novamente
    initial_trainer = LegalModelTrainer.load_model_and_index(MODEL_DIR)
    if initial_trainer:
        trainer = initial_trainer

    while True:
        print("\nEscolha uma opção:")
        print("1. Treinar/Atualizar modelo com PDFs")
        print("2. Conversar com o modelo")
        print("3. Sair")

        choice = input("Opção: ")

        if choice == "1":
            trainer = train_or_update_model()
        elif choice == "2":
            if trainer is None:
                print("Por favor, treine ou carregue o modelo primeiro (Opção 1).")
            else:
                chat_with_model(trainer)
        elif choice == "3":
            print("Saindo...")
            break
        else:
            print("Opção inválida. Tente novamente.")

if __name__ == "__main__":
    main()
```

### `src/app.py`

```python
import streamlit as st
import os
import glob
from src.pdf_processor import extract_text_from_pdf, preprocess_text
from src.model_trainer import LegalModelTrainer

# --- Configurações --- 
MODEL_DIR = "models"
PDF_DIR = "data"
PROCESSED_PDFS_LOG = os.path.join(MODEL_DIR, "processed_pdfs.log")

# Garante que os diretórios existam
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

# --- Funções Auxiliares --- 
@st.cache_resource # Cache o modelo para evitar recarregamento a cada interação
def load_model():
    trainer = LegalModelTrainer.load_model_and_index(MODEL_DIR)
    if trainer is None:
        st.warning("Nenhum modelo encontrado. Por favor, treine o modelo na seção \"Treinar/Atualizar Modelo\".")
        return LegalModelTrainer() # Retorna uma instância vazia para evitar erros
    return trainer

def get_processed_pdfs() -> set:
    if not os.path.exists(PROCESSED_PDFS_LOG):
        return set()
    with open(PROCESSED_PDFS_LOG, "r", encoding="utf-8") as f:
        return set(f.read().splitlines())

def add_processed_pdf(pdf_path: str):
    with open(PROCESSED_PDFS_LOG, "a", encoding="utf-8") as f:
        f.write(pdf_path + "\n")

# --- Interface Streamlit --- 
st.set_page_config(page_title="Consulta Jurídica")
st.title("Sistema de Consulta Jurídica")

# Abas para organização
tab1, tab2 = st.tabs(["Treinar/Atualizar Modelo", "Conversar com o Modelo"])

with tab1:
    st.header("Treinar ou Atualizar Modelo")
    st.write("Faça upload de seus PDFs jurídicos aqui para treinar ou atualizar o modelo. "\
               "Os PDFs serão processados e suas informações serão usadas para o chat.")

    uploaded_files = st.file_uploader("Escolha arquivos PDF", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        st.info(f"{len(uploaded_files)} arquivo(s) selecionado(s).")
        # Salvar PDFs temporariamente para processamento
        for uploaded_file in uploaded_files:
            file_path = os.path.join(PDF_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Arquivo salvo: {uploaded_file.name}")

    if st.button("Iniciar Treinamento/Atualização"):
        with st.spinner("Processando PDFs e atualizando o modelo... Isso pode levar um tempo."):
            trainer = load_model()
            processed_pdfs = get_processed_pdfs()
            pdf_files_in_data = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
            
            new_texts = []
            new_paths = []
            processed_count = 0

            for pdf_file in pdf_files_in_data:
                if pdf_file in processed_pdfs:
                    continue # Já processado

                st.write(f"Processando: {os.path.basename(pdf_file)}")
                raw_text = extract_text_from_pdf(pdf_file)
                if raw_text:
                    processed_text = preprocess_text(raw_text)
                    if processed_text:
                        new_texts.append(processed_text)
                        new_paths.append(pdf_file)
                        processed_count += 1
                        add_processed_pdf(pdf_file) # Marca como processado
                else:
                    st.warning(f"Nenhum texto extraído de {os.path.basename(pdf_file)}. Ignorando.")
            
            if new_texts:
                trainer.train_and_index(new_texts, new_paths)
                trainer.save_model_and_index(MODEL_DIR)
                st.success("Modelo atualizado com sucesso com {processed_count} novos documentos!")
                st.cache_resource.clear() # Limpa o cache para recarregar o modelo atualizado
            else:
                st.info("Nenhum novo PDF encontrado para processar ou todos já foram processados.")

with tab2:
    st.header("Conversar com o Modelo")
    st.write("Faça perguntas sobre os documentos jurídicos que você treinou.")

    trainer = load_model()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Sua pergunta:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if trainer.index is None or trainer.index.ntotal == 0:
                response = "Desculpe, o modelo ainda não foi treinado ou não há documentos. Por favor, treine o modelo na aba \"Treinar/Atualizar Modelo\"."
            else:
                results = trainer.search_similar_documents(prompt, k=3)
                if results:
                    response = "Informações relevantes encontradas:\n\n"
                    for i, res in enumerate(results):
                        response += f"**Documento {i+1}** (Distância: {res["distance"]:.4f})\n"
                        response += f"*Origem:* {os.path.basename(res["source_path"])}\n"
                        response += f"*Conteúdo:* {res["document"][:500]}...\n\n"
                    response += "(Nota: A resposta é baseada nos documentos encontrados. Para uma resposta mais elaborada, seria necessário integrar um modelo de linguagem maior para sumarização ou geração de texto.)"
                else:
                    response = "Nenhuma informação relevante encontrada nos documentos."
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
```

## Considerações Finais

Este sistema fornece uma base robusta para a construção de um modelo de consulta jurídica. Para aprimoramentos futuros, você pode considerar:

- **Modelos de Linguagem Maiores:** Integrar com modelos de linguagem maiores (como GPT-3, BERT, etc.) para sumarização de texto ou geração de respostas mais coerentes, se houver acesso a APIs ou recursos computacionais.
- **Segmentação de Documentos:** Para PDFs muito longos, pode ser útil segmentar o texto em chunks menores antes de gerar embeddings, para que a busca por similaridade seja mais granular e retorne trechos mais específicos.
- **Tratamento de Erros Avançado:** Implementar um tratamento de erros mais sofisticado para diferentes tipos de problemas em PDFs (imagens, PDFs escaneados, etc.).
- **Otimização de Performance:** Para volumes de dados extremamente grandes, explorar otimizações no FAISS ou outras bibliotecas de busca de vizinhos mais próximos.

Espero que este código detalhado e aprimorado seja útil para o seu projeto!

