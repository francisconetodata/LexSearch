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