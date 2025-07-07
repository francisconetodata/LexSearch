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