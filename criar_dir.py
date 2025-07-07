import os

# Estrutura de diretórios e arquivos a ser criada
# As chaves são os diretórios e os valores são listas de subdiretórios ou arquivos
structure = {
    "data": [],
    "models": [
        "sentence_transformer_model/",  # O '/' no final indica um diretório
        "faiss_index.bin",
        "documents_metadata.json",
        "processed_pdfs.log"
    ],
    "src": [
        "__init__.py",
        "pdf_processor.py",
        "model_trainer.py",
        "main.py",
        "app.py"
    ],
    "requirements.txt": None, # 'None' indica que é um arquivo na raiz
    "README.md": None
}

def create_project_structure(base_path, project_structure):
    """
    Cria uma estrutura de diretórios e arquivos para um projeto.

    :param base_path: O caminho onde a estrutura do projeto será criada.
    :param project_structure: Um dicionário definindo a estrutura.
    """
    # Garante que o caminho base seja absoluto para uma exibição clara
    abs_base_path = os.path.abspath(base_path)
    print(f"Criando estrutura do projeto em: {abs_base_path}")

    # Itera sobre a estrutura definida
    for name, content in project_structure.items():
        current_path = os.path.join(base_path, name)

        # Se o conteúdo for None, é um arquivo na raiz
        if content is None:
            if not os.path.exists(current_path):
                with open(current_path, 'w', encoding='utf-8') as f:
                    if name == "README.md":
                        f.write("# Projeto RAG\n\n")
                        f.write("Este é o arquivo de documentação do projeto.\n")
                    elif name == "requirements.txt":
                        f.write("# Adicione as dependências do Python aqui. Exemplo:\n")
                        f.write("streamlit\n")
                        f.write("sentence-transformers\n")
                        f.write("faiss-cpu\n")
                        f.write("PyPDF2\n")
                print(f"   ARQUIVO: {current_path}")
            continue

        # Cria o diretório principal (ex: data, models, src)
        if not os.path.exists(current_path):
            os.makedirs(current_path)
            print(f"DIRETÓRIO: {current_path}")

        # Cria os subdiretórios e arquivos
        for item in content:
            item_path = os.path.join(current_path, item)
            # Se terminar com '/', é um subdiretório
            if item.endswith('/'):
                if not os.path.exists(item_path):
                    os.makedirs(item_path)
                    print(f"  SUBDIR: {item_path}")
            # Senão, é um arquivo
            else:
                if not os.path.exists(item_path):
                    with open(item_path, 'w') as f:
                        pass # Cria o arquivo vazio
                    print(f"   ARQUIVO: {item_path}")

if __name__ == "__main__":
    # Usa o diretório atual ('.') como o caminho base para a criação
    create_project_structure(".", structure)
    print("\nEstrutura de diretórios e arquivos criada com sucesso!")
