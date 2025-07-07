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