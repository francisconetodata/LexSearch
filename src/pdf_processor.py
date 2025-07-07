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