import os
# Limpar cache primeiro
cache_path = '/home/user/Desktop/AlexandrePibic/pibic-2024-25/aeon_cache'
if os.path.exists(cache_path):
    import shutil
    shutil.rmtree(cache_path)
    print("Cache limpo")

os.environ['AEON_DATA_PATH'] = cache_path
os.makedirs(cache_path, exist_ok=True)

import logging
import time
from datetime import datetime
from aeon.datasets import load_classification  # AQUI estava faltando!
import numpy as np

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'debug_dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def log_print(message):
    print(message)
    logging.info(message)

def inspect_raw_file(file_path):
    """Inspeciona o arquivo .ts baixado linha por linha"""
    log_print(f"\n=== Inspecionando arquivo: {file_path} ===")
    
    if not os.path.exists(file_path):
        log_print(f"Arquivo não encontrado: {file_path}")
        return
    
    file_size = os.path.getsize(file_path)
    log_print(f"Tamanho do arquivo: {file_size} bytes")
    
    # Lê as primeiras linhas
    log_print("Primeiras 10 linhas do arquivo:")
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 10:
                log_print(f"Linha {i}: {line.strip()}")
            else:
                break
    
    # Procura pelo caso 159
    log_print(f"\n=== Procurando caso 159 ===")
    with open(file_path, 'r', encoding='utf-8') as f:
        current_case = 0
        for line_num, line in enumerate(f):
            line = line.strip()
            if line.startswith('@') or not line or line.startswith('#'):
                continue
            
            if current_case == 159:
                log_print(f"Linha {line_num}: Caso 159 encontrado")
                log_print(f"Conteúdo: {line}")
                
                # Analisa a estrutura
                parts = line.split(':')
                log_print(f"Número de partes separadas por ':': {len(parts)}")
                
                if len(parts) >= 3:
                    channel1 = parts[0].strip()
                    channel2 = parts[1].strip() 
                    class_label = parts[2].strip()
                    
                    log_print(f"Canal 1 length: {len(channel1)}")
                    log_print(f"Canal 2 length: {len(channel2)}")
                    log_print(f"Classe: {class_label}")
                    
                    # Conta pontos (x,y)
                    points1 = channel1.count('(')
                    points2 = channel2.count('(')
                    log_print(f"Canal 1: {points1} pontos")
                    log_print(f"Canal 2: {points2} pontos")
                
                break
            
            if line:
                current_case += 1

def main():
    log_print("=== Debug do dataset CharacterTrajectories ===")
    
    try:
        # Tenta forçar o download
        log_print("Tentando download...")
        X_train, y_train = load_classification("CharacterTrajectories", split="Train")
        log_print("Download bem-sucedido!")
        
    except Exception as e:
        log_print(f"Erro no download: {str(e)}")
        
        # Verifica se arquivos foram baixados parcialmente
        dataset_path = os.path.join(cache_path, "CharacterTrajectories")
        if os.path.exists(dataset_path):
            log_print("Arquivos encontrados no cache:")
            for file in os.listdir(dataset_path):
                file_path = os.path.join(dataset_path, file)
                size = os.path.getsize(file_path)
                log_print(f"  {file} ({size} bytes)")
                
                if file.endswith('.ts'):
                    inspect_raw_file(file_path)

if __name__ == "__main__":
    main()
