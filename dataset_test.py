import os
import time
import threading
import shutil
from datetime import datetime

# Configurar cache
cache_path = '/home/user/Desktop/AlexandrePibic/pibic-2024-25/aeon_cache'
backup_path = '/home/user/Desktop/AlexandrePibic/pibic-2024-25/backup_files'
os.environ['AEON_DATA_PATH'] = cache_path
os.makedirs(cache_path, exist_ok=True)
os.makedirs(backup_path, exist_ok=True)

from aeon.datasets import load_classification

def monitor_directory():
    """Monitora o diretório de cache e copia arquivos assim que aparecem"""
    dataset_dir = os.path.join(cache_path, "CharacterTrajectories")
    
    print("=== Iniciando monitoramento do diretório ===")
    
    while True:
        if os.path.exists(dataset_dir):
            files = os.listdir(dataset_dir)
            
            for file in files:
                if file.endswith('.ts'):
                    source_path = os.path.join(dataset_dir, file)
                    backup_file_path = os.path.join(backup_path, f"backup_{file}")
                    
                    if not os.path.exists(backup_file_path):  # Ainda não fez backup
                        print(f"ARQUIVO DETECTADO: {file}")
                        try:
                            # Faz backup imediatamente
                            shutil.copy2(source_path, backup_file_path)
                            print(f"Backup salvo: {backup_file_path}")
                            
                            # Inspeciona o arquivo
                            inspect_case_159(backup_file_path)
                            
                        except Exception as e:
                            print(f"Erro ao fazer backup: {e}")
        
        time.sleep(0.1)  # Verifica a cada 100ms

def inspect_case_159(file_path):
    """Inspeciona o caso 159"""
    print(f"\n=== INSPECIONANDO CASO 159 EM: {file_path} ===")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            case_count = 0
            for line_num, line in enumerate(f):
                line = line.strip()
                
                if line.startswith('@') or not line or line.startswith('#'):
                    continue
                
                if case_count == 159:
                    print(f"CASO 159 (linha {line_num}):")
                    print(f"Conteúdo: {line}")
                    
                    parts = line.split(':')
                    print(f"Partes: {len(parts)}")
                    
                    if len(parts) >= 2:
                        ch1 = parts[0].strip()
                        ch2 = parts[1].strip()
                        
                        points1 = ch1.count('(')
                        points2 = ch2.count('(')
                        
                        print(f"Canal 1: {points1} pontos")
                        print(f"Canal 2: {points2} pontos")
                        print(f"PROBLEMA: Diferença de {abs(points1 - points2)} pontos")
                        
                        # Mostra um pedaço de cada canal
                        print(f"Canal 1 (primeiros 100 chars): {ch1[:100]}")
                        print(f"Canal 2 (primeiros 100 chars): {ch2[:100]}")
                    
                    return
                
                case_count += 1
                
    except Exception as e:
        print(f"Erro ao ler arquivo: {e}")

def force_download():
    """Força o download enquanto monitora"""
    print("=== Iniciando download ===")
    try:
        X_train, y_train = load_classification("CharacterTrajectories", split="Train")
        print("Download bem-sucedido!")
    except Exception as e:
        print(f"Erro esperado: {e}")

if __name__ == "__main__":
    # Inicia monitoramento em thread separada
    monitor_thread = threading.Thread(target=monitor_directory, daemon=True)
    monitor_thread.start()
    
    # Aguarda um pouco para garantir que o monitor está rodando
    time.sleep(1)
    
    # Força o download
    force_download()
    
    # Aguarda um pouco para capturar arquivos
    print("Aguardando possíveis arquivos...")
    time.sleep(5)
    
    # Verifica se conseguiu salvar algo
    if os.path.exists(backup_path):
        backup_files = os.listdir(backup_path)
        if backup_files:
            print(f"\nArquivos salvos no backup: {backup_files}")
        else:
            print("\nNenhum arquivo foi capturado no backup")
    
    print("Fim do monitoramento")
