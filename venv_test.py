# Primeiro, vamos verificar as versões
import aeon
import numpy as np
import pandas as pd

print(f"Aeon version: {aeon.__version__}")
print(f"Numpy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")

# Agora testa seu código original
import os
import logging
import time
from datetime import datetime
from aeon.datasets import load_classification, load_from_tsv_file

DATA_PATH = "/home/user/Desktop/AlexandrePibic/pibic-2024-25/datasets/data"

# config do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'test_venv_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def log_print(message):
    print(message)
    logging.info(message)

def load_dataset(dataset_name):
    try:
        started_at = time.time()
        log_print(f"Carregando {dataset_name}")
        log_print(f"Iniciando em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        train_path = f"{DATA_PATH}/{dataset_name}/{dataset_name}_TRAIN.ts"
        test_path = f"{DATA_PATH}/{dataset_name}/{dataset_name}_TEST.ts"

        if os.path.exists(train_path) and os.path.exists(test_path):
            log_print("Carregando arquivos locais")
            X_train, y_train = load_from_tsv_file(train_path)
            X_test, y_test = load_from_tsv_file(test_path)
        else:
            log_print(f"Não foi possível carregar o dataset {dataset_name} armazenados na máquina local")
            log_print(f"Iniciando download do dataset {dataset_name}")

            X_train, y_train = load_classification(dataset_name, split="Train")
            X_test, y_test = load_classification(dataset_name, split="Test")

            log_print("Download finalizado com sucesso")

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }
    except Exception as e:
        log_print(f"ERRO: {str(e)}")
        raise
    finally:
        log_print(f"Tempo de carregamento: {time.time() - started_at} segundos")

# Testa
dataset = "CharacterTrajectories"
log_print(f"Processando dataset {dataset}")

try:
    data = load_dataset(dataset)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    log_print(f"SUCESSO! Dataset carregado")
    log_print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    log_print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
except Exception as e:
    log_print(f"FALHOU: {str(e)}")
