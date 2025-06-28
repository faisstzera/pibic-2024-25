#!/usr/bin/env python3
"""
Script para consolidar todos os arquivos Excel de benchmark no arquivo ResultadosFinais.xlsx
Evita datasets repetidos verificando se o dataset já existe no arquivo de resultados finais.
"""

import pandas as pd
import glob
import os
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('consolidacao_resultados.log'),
        logging.StreamHandler()
    ]
)

def encontrar_arquivos_benchmark(diretorio_base):
    """
    Encontra todos os arquivos Excel de benchmark na estrutura de diretórios.
    
    Args:
        diretorio_base (str): Diretório base para buscar os arquivos
        
    Returns:
        list: Lista de caminhos dos arquivos encontrados
    """
    padroes = [
        'benchmark_comparacao_*.xlsx',
        'results/benchmark_comparacao_*.xlsx',
        'resultados_finais/benchmark_comparacao_*.xlsx'
    ]
    
    arquivos = []
    for padrao in padroes:
        arquivos.extend(glob.glob(os.path.join(diretorio_base, padrao)))
    
    # Remover duplicatas e ordenar
    arquivos = sorted(list(set(arquivos)))
    
    logging.info(f"Encontrados {len(arquivos)} arquivos de benchmark")
    return arquivos

def carregar_resultados_finais(caminho_arquivo):
    """
    Carrega o arquivo de resultados finais ou cria um DataFrame vazio se não existir.
    
    Args:
        caminho_arquivo (str): Caminho para o arquivo ResultadosFinais.xlsx
        
    Returns:
        pd.DataFrame: DataFrame com os resultados finais
    """
    try:
        if os.path.exists(caminho_arquivo):
            df = pd.read_excel(caminho_arquivo)
            logging.info(f"Arquivo ResultadosFinais.xlsx carregado com {len(df)} registros")
            return df
        else:
            logging.info("Arquivo ResultadosFinais.xlsx não encontrado, criando novo DataFrame")
            # Criar DataFrame vazio com as colunas esperadas
            colunas = [
                'dataset', 'representation', 'representation_transform_time',
                'concatenation_type', 'accuracy', 'convolution_algorithm',
                'convolution_time', 'classification_algorithm', 'train_time',
                'validation_time'
            ]
            return pd.DataFrame(columns=colunas)
    except Exception as e:
        logging.error(f"Erro ao carregar ResultadosFinais.xlsx: {e}")
        raise

def processar_arquivo_benchmark(caminho_arquivo, datasets_existentes):
    """
    Processa um arquivo de benchmark e retorna os dados novos (não duplicados).
    
    Args:
        caminho_arquivo (str): Caminho para o arquivo de benchmark
        datasets_existentes (set): Conjunto de datasets já existentes
        
    Returns:
        pd.DataFrame: DataFrame com os novos dados
    """
    try:
        df = pd.read_excel(caminho_arquivo)
        
        # Verificar se tem as colunas esperadas
        colunas_esperadas = [
            'dataset', 'representation', 'representation_transform_time',
            'concatenation_type', 'accuracy', 'convolution_algorithm',
            'convolution_time', 'classification_algorithm', 'train_time',
            'validation_time'
        ]
        
        if not all(col in df.columns for col in colunas_esperadas):
            logging.warning(f"Arquivo {caminho_arquivo} não tem todas as colunas esperadas")
            return pd.DataFrame()
        
        # Filtrar apenas os datasets que não existem ainda
        datasets_no_arquivo = set(df['dataset'].unique())
        novos_datasets = datasets_no_arquivo - datasets_existentes
        
        if novos_datasets:
            df_novo = df[df['dataset'].isin(novos_datasets)]
            logging.info(f"Arquivo {os.path.basename(caminho_arquivo)}: {len(novos_datasets)} novos datasets: {novos_datasets}")
            return df_novo
        else:
            logging.info(f"Arquivo {os.path.basename(caminho_arquivo)}: nenhum dataset novo encontrado")
            return pd.DataFrame()
            
    except Exception as e:
        logging.error(f"Erro ao processar arquivo {caminho_arquivo}: {e}")
        return pd.DataFrame()

def consolidar_resultados(diretorio_base):
    """
    Função principal para consolidar todos os resultados.
    
    Args:
        diretorio_base (str): Diretório base do projeto
    """
    # Caminhos dos arquivos
    caminho_resultados_finais = os.path.join(diretorio_base, 'resultados_finais', 'ResultadosFinais.xlsx')
    
    # Carregar resultados finais existentes
    df_resultados_finais = carregar_resultados_finais(caminho_resultados_finais)
    
    # Obter datasets já existentes
    datasets_existentes = set(df_resultados_finais['dataset'].unique()) if not df_resultados_finais.empty else set()
    logging.info(f"Datasets já existentes: {len(datasets_existentes)}")
    
    # Encontrar todos os arquivos de benchmark
    arquivos_benchmark = encontrar_arquivos_benchmark(diretorio_base)
    
    # Lista para armazenar os novos DataFrames
    novos_dfs = []
    
    # Processar cada arquivo
    for arquivo in arquivos_benchmark:
        df_novo = processar_arquivo_benchmark(arquivo, datasets_existentes)
        if not df_novo.empty:
            novos_dfs.append(df_novo)
            # Atualizar o conjunto de datasets existentes
            datasets_existentes.update(df_novo['dataset'].unique())
    
    # Consolidar todos os novos dados
    if novos_dfs:
        df_novos_dados = pd.concat(novos_dfs, ignore_index=True)
        
        # Combinar com os dados existentes
        if not df_resultados_finais.empty:
            df_final = pd.concat([df_resultados_finais, df_novos_dados], ignore_index=True)
        else:
            df_final = df_novos_dados
        
        # Criar diretório se não existir
        os.makedirs(os.path.dirname(caminho_resultados_finais), exist_ok=True)
        
        # Salvar o arquivo consolidado
        df_final.to_excel(caminho_resultados_finais, index=False)
        
        logging.info(f"Consolidação concluída!")
        logging.info(f"Total de registros no arquivo final: {len(df_final)}")
        logging.info(f"Novos registros adicionados: {len(df_novos_dados)}")
        logging.info(f"Total de datasets únicos: {len(df_final['dataset'].unique())}")
        
        # Relatório de datasets
        datasets_finais = sorted(df_final['dataset'].unique())
        logging.info(f"Datasets no arquivo final: {datasets_finais}")
        
    else:
        logging.info("Nenhum dado novo encontrado para consolidar")

def main():
    """Função principal do script"""
    diretorio_base = '/home/faisst/pibic'
    
    logging.info("="*50)
    logging.info("INICIANDO CONSOLIDAÇÃO DE RESULTADOS")
    logging.info("="*50)
    
    try:
        consolidar_resultados(diretorio_base)
        logging.info("Consolidação concluída com sucesso!")
    except Exception as e:
        logging.error(f"Erro durante a consolidação: {e}")
        raise

if __name__ == "__main__":
    main()
