# Projeto PIBIC - Classificação de Séries Temporais com Representações Visuais

## 📋 Visão Geral

Este projeto de Programa Institucional de Bolsas de Iniciação Científica (PIBIC) investiga a classificação de séries temporais multivariadas utilizando diferentes representações visuais combinadas com técnicas de convolução. O objetivo é avaliar o desempenho de várias abordagens para transformar séries temporais em representações visuais e aplicar algoritmos de classificação.

## 🎯 Objetivos

- Comparar diferentes representações visuais de séries temporais (CWT, GASF, GADF, RP, MTF)
- Avaliar estratégias de concatenação de representações (pré e pós-transformação)
- Analisar o impacto de algoritmos de convolução (MiniRocket, Rocket)
- Benchmarking em múltiplos datasets de séries temporais multivariadas

## 📊 Datasets Utilizados

O projeto utiliza **26 datasets** diferentes de séries temporais multivariadas, incluindo:

- ArticularyWordRecognition
- AtrialFibrillation  
- BasicMotions
- CharacterTrajectories
- Cricket
- DuckDuckGeese
- EigenWorms
- Epilepsy
- EthanolConcentration
- FingerMovements
- HandMovementDirection
- Handwriting
- Heartbeat
- JapaneseVowels
- LSST
- Libras
- NATOPS
- PenDigits
- Phoneme
- RacketSports
- SelfRegulationSCP1
- SelfRegulationSCP2
- SpokenArabicDigits
- StandWalkJump
- UWaveGestureLibrary
- ERing

## 🔧 Metodologia

### Representações Visuais
- **CWT**: Continuous Wavelet Transform
- **GASF**: Gramian Angular Summation Field
- **GADF**: Gramian Angular Difference Field
- **RP**: Recurrence Plot
- **MTF**: Markov Transition Field

### Estratégias de Concatenação
- **Pre-transform**: Concatenação antes da transformação
- **Post-transform**: Concatenação após a transformação

### Algoritmos de Convolução
- **Sem convolução** (baseline)
- **MiniRocket**: Versão simplificada do ROCKET
- **Rocket**: Random Convolutional Kernels Transform

### Algoritmo de Classificação
- **Ridge Regression**: Classificador linear com regularização

## 📁 Estrutura do Projeto

```
pibic/
├── README.md                           # Este arquivo
├── Analysis.ipynb                      # Notebook principal de análise
├── consolidar_resultados.py           # Script para consolidar resultados
├── time_series_pibic.py               # Implementação principal
├── transformation_Classification.py    # Funções de transformação
├── resultados_finais/
│   └── ResultadosFinais.xlsx          # 🎯 RESULTADOS CONSOLIDADOS
├── results/                           # Resultados individuais por dataset
├── benchmark_comparacao_*.xlsx        # Arquivos de benchmark individuais
└── *.log                             # Logs de execução
```

## 🎯 **RESULTADOS PRINCIPAIS**

### 📊 Arquivo de Resultados Consolidados
**Localização**: `resultados_finais/ResultadosFinais.xlsx`

Este arquivo contém **954 registros** consolidados de todos os experimentos realizados nos 26 datasets, com as seguintes colunas:

- `dataset`: Nome do dataset utilizado
- `representation`: Tipo de representação visual aplicada
- `representation_transform_time`: Tempo de transformação da representação (segundos)
- `concatenation_type`: Estratégia de concatenação utilizada
- `accuracy`: Acurácia obtida na classificação
- `convolution_algorithm`: Algoritmo de convolução aplicado
- `convolution_time`: Tempo de convolução (segundos)
- `classification_algorithm`: Algoritmo de classificação (Ridge)
- `train_time`: Tempo de treinamento (segundos)
- `validation_time`: Tempo de validação (segundos)

### 📈 Análise dos Resultados

Para visualizar e analisar os resultados, utilize o notebook `Analysis.ipynb` que contém:

1. **Análises por categoria**:
   - Desempenho por dataset
   - Comparação entre representações
   - Impacto dos algoritmos de convolução
   - Análise de tempos de execução

2. **Visualizações**:
   - Boxplots comparativos
   - Scatterplots de acurácia
   - Matrizes de correlação
   - Análises de trade-offs tempo vs. performance

3. **Estatísticas resumidas**:
   - Melhores configurações por dataset
   - Rankings de performance
   - Análises de correlação

## 🚀 Como Executar

### Pré-requisitos
```bash
pip install pandas numpy seaborn matplotlib scikit-learn aeon
```

### Execução Principal
```bash
python time_series_pibic.py
```

### Consolidação de Resultados
```bash
python consolidar_resultados.py
```

### Análise dos Resultados
```bash
jupyter notebook Analysis.ipynb
```