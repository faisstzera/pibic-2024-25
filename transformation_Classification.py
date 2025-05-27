import os
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
from aeon.transformations.collection.convolution_based import Rocket, MiniRocket
from aeon.datasets import load_classification, load_from_tsv_file
from tqdm import tqdm
from pywt import cwt
from pyts.image import MarkovTransitionField, GramianAngularField, RecurrencePlot
import time
from datetime import datetime

# função para normalizar as séries na mesma escala
def znorm(x):
    return (x - np.mean(x)) / np.std(x)

# função que transforma uma série de entrada em uma imagem em 2D.
def transform_series(series, representation):
    series = np.array(znorm(series))
    if representation == "CWT":
        coeffs, _ = cwt(series, scales=np.arange(1, len(series) + 1), wavelet="morl")
        return coeffs
    elif representation == "MTF":
        series = series.reshape(1, -1)
        mtf = MarkovTransitionField(strategy="normal")
        return mtf.fit_transform(series)[0]
    elif representation == "GADF":
        series = series.reshape(1, -1)
        gaf = GramianAngularField(method="difference")
        return gaf.fit_transform(series)[0]
    elif representation == "GASF":
        series = series.reshape(1, -1)
        gaf = GramianAngularField(method="summation")
        return gaf.fit_transform(series)[0]
    elif representation == "RP":
        series = series.reshape(1, -1)
        rp = RecurrencePlot(threshold="distance")
        return rp.fit_transform(series)[0]
    elif representation == "FIRTS":
        series = series.reshape(1, -1)
        mtf = MarkovTransitionField(n_bins=4, strategy="uniform")
        gaf = GramianAngularField(method="difference")
        rp = RecurrencePlot(threshold="distance")
        return (mtf.fit_transform(series)[0] +
                gaf.fit_transform(series)[0] +
                rp.fit_transform(series)[0])

# função para concatenar dimensões das séries temporais com pre transform ou post transform
def dimensions_concatenate(data, concatenate_type, representation):
    new_data = []
    for x in data:
        if concatenate_type == "pre_transform":
            concatenated_series = np.concatenate(x, axis=0)
            image = transform_series(concatenated_series, representation)
            new_data.append(image.flatten())
        elif concatenate_type == "post_transform":
            transformed_images = [transform_series(series, representation) for series in x]
            concatenated_image = np.concatenate(transformed_images, axis=0)
            new_data.append(concatenated_image.flatten())
    return np.array(new_data)

# função para carregar o dataset
def load_dataset(dataset_name):
    try:
        started_at = time.time()
        print(f"Carregando {dataset_name}")
        print(f"Iniciando em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        train_path = f"{DATA_PATH}/{dataset_name}/{dataset_name}_TRAIN.ts"
        test_path = f"{DATA_PATH}/{dataset_name}/{dataset_name}_TEST.ts"

        if os.path.exists(train_path) and os.path.exists(test_path):
            X_train, y_train = load_from_tsv_file(train_path)
            X_test, y_test = load_from_tsv_file(test_path)
        else:
            print(f"Não foi possível carregar o dataset {dataset_name} armazenados na máquina local")
            print(f"Iniciando download do dataset {dataset_name}")

            X_train, y_train = load_classification(dataset_name, split="Train")
            X_test, y_test = load_classification(dataset_name, split="Test")

            print("Download finalizado com sucesso")

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }
    finally:
        print(f"Tempo de carregamento: {time.time() - started_at} segundos")

# config inicial
DATA_PATH = "/home/faisst/pibic/datasets/data"
representations = ['CWT', 'RP', 'MTF', 'GASF', 'GADF', 'FIRTS']
concatenate_types = ["pre_transform", "post_transform"]

# classificador base
classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))

results = pd.DataFrame(columns=[
    "dataset",
    "representation", 
    "representation_transform_time",
    "concatenation_type",
    "accuracy",
    "convolution_algorithm",
    "convolution_time", 
    "classification_algorithm",
    "train_time",
    "validation_time"
])

# se quiser rodar para mais datasets, basta incluir ou remover dessa listas
datasets = ["Cricket"]
for dataset in datasets:
    try:
        print(f"Processando dataset {dataset}")
        dataset_start = time.time()
        
        data = load_dataset(dataset)
        X_train, y_train = data["X_train"], data["y_train"]
        X_test, y_test = data["X_test"], data["y_test"]

        print(f"Dataset {dataset} carregado em {time.time() - dataset_start} segundos")

        for representation in tqdm(representations, desc="Processing representations"):
            for concat_type in concatenate_types:
                transform_start = time.time()
                X_train_transformed = dimensions_concatenate(X_train, concat_type, representation)
                X_test_transformed = dimensions_concatenate(X_test, concat_type, representation)
                transform_time = time.time() - transform_start

                # Direct Ridge Classification
                train_start = time.time()
                classifier.fit(X_train_transformed, y_train)
                train_time = time.time() - train_start

                valid_start = time.time()
                accuracy = classifier.score(X_test_transformed, y_test)
                valid_time = time.time() - valid_start

                new_result = {
                    "dataset": dataset,
                    "representation": representation,
                    "representation_transform_time": transform_time,
                    "concatenation_type": concat_type,
                    "accuracy": accuracy,
                    "convolution_algorithm": None,
                    "convolution_time": 0,
                    "classification_algorithm": "Ridge",
                    "train_time": train_time,
                    "validation_time": valid_time
                }
                results.loc[len(results)] = new_result

                # classificação com convolução (Rocket e MiniRocket)
                for conv_algo in [Rocket, MiniRocket]:
                    algo_name = conv_algo.__name__
                    conv_start = time.time()
                    algorithm = conv_algo(n_kernels=10000, n_jobs=-1, random_state=6)
                    algorithm.fit(X_train_transformed)
                    X_train_conv = algorithm.transform(X_train_transformed)
                    X_test_conv = algorithm.transform(X_test_transformed)
                    conv_time = time.time() - conv_start

                    train_start = time.time()
                    classifier.fit(X_train_conv, y_train)
                    train_time = time.time() - train_start

                    valid_start = time.time()
                    accuracy = classifier.score(X_test_conv, y_test)
                    valid_time = time.time() - valid_start

                    # Store convolution results
                    new_result = {
                        "dataset": dataset,
                        "representation": representation,
                        "representation_transform_time": transform_time,
                        "concatenation_type": concat_type,
                        "accuracy": accuracy,
                        "convolution_algorithm": algo_name,
                        "convolution_time": conv_time,
                        "classification_algorithm": "Ridge",
                        "train_time": train_time,
                        "validation_time": valid_time
                    }
                    results.loc[len(results)] = new_result

    except Exception as e:
        print(f"Error processing dataset {dataset}: {e}")

file_name = f"benchmark_comparacao_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
results.to_csv(file_name, index=False)