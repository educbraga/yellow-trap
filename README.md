<p align="center">
  <img src="logo.png" alt="Logo do Classificador de Insetos em Armadilhas Amarelas">
</p>

# Classificador de Insetos em Armadilhas Amarelas

Este repositório contém um trabalho simples de pós-graduação que combina uma
aplicação web com um classificador de imagens. A ideia é receber uma imagem de
inseto já recortada, extrair características visuais com uma ResNet50
pré-treinada, classificar o recorte com KNN e mostrar imagens parecidas da base.

O projeto serve como demonstração de integração entre framework web,
processamento de imagens e aprendizado de máquina. Ele não deve ser tratado como
uma solução pronta para uso em campo, não faz diagnóstico agronômico e não
substitui identificação profissional de pragas.

## O que a aplicação faz

- Recebe upload de imagens JPG, JPEG, PNG ou WEBP.
- Classifica recortes de insetos em três classes do dataset.
- Mostra a classe prevista e uma confiança aproximada calculada pelo KNN.
- Exibe as imagens de referência mais parecidas com o recorte enviado.
- Apresenta uma página simples com métricas offline do experimento.

Importante: a aplicação espera uma imagem de inseto já recortada. Ela não detecta
automaticamente insetos em uma foto completa da armadilha.

## Classes usadas

| Sigla | Classe |
| --- | --- |
| MR | Macrolophus pygmaeus |
| NC | Nesidiocoris tenuis |
| WF | Trialeurodes vaporariorum |

## Tecnologias

- Python
- Emmett
- PyTorch e torchvision
- scikit-learn
- Pillow
- matplotlib e seaborn

## Como funciona

O dataset original contém imagens de armadilhas amarelas e anotações em XML no
formato PASCAL VOC. O pipeline do projeto usa essas anotações para gerar recortes
dos insetos e treinar o classificador.

Fluxo geral:

1. Ler as anotações XML do dataset.
2. Recortar os insetos anotados e padronizar cada imagem em 224 x 224 pixels.
3. Usar uma ResNet50 pré-treinada como extrator de características.
4. Treinar um KNN com métrica de cosseno sobre os embeddings extraídos.
5. Avaliar o modelo em um conjunto de teste separado por imagem original.
6. Usar os artefatos gerados na aplicação web.

A separação por imagem original evita que recortes da mesma foto da armadilha
apareçam ao mesmo tempo no treino e no teste.

## Resultados do experimento

Os resultados abaixo foram gerados a partir dos artefatos e métricas presentes no
repositório. Eles ajudam a avaliar o comportamento do classificador neste
conjunto de dados, mas não significam que o sistema esteja validado para uso real
em campo.

- Acurácia: 93,63%
- AUC macro one-vs-rest: 0,9871
- Recortes de treino: 1439
- Recortes de teste: 361
- Imagens originais no treino: 212
- Imagens originais no teste: 52
- Sobreposição de imagens originais entre treino e teste: 0

As métricas detalhadas ficam em `reports/metrics.json`, e as imagens da matriz de
confusão e da curva ROC ficam em `reports/`.

## Estrutura principal

```text
app.py                  Rotas web e tela de upload
classifier.py           Preparação da base, extração de características, treino e predição
scripts/pipeline.py     Comando para preparar, treinar e avaliar
templates/              Telas HTML
static/app.css          Estilos da interface
requirements.txt        Dependências do projeto
```

O repositório inclui o dataset original em `yellow-sticky-traps-dataset-main/` e
artefatos treinados em `artifacts/`. Arquivos de ambiente local, cache, uploads e
zips ficam fora do Git.

## Créditos do dataset

O dataset usado neste projeto vem de uma versão com rótulos revisados do Yellow
Sticky Traps Dataset, descrita no README original em
`yellow-sticky-traps-dataset-main/README.md`.

Segundo a documentação do dataset, essa versão foi criada por Maurice Deserno e
Alexia Briassouli, com ajuda de Carolin Vey, a partir do conjunto original "Raw
data from Yellow Sticky Traps with insects for training of deep learning
Convolutional Neural Network for object detection", publicado por A.T.
Nieuwenhuizen e colaboradores.

Para detalhes completos de autoria, artigo, DOI e fonte dos dados, consulte o
README original do dataset dentro da pasta `yellow-sticky-traps-dataset-main/`.

## Instalar

Crie e ative o ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

## Preparar a base, treinar e avaliar

A aplicação já vem com artefatos treinados. Se quiser regenerar tudo do zero,
execute:

```bash
python scripts/pipeline.py all
```

Também é possível rodar cada etapa separadamente:

```bash
python scripts/pipeline.py prepare
python scripts/pipeline.py train
python scripts/pipeline.py evaluate
```

As etapas geram:

- `data/crops/`: recortes dos insetos;
- `data/crops_manifest.csv`: manifesto dos recortes;
- `static/reference/`: imagens usadas como exemplos na interface;
- `artifacts/`: modelo KNN, embeddings e metadados;
- `reports/`: métricas e gráficos de avaliação.

## Rodar a aplicação

Com o ambiente virtual ativo:

```bash
emmett develop
```

Acesse:

```text
http://127.0.0.1:8000
```

Na tela inicial você pode enviar uma imagem recortada de inseto, usar exemplos da
base e abrir a tela de métricas.

## Problemas comuns

Se a aplicação avisar que os artefatos não existem, rode:

```bash
python scripts/pipeline.py all
```

Se o dataset não for encontrado, confirme que a pasta abaixo está na raiz do
projeto:

```text
yellow-sticky-traps-dataset-main/
```

Se o comando `emmett develop` não for encontrado, ative o ambiente virtual e
reinstale as dependências:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

<hr>

# Yellow Sticky Traps Insect Classifier

This repository contains a simple postgraduate project that combines a web
application with an image classifier. The idea is to receive an already cropped
insect image, extract visual features with a pre-trained ResNet50, classify the
crop with KNN, and show similar images from the dataset.

The project serves as a demonstration of integration between a web framework,
image processing, and machine learning. It should not be treated as a ready-to-use
field solution, does not provide agronomic diagnosis, and does not replace
professional pest identification.

## What the application does

- Receives uploads of JPG, JPEG, PNG, or WEBP images.
- Classifies insect crops into three dataset classes.
- Shows the predicted class and an approximate confidence score calculated by KNN.
- Displays the reference images most similar to the submitted crop.
- Presents a simple page with offline experiment metrics.

Important: the application expects an already cropped insect image. It does not
automatically detect insects in a complete photo of the trap.

## Classes used

| Code | Class |
| --- | --- |
| MR | Macrolophus pygmaeus |
| NC | Nesidiocoris tenuis |
| WF | Trialeurodes vaporariorum |

## Technologies

- Python
- Emmett
- PyTorch and torchvision
- scikit-learn
- Pillow
- matplotlib and seaborn

## How it works

The original dataset contains images of yellow sticky traps and XML annotations in
PASCAL VOC format. The project pipeline uses these annotations to generate insect
crops and train the classifier.

General flow:

1. Read the dataset XML annotations.
2. Crop the annotated insects and standardize each image to 224 x 224 pixels.
3. Use a pre-trained ResNet50 as a feature extractor.
4. Train a KNN with cosine metric over the extracted embeddings.
5. Evaluate the model on a test set separated by original image.
6. Use the generated artifacts in the web application.

The separation by original image prevents crops from the same trap photo from
appearing in both the training and test sets.

## Experiment results

The results below were generated from the artifacts and metrics present in the
repository. They help evaluate the behavior of the classifier on this dataset,
but they do not mean that the system is validated for real field use.

- Accuracy: 93.63%
- Macro one-vs-rest AUC: 0.9871
- Training crops: 1439
- Test crops: 361
- Original images in training: 212
- Original images in test: 52
- Overlap of original images between training and test: 0

The detailed metrics are in `reports/metrics.json`, and the confusion matrix and
ROC curve images are in `reports/`.

## Main structure

```text
app.py                  Web routes and upload screen
classifier.py           Dataset preparation, feature extraction, training, and prediction
scripts/pipeline.py     Command to prepare, train, and evaluate
templates/              HTML screens
static/app.css          Interface styles
requirements.txt        Project dependencies
```

The repository includes the original dataset in `yellow-sticky-traps-dataset-main/`
and trained artifacts in `artifacts/`. Local environment files, cache, uploads,
and zip files are kept out of Git.

## Dataset credits

The dataset used in this project comes from a version with revised labels of the
Yellow Sticky Traps Dataset, described in the original README at
`yellow-sticky-traps-dataset-main/README.md`.

According to the dataset documentation, this version was created by Maurice
Deserno and Alexia Briassouli, with help from Carolin Vey, based on the original
set "Raw data from Yellow Sticky Traps with insects for training of deep learning
Convolutional Neural Network for object detection", published by A.T.
Nieuwenhuizen and collaborators.

For complete details about authorship, article, DOI, and data source, see the
original dataset README inside the `yellow-sticky-traps-dataset-main/` folder.

## Install

Create and activate the virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Prepare the dataset, train, and evaluate

The application already comes with trained artifacts. To regenerate everything
from scratch, run:

```bash
python scripts/pipeline.py all
```

It is also possible to run each step separately:

```bash
python scripts/pipeline.py prepare
python scripts/pipeline.py train
python scripts/pipeline.py evaluate
```

The steps generate:

- `data/crops/`: insect crops;
- `data/crops_manifest.csv`: crop manifest;
- `static/reference/`: images used as examples in the interface;
- `artifacts/`: KNN model, embeddings, and metadata;
- `reports/`: metrics and evaluation charts.

## Run the application

With the virtual environment active:

```bash
emmett develop
```

Access:

```text
http://127.0.0.1:8000
```

On the home screen, you can upload a cropped insect image, use examples from the
dataset, and open the metrics screen.

## Common problems

If the application warns that the artifacts do not exist, run:

```bash
python scripts/pipeline.py all
```

If the dataset is not found, confirm that the folder below is in the project root:

```text
yellow-sticky-traps-dataset-main/
```

If the `emmett develop` command is not found, activate the virtual environment and
reinstall the dependencies:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```
