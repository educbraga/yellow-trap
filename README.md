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
