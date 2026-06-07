<p align="center">
  <img src="logo.png" alt="Logo do Classificador de Insetos em Armadilhas Amarelas">
</p>

# Classificador de Insetos em Armadilhas Amarelas

Trabalho academico de framework web com inteligencia artificial. A aplicacao recebe uma imagem de inseto ja recortada, extrai caracteristicas visuais com uma ResNet50 pre-treinada, classifica a imagem com KNN e mostra exemplos parecidos encontrados na base.

O objetivo e demonstrar uma integracao simples entre aplicacao web, processamento de imagens e aprendizado de maquina. O sistema nao faz diagnostico agronomico e nao substitui identificacao profissional de pragas.

## Tecnologias

- Python
- Emmett
- PyTorch e torchvision
- scikit-learn
- Pillow
- matplotlib e seaborn

## Estrutura principal

```text
app.py                  Rotas web e tela de upload
classifier.py           Preparacao da base, extracao de caracteristicas, treino e predicao
scripts/pipeline.py     Comando para preparar, treinar e avaliar
templates/              Telas HTML
static/app.css          Estilos da interface
requirements.txt        Dependencias do projeto
```

O repositorio inclui o dataset original em `yellow-sticky-traps-dataset-main/` e tambem os artefatos treinados necessarios para abrir a aplicacao ja pronta. Arquivos de ambiente local, cache, uploads e zips ficam fora do Git.

## Instalar

Crie e ative o ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instale as dependencias:

```bash
pip install -r requirements.txt
```

## Preparar a base, treinar e avaliar

A aplicacao ja vem com artefatos treinados. Se quiser regenerar tudo do zero, execute:

```bash
python scripts/pipeline.py all
```

Esse comando faz tres etapas:

1. Le as anotacoes XML do dataset original.
2. Recorta os insetos anotados e salva imagens padronizadas em `data/crops/`.
3. Treina o KNN, salva os artefatos em `artifacts/` e gera metricas em `reports/`.

Tambem e possivel rodar cada etapa separadamente:

```bash
python scripts/pipeline.py prepare
python scripts/pipeline.py train
python scripts/pipeline.py evaluate
```

O treino usa separacao por imagem original da armadilha. Assim, recortes da mesma foto nao entram ao mesmo tempo no treino e no teste.

## Rodar a aplicacao

Com o ambiente virtual ativo:

```bash
emmett develop
```

Acesse:

```text
http://127.0.0.1:8000
```

Na tela inicial voce pode:

- enviar uma imagem recortada de inseto;
- usar uma imagem de exemplo da propria base, caso o pipeline ja tenha sido executado;
- abrir a tela de metricas.

## Como demonstrar no video

Uma sequencia curta e facil de explicar:

1. Mostrar que o dataset tem imagens de armadilhas e anotacoes.
2. Explicar que o pipeline gera recortes dos insetos.
3. Mostrar que a ResNet50 transforma cada imagem em um vetor de caracteristicas.
4. Explicar que o KNN compara esse vetor com os exemplos treinados.
5. Abrir a interface, classificar uma imagem e mostrar a classe prevista, a confianca aproximada e as imagens parecidas.
6. Abrir a pagina de metricas para mostrar a avaliacao offline.

## Problemas comuns

Se a aplicacao avisar que os artefatos nao existem, rode:

```bash
python scripts/pipeline.py all
```

Se o dataset nao for encontrado, confirme que a pasta abaixo esta na raiz do projeto:

```text
yellow-sticky-traps-dataset-main/
```

Se o comando `emmett develop` nao for encontrado, ative o ambiente virtual e reinstale as dependencias:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```
