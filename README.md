# POS FIAP ALURA - IA PARA DEVS - Tech Challenge Fase 5

## Integrantes Grupo 26

- André Philipe Oliveira de Andrade(RM357002) - andrepoandrade@gmail.com
- Joir Neto (RM356391) - joirneto@gmail.com
- Marcos Jen San Hsie(RM357422) - marcosjsh@gmail.com
- Michael dos Santos Silva(RM357009) - michael.shel96@gmail.com
- Sonival dos Santos(RM356905) - sonival.santos@gmail.com

Video(Youtube): 

Github: https://github.com/apandrade/tech-challenge5

# 🧠 Por que escolhemos o YOLOv8 para detecção de objetos cortantes?

## 📌 Modelos considerados

Antes de definir o modelo ideal para a detecção de objetos cortantes, avaliamos as seguintes abordagens:

- **YOLOv8**: Modelo de detecção supervisionada de última geração, altamente eficiente para tarefas em tempo real. Suporta treinamento com classes customizadas e imagens negativas. Ideal para aplicações práticas com recursos computacionais limitados.

- **CLIP (Contrastive Language–Image Pretraining)**: Modelo multimodal treinado pela OpenAI que associa imagens e textos. Excelente para buscas conceituais e classificação zero-shot, mas não é projetado para detectar múltiplos objetos com precisão espacial.

- **SAM (Segment Anything Model)**: Segmentador universal da Meta AI capaz de isolar qualquer objeto em uma imagem sem necessidade de treinamento. Precisa ser combinado com modelos classificadores (como CLIP) para atribuir rótulos às regiões segmentadas.

- **Florence2**: Modelo multimodal avançado da Microsoft, capaz de executar diversas tarefas de visão computacional. Potente e preciso, mas muito mais complexo e pesado, exigindo infraestrutura robusta.

---

## ⚖️ Comparativo entre modelos

| Modelo/Ferramenta      | Tipo                     | Prós                                                                 | Contras                                                             | Ideal para...                                |
|------------------------|--------------------------|----------------------------------------------------------------------|---------------------------------------------------------------------|------------------------------------------------|
| **YOLOv8**             | Detecção supervisionada  | Rápido, leve, fácil de treinar, ótimo suporte a imagens customizadas | Requer dataset anotado                                             | Casos práticos, vídeos em tempo real          |
| **CLIP + Segmentação** | Visão multimodal         | Entende texto e imagem, bom para zero-shot                           | Não é ideal para detecção precisa de múltiplos objetos              | Busca conceitual ou filtragem sem anotações   |
| **SAM (Segment Anything)** | Segmentação automática | Segmentação precisa de qualquer objeto                               | Não classifica; precisa de CLIP ou labels externos                 | Separar regiões de interesse                  |
| **Florence2**          | Modelo multimodal avançado | Potente, multitarefa, bom para contextos complexos                   | Muito pesado e complexo de usar em setups simples                   | Pesquisas, análise semântica multimodal       |
| **YOLOv5/YOLOv4**      | Detecção supervisionada  | Bons modelos anteriores, ainda populares                             | YOLOv8 é mais preciso e mais rápido                                 | Ambientes legados ou comparações históricas   |

---

## 🧬 Comparativo entre variantes do YOLOv8

| Variante      | Parâmetros | Tamanho do Modelo | Velocidade (FPS) | Precisão (mAP@0.5) | Ideal para...                  |
|---------------|------------|-------------------|------------------|--------------------|-------------------------------|
| **YOLOv8n**   | ~3.2M      | 🪶 Muito leve       | 🚀 Muito rápida   | Média              | Edge, mobile, inferência leve |
| **YOLOv8s**   | ~11.2M     | 🧩 Leve             | 🚀 Rápida         | Alta               | Colab, desktop, tempo real     |
| **YOLOv8m**   | ~25.9M     | ⚖️ Média            | ⚡ Boa            | Muito Alta         | Servidores, GPUs maiores       |
| **YOLOv8l/x** | 50M+       | 🏋️‍♂️ Pesado         | Mais lento        | Excelente           | Infraestrutura robusta         |

---

## 🎯 Justificativa para o uso do **YOLOv8s**

Escolhemos a versão **YOLOv8s** por oferecer o **melhor equilíbrio entre desempenho e leveza**:

- ✅ Rápido o suficiente para aplicações em tempo real, como vídeo e stream.
- ✅ Mais preciso que o `YOLOv8n` e significativamente menor que `YOLOv8m`, o que facilita o uso no **Google Colab com GPU T4**.
- ✅ Ideal para datasets customizados e com estratégias de data augmentation, como o nosso (com foco em detecção de objetos cortantes).
- ✅ Suporte excelente a múltiplas instâncias e classes com bom tempo de inferência.

---

## ✅ Resumo da escolha

O **YOLOv8s** foi selecionado como a arquitetura ideal por ser uma solução **leve, rápida e precisa**, perfeita para cenários com **recursos computacionais limitados** e necessidade de **detecção confiável em tempo real**.

Modelos mais pesados como `YOLOv8m` ou ferramentas como `Florence2` são poderosos, mas desnecessários e ineficientes para o objetivo atual. O YOLOv8s entrega **ótimo desempenho com excelente custo-benefício computacional**.


# Montagem do dataset


## 🧩 Montagem e Unificação do Dataset de Objetos Cortantes

Para garantir um bom desempenho do modelo YOLOv8 mesmo em ambientes com recursos computacionais limitados, foi necessário dedicar um pouco mais de esforço à montagem de um **dataset pequeno, mas com boa qualidade e diversidade**. A ideia foi garantir representatividade visual com variações de **ângulo, iluminação, contexto e tipos de objetos cortantes**, sem sobrecarregar o processo de treinamento.

---

## 🔍 Por que usar o Roboflow?

A ferramenta [Roboflow](https://roboflow.com/) foi escolhida por vários motivos:

- ✅ Disponibilidade de **diversos datasets rotulados** de forma pública e gratuita
- ✅ Interface prática para **pré-visualização**, **filtragem por classe** e **download no formato YOLOv8**
- ✅ Suporte para **datasets por versão**, mantendo controle das origens
- ✅ Facilidade de exportação padronizada (`images/` e `labels/` por split)

---

### 📥 1. Coleta dos Datasets

Realizou-se uma **pesquisa ativa por conjuntos de dados públicos no Roboflow**, com foco nas seguintes classes:

```python
["knife", "scissor", "scalpel", "axe", "saw", "chainsaw", "chisel", "sickle"]
```

Cada classe foi buscada individualmente, selecionando projetos com imagens reais, bounding boxes precisos e variações visuais significativas. Os datasets foram então baixados e organizados em pastas separadas por classe.

---

### 🧩 2. Unificação dos Datasets

Como os datasets coletados possuíam diferentes **estruturas e índices de classes**, tornou-se necessário unificá-los.

Para isso, foi utilizado o script `unificar-dataset-e-atualizar-indice.py`, presente na pasta `scripts-utilitarios`, que:

- 🗃️ Agrupa todos os arquivos em uma estrutura comum (`test/images`, `train/images`, `valid/images`)
- 🔄 Atualiza os arquivos de rótulo `.txt` para refletirem os **índices padronizados**

Essa etapa garante que os dados estejam **prontos para o treinamento em YOLOv8**, com consistência entre `data.yaml`, as imagens e os rótulos.

---

### ✂️ 3. Subsampling durante a unificação

Durante o processo de unificação, foi aplicado um **subsampling** para limitar a quantidade de exemplos das categorias knife e axe:

- 🔁 **Máximo de 1.000 imagens para as categorias alvo**
- 🎯 Isso evita sobrecarregar a memória e acelera o processo de treinamento
- ⚖️ Ajuda a **balancear o dataset**, evitando que uma classe como `"knife"` e `"axe"` domine o aprendizado

---

#### Resultado do Subsampling

##### 📂 Train
| Categoria    | Imagens únicas |
|--------------|----------------|
| knife        | 1000           |
| scissor      | 560            |
| scalpel      | 588            |
| axe          | 1000           |
| saw          | 633            |
| chainsaw     | 757            |
| chisel       | 309            |
| sickle       | 345            |

##### 📂 Valid
| Categoria    | Imagens únicas |
|--------------|----------------|
| knife        | 468            |
| scissor      | 160            |
| scalpel      | 166            |
| axe          | 234            |
| saw          | 0              |
| chainsaw     | 217            |
| chisel       | 29             |
| sickle       | 0              |

##### 📂 Test
| Categoria    | Imagens únicas |
|--------------|----------------|
| knife        | 127            |
| scissor      | 80             |
| scalpel      | 86             |
| axe          | 250            |
| saw          | 0              |
| chainsaw     | 106            |
| chisel       | 15             |
| sickle       | 0              |



# 🧪 Superaugmentação de Dados com Albumentations para YOLOv8

Aplicamos técnicas avançadas de **data augmentation** para resolver o desequilíbrio entre as categorias do nosso dataset de detecção de objetos cortantes.

## 🎯 Objetivo

Aumentar o número de imagens para classes minoritárias como `sickle`, `chisel`, `scalpel` e `chainsaw`, garantindo que cada classe tivesse pelo menos **1000 imagens no conjunto de treino**, para manter o equilíbrio.

---

## 🧰 Técnicas utilizadas

Aumentações aplicadas usando [Albumentations](https://albumentations.ai/), uma das bibliotecas mais rápidas e flexíveis para visão computacional.

### 🔄 Transformaçōes aplicadas:

- `HorizontalFlip`: espelhamento horizontal aleatório
- `RandomBrightnessContrast`: variação aleatória de brilho e contraste
- `MotionBlur`: simula borrões de movimento
- `Affine`: rotação, escala e deslocamento espacial
- `CoarseDropout`: técnica inspirada no Cutout, simula obstruções parciais

### ⚙️ Parâmetros de segurança

- `clip=True`: impede que bboxes ultrapassem os limites da imagem
- `filter_invalid_bboxes=True`: remove bboxes com área inválida ou posição negativa
- `min_visibility=0.1`: ignora bboxes com menos de 10% visibilidade após augmentação

---

## 🧠 Problemas que resolvemos

| Problema                                   | Solução aplicada                                  |
|--------------------------------------------|--------------------------------------------------|
| Desequilíbrio entre classes                | Augmentações direcionadas para classes minoritárias |
| Bounding boxes inválidas ou corrompidas    | Clipping, filtro por visibilidade e checagem de validade |
| Dataset dominado por `knife` e `axe`       | Limitamos manualmente para 1000 imagens por classe |

---

## 🔢 Controle de quantidade por classe

Durante o processo de augmentação, adicionamos lógica para:
- Contabilizar imagens por classe
- Interromper a geração quando a classe atingir 1000 imagens
- Permitir múltiplas classes por imagem, desde que ao menos uma esteja abaixo do limite

---

## ✅ Resultado da Superaugmentação

Após a aplicação da superaugmentação e controle de limites, o dataset ficou assim:

### 📂 Train
| Categoria    | Imagens únicas |
|--------------|----------------|
| knife        | 1000           |
| scissor      | 1560           |
| scalpel      | 1588           |
| axe          | 1000           |
| saw          | 1633           |
| chainsaw     | 1757           |
| chisel       | 1218           |
| sickle       | 1345           |

### 📂 Valid
| Categoria    | Imagens únicas |
|--------------|----------------|
| knife        | 468            |
| scissor      | 160            |
| scalpel      | 166            |
| axe          | 234            |
| saw          | 0              |
| chainsaw     | 217            |
| chisel       | 29             |
| sickle       | 0              |

### 📂 Test
| Categoria    | Imagens únicas |
|--------------|----------------|
| knife        | 127            |
| scissor      | 80             |
| scalpel      | 86             |
| axe          | 250            |
| saw          | 0              |
| chainsaw     | 106            |
| chisel       | 15             |
| sickle       | 0              |

---

# ⚖️ Rebalanceamento dos Conjuntos `valid` e `test` do Dataset

Após a aplicação de superaugmentações para balancear o conjunto de treino (`train`), foi necessário **rebalancear os conjuntos de validação (`valid`) e teste (`test`)** para garantir que todas as categorias fossem representadas adequadamente em todas as fases do treinamento.

---

## 🎯 Objetivo

- Garantir que **todas as classes relevantes** estejam presentes em `valid` e `test`
- Aplicar uma divisão próxima a:
  - **8%** do total para `valid`
  - **4%** do total para `test`
- **Evitar desbalanceamento extremo**, especialmente em classes minoritárias como `saw`, `chisel` e `sickle`

---

## 🔍 Diagnóstico inicial

Antes do rebalanceamento, as seguintes classes estavam **zeradas ou sub-representadas**:

| Classe   | Train | Valid | Test |
|----------|-------|-------|------|
| saw      | 1633  | 0     | 0    |
| chisel   | 1218  | 29    | 15   |
| sickle   | 1345  | 0     | 0    |

---

## 🛠️ Estratégia aplicada

Utilizamos um script para:

1. **Identificar imagens** em `train` que continham as classes 4 (saw), 6 (chisel) e 7 (sickle)
2. **Selecionar aleatoriamente**:
   - 8% das imagens → mover para `valid`
   - 4% das imagens → mover para `test`
3. **Mover** os arquivos de imagem (`.jpg`) e seus respectivos rótulos (`.txt`)
4. Criar as pastas necessárias caso ainda não existissem

---

## ✅ Quantidades redistribuídas

| Classe   | Movidos para `valid` | Movidos para `test` |
|----------|----------------------|----------------------|
| saw      | 130                  | 65                   |
| chisel   | 97                   | 48                   |
| sickle   | 108                  | 54                   |

---

## 📂 Resultado

Após o rebalanceamento, as três classes agora também estão presentes nos conjuntos `valid` e `test`, tornando a validação mais justa e representativa.

- Esse rebalanceamento não altera o conteúdo de treino, apenas melhora a **avaliação final do modelo**.
- A abordagem é **segura e eficiente**, pois evita duplicações e mantém o alinhamento entre `images/` e `labels/`.
- Esse processo pode ser repetido sempre que o conjunto de treino for expandido ou alterado.

### 📸 Quantidade de imagens por categoria e por split

#### 📂 Train
| Categoria    | Imagens únicas |
|--------------|----------------|
| knife        | 1000           |
| scissor      | 1560           |
| scalpel      | 1588           |
| axe          | 1000           |
| saw          | 1438           |
| chainsaw     | 1757           |
| chisel       | 1073           |
| sickle       | 1183           |

#### 📂 Valid
| Categoria    | Imagens únicas |
|--------------|----------------|
| knife        | 468            |
| scissor      | 160            |
| scalpel      | 166            |
| axe          | 234            |
| saw          | 130            |
| chainsaw     | 217            |
| chisel       | 126            |
| sickle       | 108            |

#### 📂 Test
| Categoria    | Imagens únicas |
|--------------|----------------|
| knife        | 127            |
| scissor      | 80             |
| scalpel      | 86             |
| axe          | 250            |
| saw          | 65             |
| chainsaw     | 106            |
| chisel       | 63             |
| sickle       | 54             |


## 🏋️‍♂️ Treinamento

```python
model.train(
    data='/content/drive/dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=32,
    device=0,
    workers=2,
    name='sharped-vs-nonsharped-v1',
    pretrained=True,
    augment=True,
    mosaic=1.0,
    mixup=0.2,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    flipud=0.3, fliplr=0.5,
    degrees=10.0, translate=0.1, scale=0.5, shear=2.0,
    patience=20
)
```

### ✅ Parâmetros explicados:

| Parâmetro         | Significado                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `data`           | Caminho para o arquivo `data.yaml` com caminhos e nomes das classes         |
| `epochs`         | Número máximo de épocas de treinamento                                      |
| `imgsz`          | Tamanho da imagem (redimensionamento para 640x640)                          |
| `batch`          | Tamanho do batch (quantidade de imagens por iteração)                      |
| `device`         | GPU a ser usada (`0` para a primeira GPU disponível)                        |
| `workers`        | Número de workers para carregar dados em paralelo                           |
| `name`           | Nome do experimento (usado para salvar logs, pesos e métricas)              |
| `pretrained`     | Usa pesos pré-treinados do COCO (se `True`)                                 |
| `augment`        | Ativa augmentações básicas (como flips e mudanças de brilho/contraste)      |
| `mosaic`         | Ativa **Mosaic augmentation** (1.0 = 100% das imagens usam mosaic)          |
| `mixup`          | Ativa **MixUp augmentation** com 20% de intensidade                         |
| `hsv_h/s/v`      | Variações de tonalidade, saturação e brilho                                 |
| `flipud`         | Probabilidade de flip vertical                                              |
| `fliplr`         | Probabilidade de flip horizontal                                            |
| `degrees`        | Rotação aleatória de até ±10°                                               |
| `translate`      | Translação aleatória de até 10%                                             |
| `scale`          | Escala aleatória de até ±50%                                                |
| `shear`          | Inclinação aleatória de até 2°                                              |
| `patience`       | Número de épocas sem melhoria antes do Early Stopping                       |

---

## 🎯 Objetivo do modelo

Detectar objetos cortantes vs. não cortantes, usando imagens balanceadas, com múltiplas classes (como `knife`, `scissor`, `axe`, `sickle`, `chair`, `bathtub`, etc.) agrupadas em categorias `Sharped` e `Not-Sharped`.



