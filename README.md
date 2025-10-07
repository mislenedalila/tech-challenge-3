# Tech Challenge Fase 3 - Fine-tuning com Amazon Titles

## Descrição do Projeto

Este projeto foi desenvolvido como parte do **Tech Challenge da Fase 3** do curso de IA para Devs da FIAP. O objetivo é realizar o fine-tuning de um foundation model utilizando o dataset AmazonTitles-1.3MM, criando um sistema capaz de gerar descrições detalhadas de produtos a partir de seus títulos.

## Objetivos

- Executar fine-tuning do modelo **LLaMA-3 8B** quantizado em 4 bits
- Utilizar o dataset **AmazonTitles-1.3MM** (arquivo `trn.json`)
- Implementar técnica **LoRA** (Low-Rank Adaptation) para treinamento eficiente
- Criar sistema de inferência para responder perguntas sobre produtos
- Comparar performance antes e depois do fine-tuning

## Tecnologias Utilizadas

- **Python 3.10+**
- **PyTorch**
- **Transformers (Hugging Face)**
- **Unsloth** - Otimização para fine-tuning
- **LoRA/PEFT** - Adaptação eficiente de parâmetros
- **TRL** - Treinamento supervisionado
- **Google Colab** - Ambiente de desenvolvimento

## Estrutura do Projeto

```
tech-challenge-fine-tuning/
│
├── notebooks/
│   └── tech_challenge_fine_tunning.ipynb    # Notebook principal
│
├── data/
│   ├── trn.json                             # Dataset original
│   └── data.json                            # Dataset formatado
│   └── lora_data/                           # Modelo fine-tunado
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── tokenizer files...
│
│
├── README.md                                # Este                      
```

## Como Executar

### 1. Configuração do Ambiente

**No Google Colab:**

```python
# Montar Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Instalar dependências
!pip install "unsloth[colab-new]" -U
!pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes
!pip install transformers datasets
```

### 2. Download do Dataset

1. Baixe o dataset AmazonTitles-1.3MM do [Google Drive](https://drive.google.com/file/d/12zH4mL2RX8iSvH0VCNnd3QxO4DzuHWnK/view)
2. Extraia o arquivo `trn.json`
3. Coloque no seu Google Drive na pasta: `/content/drive/MyDrive/fine-tunning-fiap/`

### 3. Execução do Notebook

1. Abra o arquivo `tech_challenge_fine_t
unning.ipynb` no Google Colab
2. Execute todas as células sequencialmente
3. Aguarde o processo de fine-tuning (aproximadamente 30-60 minutos)

## Processo de Fine-tuning

### Pré-processamento dos Dados

```python
# Formato do dataset original
{
  "title": "Nome do produto",
  "content": "Descrição detalhada do produto"
}

# Formato após processamento (estilo Alpaca)
{
  "instruction": "Describe book as accurately as possible.",
  "input": "Question: Could you give me a description of the book 'Nome do Produto'?",
  "output": "Descrição detalhada do produto"
}
```

### Configuração do Modelo

- **Modelo Base:** `unsloth/llama-3-8b-bnb-4bit`
- **Técnica:** LoRA (Low-Rank Adaptation)
- **Parâmetros LoRA:**
  - `r = 16`
  - `lora_alpha = 32`
  - `lora_dropout = 0.05`
- **Sequência máxima:** 2048 tokens

### Parâmetros de Treinamento

```python
TrainingArguments(
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    warmup_steps = 5,
    max_steps = 100,
    learning_rate = 2e-4,
    fp16 = True,  # ou bf16 se suportado
    logging_steps = 1,
    optim = "adamw_8bit",
    weight_decay = 0.01,
    lr_scheduler_type = "linear"
)
```

## Testes e Validação

### Exemplo de Uso

```python
# Template de prompt
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Describe book as accurately as possible.

### Input:
Question: Could you give me a description of the book 'Clean Architecture'?

### Response:
"""

# Inferência
outputs = pipe(
    alpaca_prompt,
    max_new_tokens=256,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)
```

### Comparação Antes vs Depois

O notebook inclui testes comparativos mostrando:
- **Modelo Base:** Respostas genéricas e pouco específicas
- **Modelo Fine-tunado:** Descrições detalhadas e contextualizadas

## Resultados

### Melhorias Observadas

1. **Especificidade:** Respostas mais específicas sobre produtos
2. **Coerência:** Maior coerência temática nas descrições
3. **Detalhamento:** Descrições mais ricas e informativas
4. **Contextualização:** Melhor compreensão do contexto do produto

### Métricas de Performance

- **Tempo de treinamento:** ~45 minutos (Google Colab T4)
- **Uso de memória:** ~8GB VRAM
- **Parâmetros treináveis:** ~4.2M (LoRA)
- **Dataset utilizado:** 1M exemplos formatados

## Demonstração

O projeto inclui uma demonstração em vídeo mostrando:

1. **Carregamento do dataset**
2. **Processo de fine-tuning**
3. **Comparação antes/depois**
4. **Testes interativos**
5. **Resultados finais**

**Link do vídeo:** [https://youtu.be/EIicNuvhIPI?si=042in8hE05zbd_hX]

## Requisitos Técnicos

### Hardware Recomendado

- **GPU:** NVIDIA T4 ou superior (disponível no Google Colab)
- **RAM:** 12GB+ (sistema)
- **VRAM:** 8GB+ (GPU)
- **Armazenamento:** 10GB+ livre

### Software

```txt
torch>=2.0.0
transformers>=4.36.0
datasets>=2.14.0
unsloth[colab-new]
peft>=0.7.0
trl<0.9.0
accelerate>=0.24.0
bitsandbytes>=0.41.0
xformers
```

## Configurações Avançadas

### Otimizações de Memória

```python
# Para GPUs com menos memória
per_device_train_batch_size = 1
gradient_accumulation_steps = 8
max_seq_length = 1024
```

### Ajuste de Hiperparâmetros

```python
# Para convergência mais rápida
learning_rate = 5e-4
max_steps = 50

# Para melhor qualidade
learning_rate = 1e-4
max_steps = 200
```

## Troubleshooting

### Problemas Comuns

1. **Erro de memória CUDA:**
   - Reduza `per_device_train_batch_size`
   - Aumente `gradient_accumulation_steps`

2. **Dataset não encontrado:**
   - Verifique o caminho do arquivo `trn.json`
   - Confirme se o Google Drive está montado

3. **Modelo não carrega:**
   - Verifique conexão com internet
   - Reinicie o runtime do Colab

## Equipe

- **Bruno Lima da Cruz**
- **Matheus Braz Giudice dos Santos**
- **Mislene Dalila da Silva**

## Licença

Este projeto foi desenvolvido para fins educacionais como parte do Tech Challenge da FIAP.

## Links Úteis

- **Dataset:** [AmazonTitles-1.3MM](https://drive.google.com/file/d/12zH4mL2RX8iSvH0VCNnd3QxO4DzuHWnK/view)
- **Unsloth:** [GitHub](https://github.com/unslothai/unsloth)
- **Transformers:** [Documentação](https://huggingface.co/docs/transformers)
- **LoRA:** [Paper Original](https://arxiv.org/abs/2106.09685)


---

**Tech Challenge Fase 3 - IA para Devs - FIAP**
