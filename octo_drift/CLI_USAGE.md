# Guia de Uso - Octo-Drift CLI

## Instalação

Substitua o arquivo:
```bash
cp run_experiment.py octo_drift/experiments/
```

## Uso Básico

### Defaults (mesmos parâmetros atuais)
```bash
octo-drift --train data/train.arff --test data/test.arff --output results/
```

### Ajustando phi
```bash
octo-drift --train data/train.arff --test data/test.arff --output results/ --phi 0.15
```

### Vários parâmetros
```bash
octo-drift \
  --train data/train.arff \
  --test data/test.arff \
  --output results_tuned/ \
  --phi 0.15 \
  --buffer-threshold 30 \
  --k-short 3 \
  --min-weight-online 10
```

## Parâmetros Disponíveis

### Clustering
- `--k 4` - Clusters por classe
- `--k-short 4` - Clusters para buffer desconhecido
- `--fuzzification 2.0` - Parâmetro de fuzificação (m)
- `--alpha 2.0` - Expoente de pertinência
- `--theta 1.0` - Expoente de tipicidade

### Detecção de Novidades
- `--phi 0.2` - Threshold de sobreposição (0.1-0.3)
- `--buffer-threshold 40` - Tamanho do buffer para trigger (T)
- `--min-weight-offline 0` - Tamanho mínimo de cluster no treino
- `--min-weight-online 15` - Tamanho mínimo de cluster no stream

### Aprendizado Incremental
- `--latency 10000` - Atraso de rótulos
- `--chunk-size 2000` - Tamanho do batch para atualização
- `--time-threshold 200` - Threshold de remoção (ts)
- `--percent-labeled 1.0` - Fração de rótulos usados (0-1)

### Avaliação
- `--evaluation-interval 1000` - Frequência de cálculo de métricas

### Debug
- `--debug` - Habilita logs detalhados

## Exemplos Práticos

### 1. Aumentar Sensibilidade de Detecção
```bash
octo-drift \
  --train data/train.arff \
  --test data/test.arff \
  --output results_sensitive/ \
  --phi 0.15 \
  --buffer-threshold 30 \
  --k-short 3 \
  --min-weight-online 10 \
  --debug
```

### 2. Latência Curta (2000 exemplos)
```bash
octo-drift \
  --train data/train.arff \
  --test data/test.arff \
  --output results_short_latency/ \
  --latency 2000 \
  --chunk-size 500
```

### 3. Rotulação Parcial (20%)
```bash
octo-drift \
  --train data/train.arff \
  --test data/test.arff \
  --output results_partial/ \
  --percent-labeled 0.2 \
  --latency 5000
```

### 4. Usar Arquivo de Config + Override
```bash
# config.json
{
  "k": 5,
  "phi": 0.2,
  "buffer_threshold": 40,
  "latency": 10000
}

# Executar (mantém k=5 mas muda phi)
octo-drift \
  --train data/train.arff \
  --test data/test.arff \
  --output results/ \
  --config config.json \
  --phi 0.15
```

### 5. Grid Search (ignora CLI params exceto base_config)
```bash
octo-drift \
  --train data/train.arff \
  --test data/test.arff \
  --output results_grid/ \
  --grid-search
```

## Ver Ajuda Completa
```bash
octo-drift --help
```

## Exemplo de Config JSON

```json
{
  "k": 4,
  "k_short": 4,
  "fuzzification": 2.0,
  "alpha": 2.0,
  "theta": 1.0,
  "phi": 0.2,
  "buffer_threshold": 40,
  "min_weight_offline": 0,
  "min_weight_online": 15,
  "latency": 10000,
  "chunk_size": 2000,
  "time_threshold": 200,
  "percent_labeled": 1.0,
  "evaluation_interval": 1000
}
```

Salvo como `config.json`, use:
```bash
octo-drift --train data/train.arff --test data/test.arff --config config.json --output results/
```

## Análise de Sensibilidade Recomendada

```bash
# Teste 1: Baseline
octo-drift --train data/train.arff --test data/test.arff --output test1_baseline/

# Teste 2: Phi mais restritivo
octo-drift --train data/train.arff --test data/test.arff --output test2_phi015/ --phi 0.15

# Teste 3: Buffer menor
octo-drift --train data/train.arff --test data/test.arff --output test3_buffer30/ --buffer-threshold 30

# Teste 4: Combinado
octo-drift --train data/train.arff --test data/test.arff --output test4_combined/ \
  --phi 0.15 --buffer-threshold 30 --k-short 3 --min-weight-online 10

# Comparar results/metrics.csv de cada teste
```

## Output Gerado

Cada execução cria:
- `results.csv` - Classificações de cada exemplo
- `metrics.csv` - Accuracy, precision, recall, F1, unknown rate
- `novelties.csv` - Flags de detecção de novidade
- `config.json` - Configuração usada (para reprodução)
- `metrics_plot.png` - Gráficos de accuracy e unknown rate
- `all_metrics.png` - Todas as métricas juntas
