# 🐳 Docker Setup - NexoCommerce Multi-Agent ML System

Guia completo para configurar e executar o sistema usando Docker.

---

## 📋 Pré-requisitos

- **Docker**: versão 20.10+
- **Docker Compose**: versão 2.0+
- **Git**: para clonar o repositório

### Verificar Instalação

```bash
docker --version
docker-compose --version
```

---

## 🚀 Quick Start

### 1. Clone o Repositório

```bash
git clone https://github.com/zemarchezi/nexocommerce-ml-agents.git
cd nexocommerce-ml-agents
```

### 2. Configure Variáveis de Ambiente (Opcional)

```bash
# Copie o arquivo de exemplo
cp .env.example .env

# Edite com suas credenciais (opcional)
nano .env  # ou vim, code, etc.
```

**Importante**: Se você não configurar o `.env`, o sistema funcionará com dados sintéticos.

### 3. Inicie os Serviços

```bash
# Iniciar todos os serviços em background
docker-compose up -d

# Ver logs em tempo real
docker-compose logs -f
```

### 4. Acesse os Serviços

Aguarde ~30-60 segundos para os serviços iniciarem completamente.

| Serviço | URL | Descrição |
|---------|-----|-----------|
| **API REST** | http://localhost:8000 | FastAPI Backend |
| **API Docs** | http://localhost:8000/docs | Swagger UI |
| **MLflow** | http://localhost:5000 | Tracking Server |
| **Streamlit** | http://localhost:8501 | Interface Web |

---

## 🔧 Comandos Úteis

### Gerenciamento de Serviços

```bash
# Iniciar serviços
docker-compose up -d

# Parar serviços
docker-compose down

# Reiniciar serviços
docker-compose restart

# Ver status
docker-compose ps

# Ver logs
docker-compose logs -f

# Ver logs de um serviço específico
docker-compose logs -f api
docker-compose logs -f mlflow
docker-compose logs -f streamlit
```

### Rebuild (após mudanças no código)

```bash
# Rebuild e reiniciar
docker-compose up -d --build

# Rebuild sem cache
docker-compose build --no-cache
docker-compose up -d
```

### Executar Comandos nos Containers

```bash
# Treinar modelo dentro do container
docker-compose exec api python src/pipeline/training_pipeline.py \
    --source synthetic \
    --n_samples 5000 \
    --model_type random_forest \
    --hyperparameter_tuning

# Acessar shell do container
docker-compose exec api bash

# Executar testes
docker-compose exec api pytest tests/ -v

# Ver estrutura de arquivos
docker-compose exec api ls -la
```

### Limpeza

```bash
# Parar e remover containers
docker-compose down

# Remover containers, networks e volumes
docker-compose down -v

# Remover imagens também
docker-compose down --rmi all

# Limpeza completa do Docker
docker system prune -a --volumes
```

---

## 📊 Verificação de Saúde

### Health Checks Automáticos

Os serviços possuem health checks configurados:

```bash
# Ver status de saúde
docker-compose ps

# Exemplo de saída:
# NAME                    STATUS
# nexocommerce-api        Up (healthy)
# nexocommerce-mlflow     Up (healthy)
# nexocommerce-streamlit  Up (healthy)
```

### Testes Manuais

```bash
# Testar API
curl http://localhost:8000/health

# Testar MLflow
curl http://localhost:5000/health

# Testar Streamlit
curl http://localhost:8501/_stcore/health
```

---

## 🔐 Configuração de Credenciais Kaggle

### Opção 1: Arquivo .env

```bash
# Edite o arquivo .env
KAGGLE_USERNAME=seu_usuario
KAGGLE_KEY=sua_chave_api
```

### Opção 2: Variáveis de Ambiente

```bash
export KAGGLE_USERNAME="seu_usuario"
export KAGGLE_KEY="sua_chave_api"
docker-compose up -d
```

### Opção 3: Arquivo kaggle.json

```bash
# Crie o diretório
mkdir -p ~/.kaggle

# Copie suas credenciais
cp kaggle.json ~/.kaggle/

# Ajuste permissões
chmod 600 ~/.kaggle/kaggle.json
```

---

## 📁 Volumes e Persistência

### Volumes Montados

```yaml
api:
  volumes:
    - ./src:/app/src          # Código fonte
    - ./models:/app/models    # Modelos treinados
    - ./data:/app/data        # Dados
    - ./mlruns:/app/mlruns    # MLflow artifacts

mlflow:
  volumes:
    - ./mlruns:/mlruns        # Tracking data
    - ./artifacts:/artifacts  # Artifacts

streamlit:
  volumes:
    - ./app:/app/app          # UI code
    - ./src:/app/src          # Shared code
```

### Backup de Dados

```bash
# Backup de modelos
tar -czf models_backup.tar.gz models/

# Backup de MLflow
tar -czf mlruns_backup.tar.gz mlruns/

# Backup completo
tar -czf nexocommerce_backup.tar.gz models/ mlruns/ data/
```

---

## 🐛 Troubleshooting

### Problema: Porta já em uso

```bash
# Verificar portas em uso
lsof -i :8000
lsof -i :5000
lsof -i :8501

# Matar processo
kill -9 <PID>

# Ou mudar portas no docker-compose.yml
ports:
  - "8001:8000"  # Usar porta 8001 no host
```

### Problema: Container não inicia

```bash
# Ver logs detalhados
docker-compose logs api

# Verificar configuração
docker-compose config

# Rebuild completo
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d
```

### Problema: Modelo não encontrado

```bash
# Treinar modelo primeiro
docker-compose exec api python src/pipeline/training_pipeline.py \
    --source synthetic \
    --n_samples 5000 \
    --model_type random_forest

# Verificar se modelo foi criado
docker-compose exec api ls -la models/
```

### Problema: Erro de permissão

```bash
# Ajustar permissões dos diretórios
sudo chown -R $USER:$USER models/ mlruns/ data/
chmod -R 755 models/ mlruns/ data/
```

### Problema: Out of Memory

```bash
# Aumentar memória do Docker
# Docker Desktop > Settings > Resources > Memory

# Ou limitar recursos no docker-compose.yml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 2G
```

---

## 🔄 Workflow Completo

### 1. Setup Inicial

```bash
# Clone e configure
git clone https://github.com/zemarchezi/nexocommerce-ml-agents.git
cd nexocommerce-ml-agents
cp .env.example .env

# Inicie serviços
docker-compose up -d

# Aguarde health checks
docker-compose ps
```

### 2. Treinamento

```bash
# Treinar modelo
docker-compose exec api python src/pipeline/training_pipeline.py \
    --source synthetic \
    --n_samples 10000 \
    --model_type gradient_boosting \
    --hyperparameter_tuning

# Verificar no MLflow
# Acesse: http://localhost:5000
```

### 3. Teste da API

```bash
# Predição individual
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "product_id": "PROD001",
    "category": "Eletrônicos",
    "price": 299.90,
    "stock_quantity": 50,
    "sales_last_30d": 120,
    "views_last_30d": 1500,
    "rating": 4.5,
    "reviews_count": 45,
    "return_rate": 0.05
  }'
```

### 4. Interface Web

```bash
# Acesse o Streamlit
open http://localhost:8501

# Ou
xdg-open http://localhost:8501  # Linux
start http://localhost:8501     # Windows
```

---

## 📈 Monitoramento

### Logs em Tempo Real

```bash
# Todos os serviços
docker-compose logs -f

# Apenas API
docker-compose logs -f api

# Últimas 100 linhas
docker-compose logs --tail=100 api
```

### Métricas de Recursos

```bash
# Ver uso de recursos
docker stats

# Específico
docker stats nexocommerce-api
```

### MLflow Tracking

```bash
# Acessar MLflow UI
open http://localhost:5000

# Ver experimentos via CLI
docker-compose exec api mlflow experiments list
```

---

## 🚀 Deploy em Produção

### Recomendações

1. **Use variáveis de ambiente** para configurações sensíveis
2. **Configure volumes externos** para persistência
3. **Use reverse proxy** (Nginx, Traefik)
4. **Configure SSL/TLS** para HTTPS
5. **Implemente autenticação** na API
6. **Configure backup automático** de modelos e dados
7. **Use orquestração** (Kubernetes, Docker Swarm)

### Exemplo com Nginx

```yaml
# docker-compose.prod.yml
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api
      - streamlit
```

---

## 📚 Recursos Adicionais

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [MLflow Deployment](https://mlflow.org/docs/latest/deployment.html)

---

## 🆘 Suporte

Se encontrar problemas:

1. Verifique os logs: `docker-compose logs -f`
2. Consulte a seção de Troubleshooting
3. Abra uma issue no GitHub
4. Entre em contato: jpmarchezi@gmail.com

---

**Made byJosé Marchezi**