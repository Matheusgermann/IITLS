# 🐳 Docker Environment

Este projeto possui um ambiente Docker configurado para garantir consistência e reprodutibilidade do ambiente de execução.

Todos os desenvolvedores podem utilizar exatamente o mesmo ambiente, independentemente da configuração local.

> ⚠️ **Importante:** o ambiente Docker padroniza bibliotecas, Python, Git e suporte a GPU, mas **não inclui o arquivo de pesos treinado (`best.pt`)**. Portanto, **o projeto não pode ser executado integralmente por terceiros** sem acesso ao modelo privado.

---

## 📦 O que o ambiente entrega


- Python 3.11
- PyTorch 2.7.1 com CUDA 11.8
- torchvision 0.22.1
- torchaudio 2.7.1
- Ultralytics YOLO
- OpenCV
- Git instalado dentro do container
- compatibilidade com **GPU NVIDIA**

---

## ⚠️ Pré-requisitos

Antes de utilizar o Docker é necessário:

1. **Instalar Docker Desktop** <https://www.docker.com/products/docker-desktop/>
2. **Ativar WSL2 no Docker Desktop** `Docker Desktop` → `Settings` → `General` → `Enable WSL2`
3. **Ter driver NVIDIA atualizado** <https://www.nvidia.com/Download/index.aspx>

---

## 🖥️ Testar acesso à GPU

Execute no terminal para verificar se o Docker reconhece sua placa de vídeo:

```bash
docker run --rm --gpus all nvidia/cuda:12.9.0-base-ubuntu22.04 nvidia-smi
```

Se o comando retornar as informações da GPU, o Docker está configurado corretamente.

---

## 🏗️ Build e execução

**Dentro da pasta do projeto, execute o build:**
```bash
docker compose -f docker_env/docker-compose.yml build
```

**Executar o projeto diretamente:**
```bash
docker compose -f docker_env/docker-compose.yml run --rm traffic-ai python app.py
```

**Abrir container (shell interativo):**
```bash
docker compose -f docker_env/docker-compose.yml run --rm traffic-ai bash
```

*Observação: como o `docker-compose.yml` está dentro da pasta `docker_env/`, o uso da flag `-f docker_env/docker-compose.yml` deixa o comando explícito e evita ambiguidade.*

---

## 🧪 Testes rápidos dentro do container

Ao acessar o shell do container, você pode rodar os seguintes comandos para validar o ambiente.

**Versões básicas:**
```bash
python --version
git --version
```

**Teste de GPU:**
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'sem gpu')"
```

**Teste YOLO:**
```bash
python -c "import ultralytics; print(ultralytics.__version__)"
```

---

## ⚙️ Visualização no Docker

O projeto foi estruturado para permitir o mesmo código em dois cenários:
- **VS Code / execução local interna:** `SHOW_WINDOW = True`
- **Docker:** `SHOW_WINDOW = False`

No Docker, o ideal é manter `SHOW_WINDOW = False`, já que o foco do container é processar o vídeo e salvar o resultado final em arquivo, sem depender de interface gráfica.

---

## 📌 Observações

- O mesmo ambiente será utilizado por todos os membros do projeto, garantindo consistência entre máquinas diferentes.
- O Git utilizado dentro do ambiente Docker é o Git do próprio container.
- O projeto é executado no container utilizando o mesmo código-fonte montado no diretório de trabalho.
- A lógica atual do sistema utiliza duas linhas independentes de contagem, uma para cada fluxo da pista, com configuração feita diretamente no arquivo `config.py`.
- Para executar o processamento completo, ainda é necessário ter acesso ao arquivo privado `pesos/best.pt`.