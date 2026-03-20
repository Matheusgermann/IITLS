# 🚦 IITLS - Intelligent Integrated Traffic Light System

O **IITLS** (Sistema Inteligente e Integrado de Semáforos) é uma solução de *Edge AI* desenvolvida para modernizar a mobilidade urbana. 

Em vez de operar com tempos fixos que geram congestionamentos, o sistema utiliza Visão Computacional na borda para detectar, classificar e contar veículos em tempo real. O objetivo final do IITLS é converter o fluxo de vídeo das ruas em dados estruturados leves, permitindo a sincronização dinâmica de redes de semáforos (ondas verdes) para reduzir o tempo de viagem e a emissão de carbono nas cidades.

> 🏆 **Reconhecimento:** Este projeto foi selecionado como **Finalista do Red Bull Basement 2026**, figurando entre os **5 melhores projetos inovadores do Brasil**.

---

## 🎯 O Desafio e a Solução

A maioria das cidades ainda opera com semáforos de tempos fixos, gerando congestionamentos desnecessários e alta emissão de poluentes. O **IITLS** resolve esse problema transformando cruzamentos comuns em ecossistemas inteligentes e dinâmicos.

Através de **Visão Computacional** avançada rodando diretamente no local (*Edge Computing*), o sistema analisa o fluxo de veículos em tempo real e gera dados estruturados. Esses dados permitem a sincronização inteligente de semáforos, criando "ondas verdes" que se adaptam à realidade da rua instantaneamente.

> 🔒 **Nota de Propriedade Intelectual:** Este repositório atua exclusivamente como um portfólio e demonstração (Showcase) da tecnologia. O código-fonte proprietário, os pesos das redes neurais, a arquitetura de telemetria e os algoritmos de integração em nuvem são mantidos em repositórios privados.

---

## 🎬 Demonstração da Tecnologia

Abaixo está uma demonstração do nosso algoritmo proprietário processando um fluxo real de tráfego, realizando a detecção, classificação e contagem multifaixas em tempo real.

<img src="./GIFS/IITLS_GIF.gif" width="800">

---

## 🚀 Diferenciais da Arquitetura IITLS

O sistema foi desenhado para ser viável, escalável e de baixo custo para governos e concessionárias:

- **Processamento na Borda (Edge AI):** A análise de vídeo ocorre dentro do próprio poste. Não enviamos vídeos pesados para a nuvem, o que zera a necessidade de planos de dados móveis caros e garante respostas em milissegundos.
- **Rede Privada P2MP:** Utilização de arquitetura de telemetria via rádio de alta velocidade, criando uma rede fechada, segura e sem custos mensais de operadora.
- **Auditoria Sob Demanda:** Sistema híbrido inteligente. O nó de processamento local mantém um *loop* de gravação de segurança, permitindo uploads de trechos específicos de vídeo apenas quando há necessidade de auditoria ou em casos de anomalias (acidentes).
- **Precisão Multivias:** Rastreamento avançado de objetos (evitando contagens duplicadas) e divisão de linhas virtuais, garantindo dados precisos por faixa de rolamento, sentido da via e tipo de veículo (carros, caminhões, ônibus, motos e pedestres).

---

## 📊 Impacto Esperado (Smart Cities)

A implementação da arquitetura IITLS gera resultados diretos na mobilidade urbana:
1. **Redução do tempo de viagem** e de veículos parados sem necessidade.
2. **Queda drástica na emissão de CO2**, diminuindo o tempo de veículos acelerando e freando por conta dos semáforos mal otimizados.
3. **Geração de dados valiosos** (Dashboard em tempo real) para o planejamento de malhas viárias pelas prefeituras e concessionárias.

---

## 👥 Fundadores / Desenvolvimento

Projeto desenvolvido por:
- **Matheus Germann**
- **João Vitor Soska**
- **Pedro Otávio Nunes**