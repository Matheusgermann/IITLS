<p align="center"><img src="assets/logo.png" alt="IITLS" height="64"></p>

# IITLS — Intelligent Integrated Traffic Light System

Semáforos que se ajustam ao trânsito real e conversam entre si. Uma câmera em cada cruzamento conta veículos e pedestres por faixa e mede a fila dentro de uma distância fixa da faixa de retenção, direto na borda (NVIDIA Jetson Orin Nano). Para a nuvem vão só números, nunca vídeo. Com esses dados, os semáforos de um corredor passam a abrir em sequência (onda verde), sempre dentro de regras fixas de segurança: verde mínimo, travessia de pedestre e tempo máximo de espera para qualquer aproximação.

**Finalista do Red Bull Basement 2026**, entre os 5 projetos selecionados no Brasil.

## Como funciona

1. **Detecção e contagem:** detector de objetos baseado em transformer (RF-DETR) com rastreamento, separando pedestre, bicicleta, moto, carro, utilitário, ônibus e caminhão.
2. **Fila por faixa:** cada faixa tem uma zona calibrada em metros reais; quem está parado dentro dela é fila, quem passa é chegada.
3. **Coordenação:** a nuvem cruza os dados de todos os cruzamentos e calcula os tempos de cada semáforo; a borda aplica o plano sem nunca violar as regras de segurança e segue sozinha se a conexão cair.

## Roteiro

- [ ] Detector e contagem de fila validados em vídeo de cruzamentos brasileiros (dia, noite e chuva)
- [ ] Modelo genérico publicado no Hugging Face e contador aberto neste repositório
- [ ] Simulações para definir posição e altura das câmeras e o desempenho na Jetson Orin Nano
- [ ] Modelo de controle com regras de segurança validadas em simulação
- [ ] Testes em cruzamentos reais em modo observação, sem atuar no semáforo
- [ ] Piloto atuando em campo

O código do sistema completo é privado.

## Contato

Site: [iitls.vercel.app](https://iitls.vercel.app) · Prefeituras e concessionárias: iitls.startup@gmail.com

Fundadores: Matheus Germann, João Vitor Soska e Pedro Otávio Nunes.

---

**EN —** IITLS turns ordinary intersections into a coordinated network: on-device vision counts traffic and measures queues per lane within a calibrated distance, a cloud coordinator computes green-wave timings, and every intersection keeps hard safety rules and runs on its own if the link drops. Red Bull Basement 2026 finalist (top 5 in Brazil). Currently building and validating the detection and queue-counting model.
