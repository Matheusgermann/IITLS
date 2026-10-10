# IITLS nuvem: coordenação de semáforos e onda verde

Recebe dados agregados por faixa das câmeras (Jetson na borda) e devolve a cada cruzamento um plano
(ciclo, defasagem, verdes). O objetivo é minimizar **atraso de pessoas + paradas** na rede inteira,
não só em um cruzamento.

```
câmeras + Jetson ──MQTT/WireGuard 82 B/5 s──▶ nuvem: estimador ▶ modos ▶ ciclo/verdes ▶ defasagens ──plano──▶ executor local ▶ controlador
      │  (YOLO + contagem por faixa)                                                          ▲  (coordenado-atuado)
      └── emergência / ônibus: reação local imediata ─────────────────────────────────────────┘
```

Contrato de mensagens: [`docs/protocolo-edge-nuvem.md`](../docs/protocolo-edge-nuvem.md).

## Como decide

| etapa | arquivo | o que faz |
|---|---|---|
| estimativa | `estimator.py` | fluxo em PCU/h por faixa (média móvel), fila residual (a que não esvazia no verde), peso de pessoas por veículo, histórico por janela de 15 min para cobrir falha de câmera |
| modo por cruzamento | `controller.py` | NORMAL, BAIXA DEMANDA, SATURADO, PREEMPÇÃO, FALHA, PISCANTE |
| ciclo e verdes | `timing.py` | Webster como ponto de partida, divisão proporcional à razão crítica com mínimos de veículo e pedestre, atraso HCM (vale com supersaturação) |
| grupos | `coordination.py` | cruzamentos ligados por vias curtas o bastante para o pelotão não se dispersar (≤ 90 s de viagem) formam um grupo com **ciclo comum** |
| defasagens | `coordination.py` | modelo de pelotão (descarga da fila, viagem, dispersão de Robertson) e fila determinística segundo a segundo geram uma tabela de custo por ligação para cada defasagem possível. Em corredor/árvore a programação dinâmica dá o **ótimo global exato**; em grade, refina por descida coordenada. Os dois sentidos entram juntos, ponderados por volume e ocupação |
| escolha do ciclo | `controller.py` | avalia ciclos em torno do de Webster com o modelo da rede inteira e escolhe o de menor custo, com histerese de 5% para não ficar trocando |
| execução | `edge_runtime.py` | executor local coordenado-atuado: fase coordenada presa à defasagem, demais fases pulam/encerram cedo sem demanda (câmera em tempo real), transição suave de plano, preempção, prioridade a ônibus, plano de segurança se a nuvem sumir |

### Casos cobertos

- **Fluxo assimétrico** (pico de ida/volta): o custo pondera os dois sentidos pelo volume real.
- **Supersaturação**: fila residual vira demanda, ciclo sobe, e o controle de transbordo segura o verde que
  alimenta uma via cuja fila não esvazia (evita travar o cruzamento de cima).
- **Baixa demanda / madrugada**: ciclo mínimo; amarelo piscante onde o cruzamento permitir (`allow_flash`).
- **Pedestres**: verde mínimo de travessia por fase (`ped_min_green`), respeitado mesmo se a fase for encurtada.
- **Transporte coletivo**: otimiza atraso de **pessoas** (ônibus ≈ 30 pessoas) e estende o verde em até 10 s
  quando a câmera vê um ônibus chegando.
- **Veículo de emergência**: a borda preempta na hora; a nuvem abre o corredor verde nos próximos 3 cruzamentos.
- **Incidente**: faixa bloqueada perde capacidade até o evento de liberação.
- **Falha de câmera / dado velho**: usa o histórico daquela faixa e horário; sem histórico, plano fixo seguro.
- **Queda do enlace**: a borda segue o último plano por 5 ciclos e depois cai no plano fixo local.
- **Troca de plano**: sem salto de defasagem; corrige no máximo 20% do ciclo por ciclo, nunca corta mínimos.

## Resultado em simulação (SUMO)

Corredor de 5 cruzamentos (250–400 m), arterial com 2 faixas por sentido, transversais de 1 faixa,
conversões permissivas, 3 períodos de 20 min (moderado, pico assimétrico, madrugada), mistura com ônibus,
caminhões e motos, uma ambulância no pico e **câmera do cruzamento 2 falhando por 10 min** (só no IITLS).
A telemetria passa pelo protocolo binário de verdade e cada cruzamento roda o mesmo executor local da borda.
Média de 3 sementes (`python -m simulacao.rodar`):

| métrica | fixo (90 s) | atuado isolado | IITLS sem onda | **IITLS** |
|---|---|---|---|---|
| atraso médio (s/veic) | 86,7 | 37,2 | 51,4 | **38,2** |
| atraso no arterial (s/veic) | 146,4 | 47,9 | 59,8 | **39,9** |
| paradas por veículo | 2,0 | 1,5 | 1,8 | **1,3** |
| paradas no arterial | 3,1 | 1,9 | 2,3 | **1,3** |
| atraso de pessoas (h) | 142,7 | 61,5 | 85,7 | 63,1 |
| atraso de ônibus (s) | 81,6 | 35,6 | 50,2 | 37,0 |
| CO₂ (kg) | 1103 | 870 | 930 | **848** |
| ambulância, tempo de viagem (s) | 221 | 175 | 150 | **130** |

Leitura honesta:
- Contra **tempo fixo** (o que a maioria das cidades usa): −56% de atraso, −35% de paradas, −23% de CO₂, ambulância 41% mais rápida.
- Contra **controlador atuado** (baseline forte, já é "semáforo inteligente" isolado): arterial −17% de atraso e −30% de paradas,
  −3% de CO₂, ambulância −26%; atraso médio total ainda 3% pior, porque as transversais esperam mais. Próximo passo de algoritmo.
- A coluna "sem onda" mostra que a coordenação (não só a adaptação) é o que faz a diferença.

## Rodar

```bash
cd nuvem
pip install -r requirements.txt                 # serviço
pip install -r requirements-sim.txt             # simulação
python -m pytest -q                             # testes
python -m simulacao.rodar --seeds 1 2 3         # comparação no SUMO (segundos por rodada)
python -m simulacao.cenario --saida simulacao/out   # só gera rede + topologia.json
IITLS_TOPOLOGY=simulacao/out/topologia.json uvicorn iitls_cloud.api:app --port 8000
```

Transporte de produção: MQTT (`IITLS_MQTT_HOST`, `IITLS_MQTT_PORT`), com tópicos `iitls/N/up`, `iitls/N/plan` (retido) e
`iitls/N/cmd`; ver `mqtt_bridge.py` (lado nuvem `CloudBridge`, lado borda `EdgeLink`). Alternativa: UDP 9700
(`IITLS_UDP_PORT`, 0 desliga). HTTP: `POST /v1/ingest` (binário ou JSON), `GET /v1/plans/{nó}`,
`GET /v1/state` (dashboard, inclui enlace online e se a borda está no plano certo), `POST /v1/replan`.

## Topologia

`topologia.json` descreve cruzamentos, faixas (com saturação e alcance da câmera), fases (mínimos, máximos,
entreverdes, mínimo de pedestre) e ligações (origem/destino, fases, faixas, distância, velocidade).
`simulacao/cenario.py` gera uma a partir da rede do SUMO; em campo ela sai do cadastro de cada cruzamento.

## Limitações conhecidas / próximos passos

- Fases em anel único (sequência simples). Falta modelar anel duplo NEMA e conversões protegidas como fases próprias.
- A saturação medida pela borda (`sat_obs`) já trafega no protocolo, mas está desligada (`learn_sat_flow`) até validar a
  medição em campo: na simulação ela saiu enviesada para baixo.
- Controle de transbordo usa só o que a câmera do cruzamento de baixo enxerga; uma câmera de saída no cruzamento de cima
  detectaria o bloqueio antes.
- Interface com o controlador físico (relé, NTCIP 1202, protocolo do fabricante) depende do equipamento de cada cidade.
- Validado em um corredor; grade com ciclos só nos testes unitários.
