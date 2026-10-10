# Protocolo IITLS borda ↔ nuvem (v1)

Contrato entre a Jetson de cada câmera (borda) e o coordenador na nuvem.
Implementação de referência: `nuvem/iitls_cloud/protocol.py` (encode/decode + testes).

## Princípios

- A borda manda **só agregados por faixa**, nunca vídeo nem caixas delimitadoras.
- Formato primário **binário little-endian**, com **CRC-16/CCITT** no fim (enlace de rádio pode corromper bytes).
  O mesmo conteúdo existe em **JSON compacto** (chaves curtas) para debug e HTTP.
- Transporte primário: **UDP** (porta 9700) dentro de um túnel **WireGuard** (criptografia + autenticação do nó).
  Cada mensagem da borda é respondida com o plano vigente do nó, então não precisa de conexão aberta nem polling.
- **Nada de segurança depende da nuvem.** O executor local (`edge_runtime.py`) garante mínimos, entreverdes,
  preempção local e plano fixo de segurança quando o plano vence (`valid_until`).
- Relógios sincronizados (GPS/PPS ou NTP via chrony). A onda verde depende disso: o plano é ancorado em tempo absoluto.

## Borda → nuvem

### TELEMETRY (0x01), a cada `interval` s (padrão 5 s)

Cabeçalho, 14 bytes:

| campo | tipo | descrição |
|---|---|---|
| version | u8 | 1 |
| type | u8 | 0x01 |
| node | u16 | id do cruzamento |
| cam | u8 | id da câmera no cruzamento |
| seq | u16 | sequência (detecta perda) |
| t_end | u32 | fim da janela, unix s |
| interval | u8 | duração da janela, s |
| flags | u8 | saúde: 0x01 câmera ok, 0x02 pouca luz, 0x04 oclusão, 0x08 relógio sem sync, 0x10 modelo degradado |
| n_lanes | u8 | número de faixas |

Por faixa, 11 bytes:

| campo | tipo | descrição |
|---|---|---|
| lane | u8 | id da faixa no cruzamento (definido na topologia) |
| car, moto, bus, truck, bike, ped | 6 × u8 | quantos cruzaram a linha virtual na janela |
| queue | u8 | veículos parados na fila no fim da janela |
| occupancy | u8 | fração do tempo com a zona ocupada × 200 |
| speed | u8 | velocidade média em km/h |
| sat_obs | u8 | escoamento medido na retenção com fila e verde, em 10 veic/h (0 = não medido) |

CRC-16 (2 bytes) no fim. **Tamanho: 16 + 11 × faixas.** Câmera com 4 faixas = 60 bytes.

Com UDP/IP (28 B) e WireGuard (32 B): ~120 B a cada 5 s = **~0,2 kbit/s por câmera**, ~62 MB/mês.
O plano de volta tem tamanho parecido.

Posição da linha virtual de contagem: **≥ 50 m antes da retenção**, para contar chegadas (e não só a
descarga da fila). A zona de fila/ocupação cobre o que a câmera enxerga a partir da retenção (`view_m`, padrão 80 m).

JSON equivalente:

```json
{"v":1,"m":1,"n":7,"c":1,"s":42,"t":1760000000,"dt":5,"f":1,"l":[[0,3,0,1,0,0,2,4,70,31,165]]}
```

### EVENT (0x02), na hora, 17 bytes

`version u8, type u8, node u16, cam u8, seq u16, t u32, event u8, lane u8, value u16, crc u16`

| event | value |
|---|---|
| 1 veículo de emergência | ETA até a retenção, décimos de s |
| 2 ônibus chegando | ETA, décimos de s |
| 3 incidente (faixa bloqueada) | duração estimada, s |
| 4 incidente liberado | — |
| 5 falha de câmera | código |
| 6 câmera restaurada | — |

### HEARTBEAT (0x03), 15 bytes

`version, type, node, cam, seq, t, flags, temp_c` — saúde da Jetson quando não há telemetria.

## Nuvem → borda

### PLAN (0x10), 22 + 5 × fases bytes

| campo | tipo | descrição |
|---|---|---|
| version, type | u8, u8 | 1, 0x10 |
| node | u16 | |
| plan_id | u16 | muda quando o plano muda |
| t_ref | u32 | referência de tempo comum à rede |
| cycle | u16 | ciclo em décimos de s |
| offset | u16 | defasagem em décimos de s |
| valid_until | u32 | unix s; depois disso a borda cai no plano de segurança |
| mode | u8 | 0 normal, 1 baixa demanda, 2 saturado, 3 preempção, 4 falha, 5 piscante |
| n_phases | u8 | |
| por fase | u8 + u16 + u8 + u8 | id, verde (décimos), amarelo (décimos), vermelho geral (décimos) |

Semântica: a **primeira fase é a coordenada** e começa em `t_ref + offset + k × cycle`. As demais fases
seguem na ordem; os verdes delas são **máximos** (a borda pode encurtar/pular por falta de demanda e devolver
o tempo à fase coordenada). Troca de plano é feita pela borda **sem salto**: corrige no máximo 20% do ciclo por ciclo.

### PREEMPT (0x11), 12 bytes

`version, type, node u16, phase u8, t_start u32, hold u8, crc` — verde garantido de `hold` s na fase,
a partir de `t_start`. A borda que detecta a viatura preempta sozinha; a nuvem manda PREEMPT para os
cruzamentos seguintes (corredor verde), com `t_start` escalonado pelo tempo de viagem.

## Tempos

| etapa | valor |
|---|---|
| agregação na borda | 5 s (configurável) |
| replanejamento na nuvem | a cada 60 s, ~10–80 ms de CPU para 5 cruzamentos |
| aplicação do plano | no próximo limite de ciclo, com transição suave |
| emergência | imediata na borda; corredor a jusante em 1 RTT (rádio/Starlink: 20–60 ms) |
| perda de enlace | plano atual segue até `valid_until` (5 ciclos), depois plano fixo local |
