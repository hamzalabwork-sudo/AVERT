fleets: [np.int64(430), np.int64(431), np.int64(432), np.int64(433), np.int64(434), np.int64(435), np.int64(436), np.int64(437), np.int64(438), np.int64(439)]

**H6a** RL payer loss <= Theorem 2 bound under AVERT in 10/10 fleets (median loss/bound = 0.045)

**H6b** RL loss DTA 74161 vs AVERT 7871; p_Holm=0.00195

**H6c** RL loss under DTA 74161 vs K1 69969; p_Holm=0.0186

AVERT worst case over references 21643 -> with RL 21643


|                              |   payer_net_loss |   extracted_value |
|:-----------------------------|-----------------:|------------------:|
| ('AVERT', 'K0 oblivious')    |             3419 |              4469 |
| ('AVERT', 'K1 feedback')     |             4635 |             14789 |
| ('AVERT', 'K3 white-box')    |            21643 |             27999 |
| ('AVERT', 'K3* risk-aware')  |              277 |             21895 |
| ('AVERT', 'RL (Q-learning)') |             7871 |             10369 |
| ('DTA', 'K0 oblivious')      |            10287 |             10287 |
| ('DTA', 'K1 feedback')       |            69969 |             69969 |
| ('DTA', 'K3 white-box')      |           162130 |            162130 |
| ('DTA', 'K3* risk-aware')    |           178691 |            178691 |
| ('DTA', 'RL (Q-learning)')   |            74161 |             74161 |