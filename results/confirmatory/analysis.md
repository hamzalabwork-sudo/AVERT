**H1** DTA extracted, adaptive (max K1/K3/K3*) vs K0: 208k vs 15k (x13.5), Wilcoxon p=9.5e-07

**H5a** DTA extracted, adaptive (max K1/K3/K3*) vs K0: 210k vs 12k (x17.3), Wilcoxon p=9.5e-07

**H2a** worst-case payer loss DTA - AVERT: 186k [158k, 218k], p=9.5e-07

**H2b** worst-case payer loss uniform audits q=0.15 - AVERT: 136k [120k, 155k], p=9.5e-07

**H2c** worst-case payer loss uniform audits q=0.100 + penalty/slash - AVERT: 53k [48k, 59k], p=9.5e-07

**H5b** SGCC worst-case payer loss DTA - AVERT: 189k [182k, 197k], p=9.5e-07

Holm-adjusted p: H1=5.7e-06, H2a=5.7e-06, H2b=5.7e-06, H2c=5.7e-06, H5a=5.7e-06, H5b=5.7e-06

**H4** AVERT legitimate-hold rate <= 5% in 100% of 40 fleets (all shifts; max over attackers)


| shift               | defence                                |    worst |   hold |   flagged |
|:--------------------|:---------------------------------------|---------:|-------:|----------:|
| S1-usage-surge      | AVERT                                  |  28753   |  0.009 |         0 |
| S1-usage-surge      | DTA                                    | 214223   |  0.011 |         0 |
| S1-usage-surge      | uniform audits q=0.075 + penalty/slash |  85672.6 |  0.011 |         0 |
| S2-noisy-meters     | AVERT                                  |  21795.1 |  0.024 |         0 |
| S2-noisy-meters     | DTA                                    | 150345   |  0.026 |         0 |
| S2-noisy-meters     | uniform audits q=0.075 + penalty/slash |  40912.7 |  0.026 |         0 |
| S3-more-compromised | AVERT                                  |  31229.8 |  0.004 |         0 |
| S3-more-compromised | DTA                                    | 313820   |  0.007 |         0 |
| S3-more-compromised | uniform audits q=0.075 + penalty/slash | 143687   |  0.008 |         0 |
| S4-unreliable-links | AVERT                                  |  26923.4 |  0.008 |         0 |
| S4-unreliable-links | DTA                                    | 218174   |  0.011 |         0 |
| S4-unreliable-links | uniform audits q=0.075 + penalty/slash |  82234.6 |  0.011 |         0 |
| none                | AVERT                                  |  21953.6 |  0.005 |         0 |
| none                | AVERT -aggregate                       |  53138.4 |  0.005 |         0 |
| none                | AVERT -conformal                       |  23299.2 |  0.005 |         0 |
| none                | AVERT -penalty                         |  21953.6 |  0.005 |         0 |
| none                | AVERT -slash                           | 102798   |  0.004 |         0 |
| none                | AVERT -value weighting                 |  50904.5 |  0.005 |         0 |
| none                | DTA                                    | 207559   |  0.005 |         0 |
| none                | uniform audits q=0.05                  | 189508   |  0.005 |         0 |
| none                | uniform audits q=0.050 + penalty/slash |  82888.5 |  0.005 |         0 |
| none                | uniform audits q=0.075 + penalty/slash |  79457.3 |  0.005 |         0 |
| none                | uniform audits q=0.10                  | 171763   |  0.005 |         0 |
| none                | uniform audits q=0.100 + penalty/slash |  75305.4 |  0.005 |         0 |
| none                | uniform audits q=0.15                  | 157997   |  0.005 |         0 |

**H3** payer loss <= Theorem 2 bound in 100% of 280 cells; profit <= bound in 100%; profit <= 0 at q >= q* in 70% of 50 cells

**H3c** honest devices flagged (max over runs): 0; salami +0.30/+0.40 h detected share (min over fleets): 75%


| attacker       |   detected_share |   median_verified_before_flag |   prop3_m_median |   prop3_m_95 |   honest_devices_flagged |
|:---------------|-----------------:|------------------------------:|-----------------:|-------------:|-------------------------:|
| salami +0.10 h |             0.71 |                         inf   |              6.4 |        12.28 |                        0 |
| salami +0.20 h |             0.8  |                          16   |              4   |         4    |                        0 |
| salami +0.30 h |             0.88 |                           9.7 |              4   |         4    |                        0 |
| salami +0.40 h |             0.98 |                           5.3 |              4   |         4    |                        0 |