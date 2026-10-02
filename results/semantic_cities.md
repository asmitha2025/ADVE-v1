# Content-space routing benchmark

```

  CONTENT-SPACE ROUTING — cities_and_decarbonization_standard_res.mp4
  400 frames analysed · 2412s · 20 OCR-grounded queries · top-5 · text weight 1
  text observations: 336/400 non-empty · 105 changes detected
  --------------------------------------------------------
  arm                 calls       hit@k   precision@k
  full                  400       0.450         0.400
  uniform               104       0.600         0.240
  router                104       0.750         0.320
  router+text           104       0.800         0.350
  --------------------------------------------------------
  router vs uniform:      +0.150 hit@k
  router+text vs uniform: +0.200 hit@k
  router+text vs router:  +0.050 hit@k

```
