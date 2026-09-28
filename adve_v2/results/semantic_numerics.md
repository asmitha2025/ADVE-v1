# Content-space routing benchmark

```

  CONTENT-SPACE ROUTING — numerics_high_res.mp4
  400 frames analysed · 5112s · 20 OCR-grounded queries · top-5 · text weight 1
  text observations: 396/400 non-empty · 73 changes detected
  --------------------------------------------------------
  arm                 calls       hit@k   precision@k
  full                  400       0.650         0.500
  uniform               190       0.600         0.420
  router                188       0.650         0.450
  router+text           190       0.600         0.430
  --------------------------------------------------------
  router vs uniform:      +0.050 hit@k
  router+text vs uniform: +0.000 hit@k
  router+text vs router:  -0.050 hit@k

```
