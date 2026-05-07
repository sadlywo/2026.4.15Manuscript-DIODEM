**DIODEM Dataset Brief**

This study uses the DIODEM dataset, which provides synchronized rigidly attached IMU recordings, non-rigidly attached IMU recordings, and optical motion capture (OMC) reference measurements acquired during the same motion trials. The local release used here contains 88 synchronized recording sets spanning arm and gait contexts and covers a broad range of motion patterns, from quasi-static and pause conditions to faster and more dynamic behaviors such as shaking, rotation, and gait-related movements. Across the raw paired IMU recordings, this corresponds to 110,730 synchronized time samples (approximately 46 min of data) acquired at 40 Hz; the accompanying OMC streams are sampled at 30-120 Hz depending on the subset. In this work, DIODEM serves two purposes: first, it provides paired rigid/non-rigid IMU measurements that allow motion-artifact compensation to be formulated as a supervised segment-level signal-to-signal mapping problem; second, its diversity of motion conditions enables systematic evaluation of the proposed compensator under both routine and challenging dynamic scenarios. After segment-level expansion, the dataset used here comprises 439 paired segment streams, corresponding to 552,930 synchronized segment-level IMU samples and yielding 33,068 sliding-window training/evaluation instances under the adopted preprocessing configuration.

Figure title suggestion:

**Representative paired rigid and non-rigid IMU signals from DIODEM under diverse motion conditions**

Text to introduce the figure in the dataset paragraph:

As illustrated in Fig. X, the paired DIODEM recordings exhibit markedly different rigid and non-rigid inertial responses across representative motion conditions, including standard motion, shaking, freeze, cyclic movement, dangle-like loose attachment, and slow movement. The acceleration and gyroscope traces show that the discrepancy between rigid and non-rigid measurements is strongly motion-dependent: it remains relatively small under more stable conditions but becomes substantially amplified during dynamic or loosely coupled motion. This paired structure is particularly valuable for the present study, as it provides a direct reference for learning and evaluating motion-artifact compensation under heterogeneous attachment and movement scenarios.

Suggested figure assets for this section:

- [Dataset overview](</E:/VSCode_Study/2026.4.15Manuscript-DIODEM/docs/figures/diodem_dataset_overview.png>)
- [Paired measurement example](</E:/VSCode_Study/2026.4.15Manuscript-DIODEM/docs/figures/diodem_pair_example.png>)
