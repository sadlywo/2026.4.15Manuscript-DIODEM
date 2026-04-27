| Module | Setting | Value | Presentation Note |
| --- | --- | --- | --- |
| Task definition | Input | `Nonrigid IMU window X_nr, shape [B, T, 6]` | 6-axis inertial window from nonrigid attachment |
| Task definition | Target | `Rigid-reference IMU window X_r, shape [B, T, 6]` | Supervised compensation target |
| Task definition | Prediction form | `X_hat = X_nr + Delta X` | Residual compensation rather than direct replacement |
| Data setup | Sampling frequency | `40 Hz` | Directly parsed from DIODEM headers and enforced in preprocessing |
| Data setup | Window / stride | `64 samples / 16 samples` | Equivalent to `1.6 s` window and `0.4 s` hop |
| Channels | Input channels | `acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z` | Six synchronized IMU channels |
| Normalization | Preprocessing | `per-channel z-score` | Applied before training and evaluation |
| Backbone | Model name | `TCN-causal` | Deployment-oriented main model |
| Backbone | Temporal blocks | `4 residual Conv1D blocks` | Dilated temporal modeling |
| Backbone | Dilation schedule | `1, 2, 4, 8` | Expands temporal receptive field |
| Backbone | Hidden dimension | `64` | Compact temporal backbone width |
| Backbone | Kernel size | `3` | Used in each temporal convolution |
| Attachment modeling | Attachment latent | `8-dim latent state` | Encodes attachment / coupling condition |
| Attachment modeling | Feature modulation | `Feature-wise gate + shift` | Attachment-aware modulation of backbone features |
| Streaming | Causal meaning | `Uses only current and past samples` | No future samples are used during inference |
| Streaming | Receptive field | `61 samples` | Effective history used by causal inference |
