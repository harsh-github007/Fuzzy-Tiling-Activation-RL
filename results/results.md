# Results

Re-run of the experiments in Raj and Shaik (ETTIS 2024) at reduced scale. Final return = each seed's mean over the last fifth of training, averaged across seeds, with a 95% t interval.

## Cartpole (50k steps, 5 seeds)

| Agent | Final return | 95% interval |
| --- | ---: | ---: |
| DQN-FTA-tanh (u=1, k=20) | 453.2 | ±38.8 |
| DQN-FTA (u=1, k=20) | 450.8 | ±63.1 |
| DQN-FTA (u=100, k=20) | 439.5 | ±91.0 |
| DQN-FTA-tanh (u=100, k=20) | 337.3 | ±137.8 |
| DQN-FTA (u=0.01, k=20) | 296.3 | ±90.6 |
| DQN-FTA-tanh (u=0.01, k=20) | 242.4 | ±81.4 |

## Lunarlander (100k steps, 3 seeds)

| Agent | Final return | 95% interval |
| --- | ---: | ---: |
| DQN-FTA (u=1, k=20) | 50.2 | ±230.7 |
| DQN-FTA-tanh (u=1, k=20) | -6.6 | ±101.0 |
| DQN-Large | -84.9 | ±63.5 |
| DQN-FTA (u=100, k=20) | -122.3 | ±153.9 |
| DQN | -131.3 | ±101.2 |
| DQN-FTA-tanh (u=100, k=20) | -132.1 | ±124.4 |
| DQN-FTA (u=20, k=20) | -138.3 | ±21.6 |
| DQN-FTA (u=0.01, k=20) | -167.4 | ±136.2 |
| DQN-FTA-tanh (u=0.01, k=20) | -169.3 | ±45.5 |
