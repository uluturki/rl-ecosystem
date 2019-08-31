# The large-scale deep multi-agent reinforcement learning for predator-prey ecosystem

Implementation for the master thesis **Evolution of a Complex predator-PreyEcosystem on Large-scale Multi-AgentDeep Reinforcement Learning**

## How to Use

### Training

```
python main.py -- algorithm_name  --experiment_id id(int) --env_type env_name(string)
```

For example

```
python main.py -- drqn --experiment_id 1 --env_type genetic_population_dynamics
```

All results are stored in the result folder.

### Test

```
python test.py --model_file model_file_path(str) --experiment_id id(int) --test_id test_id(int)  --path_prefix folder_path(which you stored the result during training)
```

For example,

```
python test.py --model_file ./results/simple_population_dynamics/exp_1/models/model_1.h5 --experiment_id 1 --test_id 1  --path_prefix ./results/simple_population_dynamics/exp_1/
```


### Algorithms

Deep Q-learning, Double Q-learning, and Deep Recurrent Q-learning are implemented. Those models are basically for a single agent reinforcement learning. Therefore, we extend those models to the large-scale deep multi-agent reinforcement learning


