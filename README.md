# Project 2: Lunar Lander

| Path | Description |
| -------     | ----------- |
| **Directories** |
| models/             | Directory of trained models |
| runs/               | Directory of tensorboard logdata for each run |
| utils/              | Python utility scripts                        |
|                                                                     |
| **Python Scripts**                                                  |
| cartpole.py         | Train model for Cartpole-V0                   |
| cartpole-working.py | Script for seeded model for cartpole          |
| lander.py           | Train model for LunaLander-V2                 |
| dqn.py              | Class for neural network                      |
| model.py            | Classes for model training and configurations |
| replay.py           | Class for replay memory                       |
| playmodel.py        | Play trained model                            |
| readrun.py          | Generate graphs for runs                      |
|                                                                     |
| **Misc**                                                            |
| env.yml             | Conda exported environment                    |

### Creating conda environment

Create the conda environment with the saved environment file `env.yml` with the command:

```bash
conda env create -f ./env.yml
conda activate env
```

### Training LunarLander

Start the model training with preconfigured configuration:
```
python ./lander.py
```

To render the environment while training run:
```bash
python ./lander.py render
```

![tensorboard screenshot](./screenshots/tbscreenshot.png)

The training event data is recorded with tensorboard and event files is stored in `./runs` and after the complete run the model is stored in `./models`

#### Configuration

The training parameters and configuration are hardcoded in `lander.py`.

| Config                                | Description                                                |
| ------------------------------------- | ---------------------------------------------------------- |
| config.Training.Gamma = 0.99          | Gamma parameter                                            |
| config.Training.Alpha = 0.0001        | Alpha parameter                                            |
| config.Training.Epsilon = [0.9, 0.05] | Epsilon parameter                                          |
| config.Training.EpsilonDecay = 20000  | Epsilon decay rate                                         |
| config.Training.BatchSize = 128       | Mini batch size                                            |
| config.Training.MemorySize = 10000    | Replay memory size                                         |
| config.Training.MemoryInitFill = 0.2  | Initial replay memory fill percentage                      |
| config.Training.TargetUpdate = 10     | Steps before target network update, (1: no target network) |
| config.Training.EpisodeLimit = 750    | Cut the episode after n steps                              |
| config.HiddenLayers = [100,100]       | Hidden layer size for neural network                       |



### Monitoring live training

To monitor the live training data with tensorboard run the following and navigate to [https://localhost:8080/](https://localhost:8080/)

```bash
tensorbord --logdir=./runs --port=8080
```


### Running trained model

To run a trained model select a model from `./models/` and run `playmodel.py` as follows

```bash
#                      Path to model         Number of episodes to run
python playmodel.py    ./models/run.model    100
```

