# PriEco-DRL

Minimal source-code implementation of **PriEco-DRL: Coordinated Electric-Bus Eco-Driving under Transit-Priority Adaptive Signal Control via Deep Reinforcement Learning**.

PriEco-DRL combines pressure-aware electric-bus acceleration control, stop-approach guidance, and transit-priority max-pressure signals. This release contains five Python source files from the maintained `Four_line` implementation, including stop/intersection overlap guidance and a continuous green-window slack reward.

## Included files

| File | Role |
| --- | --- |
| `sutraffic_env.py` | Gymnasium environment; observations, stage selection, rewards, electrical-energy model, and vehicle execution. |
| `extract_CV_information.py` | Connected-vehicle and electric-bus state extraction. |
| `pressure_controller.py` | Movement-pressure calculation and transit-priority phase selection. |
| `train_rl_par.py` | Parallel-environment SAC training entry point. |
| `eval.py` | Shared-policy bus evaluation and metric export. |

The energy model is implemented inside `sutraffic_env.py`; no separate energy postprocessing script is needed by these modules. Only the source implementation is distributed here. Corridor input files, pretrained models, normalization statistics, and experimental outputs are outside this release.

## Software setup

Use Python 3.10 and install the Python dependencies:

```sh
python -m pip install -r requirements.txt
```

Install SUMO separately and configure its command-line binaries on `PATH` and its installation directory in `SUMO_HOME`. The supplied Python TraCI and sumolib bindings are version 1.18.0. The dependency versions describe the inspected source-test environment; `tensorboard` is additionally required by the training logger.

The three core modules can be imported without starting a simulation:

```sh
python -B -c "import sutraffic_env, extract_CV_information, pressure_controller; print('Core imports OK')"
```

## Inputs for training and evaluation

The entry points expect a compatible SUMO scenario in the working directory:

- `1_osm.sumocfg` and the network, routes, stop definitions, and other files referenced by that configuration;
- `For_correcting_LTT.txt`, `Bus_stop_position.txt`, `Edge_length_speed.txt`;
- `In_out_pairs.txt`, `Signal_phase.txt`, `Sourced_link.txt`, and `Corridor_edge.txt`.

The pressure controller uses the corridor junction identifiers `TL_ZZ`, `TL_IM`, and `TL_WL`, with 16, 22, and 24 movement slots. A different network requires corresponding movement mappings and signal-controller configuration. The entry points read scenario metadata at import time and are intended to run after these external inputs are supplied.

Training:

```sh
python train_rl_par.py
```

Inspect the settings before launching: the supplied entry point uses ten parallel environments, 500,000 training transitions per run, and two sequential training seeds. Reward weights are configured by `w_energy` and `w_eff` in `SUTrafficEnv.step`; the current source defaults to the energy-oriented setting `(2, 1)`. The paper's T/E/B preferences use energy/time weights `(1, 2)`, `(2, 1)`, and `(1, 1)`, respectively, with the auxiliary scale held fixed.

Evaluation additionally requires a compatible `best_model.zip` and its matching `vecnorm.pkl`. Set `PRIECO_MODEL_DIR` to their directory, then run:

```sh
python eval.py
```

The evaluation defaults to Priority-MP signals, traffic seed 1, and a three-hour horizon. Configure these arguments in the final `run_eval` call. Predictions use the loaded policy and its saved observation-normalization statistics. Load model and normalization pickle files only from trusted sources.

## Implementation conventions

The internal stage labels retain the source implementation's numbering:

| Internal label | Driving task | Manuscript stage |
| --- | --- | --- |
| 1 | Scheduled dwell | 3 |
| 2 | Intersection approach/crossing | 4 |
| 3 | Cruise | 1 |
| 4 | Stop approach | 2 |

These labels identify reward branches. During intersection-led overlap, stop-approach guidance is added to the intersection guidance. Acceleration execution and safety-relevant reward terms are defined in the environment source.

This is a snapshot of the maintained implementation, rather than an archive of every historical training revision. Evaluation of a previously trained policy requires its matching observation definition, scenario settings, and normalization state. Published table and figure data are not bundled with this code-only release.

## Release preparation

The three core modules and the training entry point retain their source contents. The evaluation copy uses a configurable model directory, timestamped output directories, and opt-in debug printing. These changes concern file handling and diagnostics; the policy and reward calculations are retained.

Preparation checks cover Python syntax, the local dependency closure, core-module imports, and lightweight reward/overlap unit tests. No training or traffic simulation is launched as part of these checks.

## License

The distributed source code is provided under the MIT License; see `LICENSE`. External simulators, scenario inputs, pretrained models, and third-party components retain their own terms.
