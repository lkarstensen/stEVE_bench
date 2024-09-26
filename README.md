# stEVE_bench
Collection of benchmark environments based on [stEVE - simulated Endovascular Environment](https://github.com/lkarstensen/stEVE) for research in robotic endovascular interventions. 

## Getting Started

1. Setup [stEVE](https://github.com/lkarstensen/stEVE?tab=readme-ov-file#getting-started) (including Sofa)
2. Install stEVE_bench package
   ```
   python3 -m pip install -e .
   ```
3. Test the installation
    ```
    python3 examples/function_check.py
    ```


## Benchmark Environments


|               |                                                    |
| ------------- | -------------------------------------------------- |
| BasicWireNav  | <img src="figures/BasicWireNav.gif" width="600"/>  |
| ArchVariety   | <img src="figures/ArchVariety.gif" width="600"/>   |
| DualDeviceNav | <img src="figures/DualDeviceNav.gif" width="600"/> |

## How to use
This collection implements *interventions* of the EVE framework. 

You can use the intervention directly if you add a visualization from stEVE. Look [here](https://github.com/lkarstensen/stEVE_bench/tree/main/example) to find examples how to add the visualization. 

To be used as gymnasium Env you need to add observation, reward, terminal and truncation from stEVE. Look [here](https://github.com/lkarstensen/stEVE_training/tree/main/training_scripts) to find examples how to use the benchmark environments in an reinforcement learning setup. 
