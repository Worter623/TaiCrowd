# TaiCrowd: A High-Performance Simulation Framework for Massive Crowd

This is a framework that utilize Taichi to improve the performance for massive crowd simulation.

We now support social force model, steering model, and ORCA model for crowd simulation.




## Requirements

Simulation is run with Taichi version 1.4.1, python 3.9.7.

Install the required dependencies:

```bash
pip install -r requirements.txt
```



## Running the Simulation

### 1. GUI-based Simulation

To run the GUI and start the simulation, execute the following command:

```bash
python .\window_control.py
```

- After running the command, a pyQT GUI window will appear.
- You can start a simulation by clicking one of the following buttons:
  - **Single Scene**: Choose a single `.ini` file from your file explorer, click "Rerun" button to re-simulate current scene.
  - **Batch Test**: Choose multiple `.ini` files from your file explorer, simulation will start one by one.
- The **current scene** being simulated will be displayed under the "Current Scene" textbox.

### 2. Customizable Script

You can also use the following command:

```bash
python .\single_test.py
```

This example script is an example to use the code and run the simulation without pyQT GUI window involving.



## Directory Structure

- **[config](./config)**: Sample simulation scenes and configurations.
- **[map](./map)**: Contains map images used for simulation.
- **[data](./data)**: Stores additional data files like CSVs for A* and agent initial positions.



## INI File Configuration

The simulation is configured through an `.ini` file, which consists of three main sections as follows: 

```ini
[SIMULATION]
method : choose from utils.METHODS, if not found, will use ORCA
export_csv : export agent's position in each frame enables data-driven crowd simulation with professional rendering engines using exported data. you can adjust the output file format in export_crowd_simulation() in window_control.py
frame : limit the simulation frames
ggui : use taichi gui or ggui to render the result
analysis : whether to call function self.analysis()

[CONFIG_VALUE]
in this section, set all the params you can see in config.py through function set()
path : file path pointing to a binary image as simulation map, such as ./map/maze.png
[SCENE]

scene : a string choose from config.set_scene()
astar_file : file path pointing to a csv that stored previously calculated A* results, such as ./data/astar_maze_rl140.csv
pos_path : file path pointing to a csv, the contents of which will be used as the initial location of the agent, such as ./data/Amaze_group_pos0_v1.csv
```

Please see ini files from **[config](./config)** for more detailed example, like [this](./config/Amaze_nogroup.ini).



## Additional Information

- The GUI-based simulation (using `window_control.py`) provides a user-friendly way to test various simulation scenes. You can modify the simulation behavior by editing the `.ini` files as needed. Our paper's performance evaluation experiment with this framework is conducted using social force model without GUI rendering.
- Video clips and simulation demo: Coming soon.

If you encounter any issues, please feel free to contact us or open an issue on the repository. We hope this work helps.

