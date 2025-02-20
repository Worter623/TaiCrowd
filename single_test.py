import taichi as ti
import sys
from PyQt5.QtWidgets import QMainWindow,QApplication,QWidget,QFileDialog
from Ui_enter_window import Ui_MainWindow 
from configparser import ConfigParser
from config import Config
from orca_people import ORCA_People
from sf_people import SF_People
from steer_people import Steer_People
import taichi as ti
import time
import utils
ti.init(arch=ti.cpu,kernel_profiler=True, debug=True)
config = Config()

def read_ini_file( file_name):
    """read ini file and set config"""
    cfg = ConfigParser()
    cfg.read(file_name)
    src = cfg.sections() 
    for section in src:      
        if section == "SCENE":
            # set scene predefined in config.py
            for item in cfg.items(section):
                if item[0] == "scene":
                    config.set_scene(item[1])
        else: 
            # set config value
            for item in cfg.items(section):
                config.set(item[0], item[1])
        
    config.post_init()

read_ini_file("config/triangle_circle_socialforce.ini")
gui = None
gui = ti.GUI("test", res=(config.WINDOW_WIDTH, config.WINDOW_HEIGHT),background_color=0xffffff)

start = time.time()

people = SF_People(config)


end = time.time()
initialize_time = end - start


# run till exit manually
while gui.running:
    if gui.get_event(ti.GUI.RMB,(ti.GUI.PRESS,ti.GUI.SPACE)):
        if gui.event.key == ti.GUI.SPACE:
            gui.running = False
    people.compute()
    people.update()
    people.render(gui)
    gui.show()

end_run = time.time()
simulation_time = end_run - end
print(simulation_time)
