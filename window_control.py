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

class MyMainWindow(QMainWindow,Ui_MainWindow):
    def __init__(self,parent =None):
        super(MyMainWindow,self).__init__(parent)
        ti.init(arch=ti.cpu,kernel_profiler=True, debug=True)#,cpu_max_num_threads=1, debug=True
        self.setupUi(self)
        self.config = Config()
        # if button1 is clicked, call the function
        self.pushButton.clicked.connect(self.open_single_file)
        # if button2 is clicked, call the function
        self.pushButton_2.clicked.connect(self.open_multi_files)
        # if button3 is clicked, call the function
        self.pushButton_3.clicked.connect(self.rerun_file)
        # simulation method, defined in utils.py dictionary. default:social force
        self.simu_method = 1
        # simulate frame, -1 means running till exit
        self.frame = -1
        # output csv file or not, 0 means not output
        self.export_csv = 0
        # output file path
        self.out_path = ""
        self.ggui = 0

    def get_scene_name(self,file_name):
        """get scene name from file name, scene name is the last part of file name"""
        return file_name.split("/")[-1].split(".")[0]
    
    def get_scene_path(self,file_name):
        """get scene path from file name, scene path is the path before file name"""
        return file_name.split("/")[:-1]

    def clear_file(self):
        """clear output file content"""
        with open(self.out_path+'/outfile.txt', 'w') as f:
            f.write("")
        
    def rerun_file(self):
        """rerun crowd simulation"""
        file_name = self.textBrowser.toPlainText()
        if file_name:
            # read ini file           
            self.read_ini_file(file_name)
            # set output file path and clear output file content
            self.clear_file()
            # run crowd simulation
            self.run_crowd_simulation(file_name)

    def open_single_file(self):
        """open single file from file dialog"""
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", "ini Files (*.ini)")
        if file_name:
            self.textBrowser.setText(file_name)
            # read ini file           
            self.read_ini_file(file_name)
            # set output file path and clear output file content
            self.out_path = "/".join(self.get_scene_path(file_name))
            self.clear_file()
            # run crowd simulation
            self.run_crowd_simulation(file_name)

    def open_multi_files(self):
        """open multi ini files from file dialog and run crowd simulation one by one"""
        files, _ = QFileDialog.getOpenFileNames(self, "Open Files", "", "ini Files (*.ini)")
        if files:
            # set output file path and clear output file content
            self.out_path = "/".join(self.get_scene_path(files[0]))
            self.clear_file()

            # read files and set its content to textbrowser
            for file_name in files:
                self.textBrowser.append(file_name)
                with open(file_name, 'r') as f:
                    self.textBrowser.append(f.read())

            # read ini file one by one and run crowd simulation
            for file_name in files:
                self.read_ini_file(file_name)
                self.run_crowd_simulation(file_name)
    
    def read_ini_file(self, file_name):
        """read ini file and set config"""
        cfg = ConfigParser()
        cfg.read(file_name)
        src = cfg.sections() 
        for section in src:
            if section == "SIMULATION":
                # set simulation config value
                for item in cfg.items(section):
                    if item[0] == "frame":
                        self.frame = int(item[1])
                    elif item[0] == "export_csv":
                        self.export_csv = int(item[1])
                    elif item[0] == "method":
                        # get simulation method from utils.py dictionary, if not found, use ORCA
                        self.simu_method = utils.METHODS.get(item[1],1) 
                    elif item[0] == "ggui":
                        self.ggui = int(item[1])
                    
            elif section == "SCENE":
                # set scene predefined in config.py
                for item in cfg.items(section):
                    if item[0] == "scene":
                        self.config.set_scene(item[1])
                    elif item[0] == "pos_path":
                        self.config.fill_pos0_from_csv(item[1])
                    elif item[0] == "astar_file":
                        self.config.astar_file = item[1]
            else: 
                # set config value
                for item in cfg.items(section):
                    self.config.set(item[0], item[1])
            
        self.config.post_init()

    def export_crowd_simulation(self,scene_name="crowd_simulation"):
        """export csv result, not update on output file"""
        gui = ti.GUI(scene_name, res=(self.config.WINDOW_WIDTH, self.config.WINDOW_HEIGHT),background_color=0xffffff)

        people = None
        if self.simu_method == 0:
            people = ORCA_People(self.config)
        elif self.simu_method == 1:
            people = SF_People(self.config)
        elif self.simu_method == 2:
            people = Steer_People(self.config)

        csv = [[] for _ in range(self.config.N)] # Record the linked list of each person's position in each frame
        map_size = [self.config.WINDOW_WIDTH*10, self.config.WINDOW_HEIGHT*10]# cm as unit

        if self.frame == -1:
            # run till exit manually, count the simulation frame
            while gui.running:
                self.frame += 1
                if gui.get_event(ti.GUI.RMB,(ti.GUI.PRESS,ti.GUI.SPACE)):
                    if gui.event.key == ti.GUI.SPACE: 
                        gui.running = False       
                for i in range(self.config.N):
                    csv[i].append(list(map(lambda x,y: x*y ,people.pos[i],map_size)))
                people.compute()
                people.update()
                people.render(gui)
                gui.show()
        else:
            # run for frame times, allow to exit by space key
            for i in range(self.frame):
                if gui.get_event(ti.GUI.RMB,(ti.GUI.PRESS,ti.GUI.SPACE)):
                    if gui.event.key == ti.GUI.SPACE: 
                        self.frame = i
                        gui.running = False
                if self.export_csv == 1:           
                    for i in range(self.config.N):
                        csv[i].append(list(map(lambda x,y: x*y ,people.pos[i],map_size)))
                people.compute()
                people.update()
                people.render(gui)
                gui.show()

        utils.export_csv_UE(csv,self.out_path[:-6]+'data/'+scene_name+'_data.csv') 
        #utils.export_txt_Transformer(csv,self.out_path+'/'+scene_name+'_data.txt') 

    def run_crowd_simulation(self,file_name="crowd_simulation"):
        """run crowd simulation according to config"""
        scene_name = self.get_scene_name(file_name)
        if self.export_csv == 1:
            self.export_crowd_simulation(scene_name)
        else:
            self.test_crowd_simulation(scene_name)
        #ti.profiler.print_scoped_profiler_info()
             
    def test_crowd_simulation(self,scene_name="crowd_simulation"):
        """test crowd simulation according to config and update result on output file"""
        gui = None
        canvas = None
        if self.ggui == 0:
            gui = ti.GUI(scene_name, res=(self.config.WINDOW_WIDTH, self.config.WINDOW_HEIGHT),background_color=0xffffff)
        if self.ggui == 1:
            gui = ti.ui.Window(scene_name, res = (self.config.WINDOW_WIDTH, self.config.WINDOW_HEIGHT), pos = (150, 150),vsync=False) # Unlimited frame rate
            #gui.fps_limit = 144
            canvas = gui.get_canvas()
        
        start = time.time()

        people = None
        if self.simu_method == 0:
            people = ORCA_People(self.config)
        elif self.simu_method == 1:
            people = SF_People(self.config)
        elif self.simu_method == 2:
            people = Steer_People(self.config)

        end = time.time()
        initialize_time = end - start

        # for test
        # utils.export_pos0(self.config.N,self.config.pos_0,people.belong_batch,self.out_path[:-6]+'data/'+scene_name+'_pos0.csv')

        if self.frame == -1:
            if self.ggui == 0:
                # run till exit manually, count the simulation frame
                while gui.running:
                    self.frame += 1
                    if gui.get_event(ti.GUI.RMB,(ti.GUI.PRESS,ti.GUI.SPACE)):
                        if gui.event.key == ti.GUI.SPACE:
                            gui.running = False
                    people.compute()
                    people.update()
                    people.render(gui)
                    gui.show()
            else:
                while gui.running:
                    self.frame += 1
                    if gui.is_pressed(ti.ui.SPACE):
                        gui.running = False
                    people.compute()
                    people.update()
                    people.ggui(canvas)
                    gui.show()
        else:
            # run for frame times, allow to exit by space key
            for i in range(self.frame):
                if self.ggui == 0:
                    if gui.get_event(ti.GUI.RMB,(ti.GUI.PRESS,ti.GUI.SPACE)):
                        if gui.event.key == ti.GUI.SPACE:
                            self.frame = i
                            gui.running = False
                    people.compute()
                    people.update()
                    people.render(gui)
                    gui.show()
                else:
                    if gui.is_pressed(ti.ui.SPACE):
                        self.frame = i
                        gui.running = False
                    people.compute()
                    people.update()
                    # people.ggui(canvas)
                    gui.show()

        end_run = time.time()
        simulation_time = end_run - end
        
        # output the simulation result to file, add the result to the end of the file
        with open(self.out_path+'/outfile.txt', 'a') as f:
            f.write("scene_name: " + scene_name + "\t" + "people_num: " + str(self.config.N) + "\n")
            f.write("initialize time: " + str(initialize_time)+ "s\n")
            f.write("simulation time: " + str(simulation_time)+ "s\n")
            f.write("total time: " + str(initialize_time + simulation_time)+ "s\n")
            f.write("total frame: " + str(self.frame)+ "\n")
            # FPS = frame per second
            f.write("Average FPS: " + str(self.frame / simulation_time)+ "\n")
            if self.config.analysis == 1:
                ArrivalRate = 0.0
                if people.ArrivalRate > 0:
                    ArrivalRate = people.ArrivalRate / self.config.N
                f.write("Arrival People Num: " + str(people.ArrivalRate)+ "\t"+"Arrival Rate: " + str(ArrivalRate)+ "\n")
                CollisionRate = 0.0
                Collision = people.Collision.to_numpy()
                if Collision[0] > 0:
                    CollisionRate = Collision[0] / self.frame
                f.write("Collision times: " + str(Collision[0])+ "\t"+"Collision Rate: " + str(CollisionRate)+ "\n")
                ATTime = 0.0
                if people.TotalTime > 0:
                    ATTime = people.TotalTime / people.ArrivalRate
                f.write("Average Travel Time: " + str(ATTime)+ "\n")
            f.write("\n")
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    myWin.show()
    sys.exit(app.exec_())  