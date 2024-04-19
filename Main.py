from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, ttk
import json
import matplotlib.pyplot as plt
import numpy as np
from EEDFB_Finite_Mode_Solver import EEDFB_Solver
from EEDFB_Finite_Mode_Solver import load_parameters
from EEDFB_Finite_Mode_Solver import save_parameters
from Post_Process import default_Calculations
from EEDFB_Finite_Mode_Solver import NumpyEncoder
from Sort_Data_Infinite import Sort

import Post_Process

### Init with json_filepath as parameter autofill (slect json template?). Select a .txt file, and the params plot will load to select indices for the mode_solver plot. 
### After loading both, if modes_solved in the 
#json_filepath = r"Data/JSON_Template.json"

### Run the script - GUI prompts the user to load the .txt file
### After loading the TXT file, Sort the data automatically 

### GUI
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title('DFB Solver')
        self.geometry('1920x1080')
        self.configure(bg='#f0f0f0')
        
        self.params = {}
        self.default_JSON_filepath = r"Data/JSON_Template.json"
        self.load_default_json()
        self.canvas = None
        
        self.JSON_file_var = tk.StringVar()
        self.JSON_file = tk.Label(self, textvariable=self.JSON_file_var, bg='#f0f0f0')  
        self.JSON_file.pack()
        
        self.data_file_var = tk.StringVar()
        self.data_file = tk.Label(self, textvariable=self.data_file_var, bg='#f0f0f0')  
        self.data_file.pack()


        self.load_default_COMSOL_data()
        
        ### if not params['mode_solved']:
        ###    EEDFB_Solver()
        ### else:
        ###    default_Calculations()

        self.add_parameter_entries()
        self.init_UI()
    
    def on_close(self):
        print(f"Closing")
        self.quit()
        self.destroy()
    
    def init_UI(self):
        self.add_menu()
        self.add_json_file_button()
        self.add_txt_file_button()
        self.add_index_plot()
        self.add_parameter_entries()
        self.add_widgets
        self.plots_grid()
        
    def load_default_json(self):
        default_JSON = self.default_JSON_filepath
        self.params = load_parameters(default_JSON)
        self.params['Lambda'] = self.params['wavelength'] / (2 * self.params['neff'])

    def load_default_COMSOL_data(self):
        filename = self.params['filename']
        ### Assuming the Data folder is being used. This can be changed.
        try:
            self.filename = filename
        except:
            self.filename = "Data/" + filename
            
        self.load_COMSOL_data(filename)
    
    
    def index_Plot(self, cladding_thickness, grating_height):
        # Generating meshgrid for plotting
        X, Y = np.meshgrid(cladding_thickness, grating_height)

        # Create a Figure object
        fig = Figure(figsize=(8, 5), facecolor = 'none', edgecolor='none')
        ax = fig.add_subplot(111)
        ax.set_facecolor('none')

        # Plot onto the figure's axes
        ax.plot(X, Y, 'ko', linestyle="dashed", markersize=5, label='Parameter Points')
        for i, x_val in enumerate(cladding_thickness):
            for j, y_val in enumerate(grating_height):
                ax.text(x_val, y_val, f"({i},{j})", fontsize=10, ha='right')

        ax.set_xlabel('Cladding Thickness')
        ax.set_ylabel('Grating Height')
        ax.set_title('Parameter Sweep Indices')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Create a canvas and add it to the GUI
        canvas = FigureCanvasTkAgg(fig, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, anchor=tk.NE, fill=tk.NONE)

    def plots_grid(self):
        fig, axs = plt.subplots(2, 3, figsize = (8, 6))
        for i in range(3):
            for j in range(2):
                axs[i,j].plot(np.random.rand(10))
        canvas = FigureCanvasTkAgg(fig, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    
    def load_JSON(self, filepath):
        self.params = load_parameters(filepath)
        self.params['Lambda'] = self.params['wavelength'] / (2 * self.params['neff'])

    ### This can be generalized for arbitrary x1 x2 x3 
    def load_COMSOL_data(self, filename):
        fn = "Data/" + filename
        derived_values = Sort(fn, self.params['wavelength'], self.params['Lambda'] )
        self.derived_values = derived_values
        self.params['cladding_thickness'] = derived_values['params'][0]
        self.params['grating_height'] = derived_values['params'][1]
        self.params['duty_cycle'] = derived_values['params'][2]
    # def save_params(self):
    # save_parameters(self.params)
    
    def update_GUI(self):
        pass

    def add_menu(self):
        self.menu_bar = tk.Menu(self)
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(label="Load parameters file (.json)", command=self.open_json_file)
        self.file_menu.add_command(label="Load COMSOL results (.txt)", command=self.open_txt_file)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        self.config(menu=self.menu_bar)

    def open_json_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("Parameters Files", "*.json")])
        if filepath:
            self.JSON_file_var.set(filepath)  # Set the file path in the entry
            # self.json_file_path_entry.config(state='normal') # If you need to update the entry
            # self.json_file_path_entry.insert(0, filepath)
            # self.json_file_path_entry.config(state='readonly') # Make the entry readonly again
            self.load_JSON(filepath)

    def open_txt_file(self):
        filename = filedialog.askopenfilename(filetypes=[("COMSOL Results Files", "*.txt")])
        if filename:
            self.data_file_var.set(filename)  # Set the file path in the entry
            # self.txt_file_path_entry.config(state='normal') # If you need to update the entry
            # self.txt_file_path_entry.insert(0, filename)
            # self.txt_file_path_entry.config(state='readonly') # Make the entry readonly again
            self.load_COMSOL_data(filename)
                        
    def add_widgets(self):
        self.run_sim_button = ttk.Button(self, text="Run Simulation", command=lambda: self.run_simulation(self.params))
        self.run_sim_button.pack(pady=10)

        self.post_process_button = ttk.Button(self, text="Post-Process", command=lambda: Post_Process(self.params))
        self.post_process_button.pack(pady=10)
    def run_simulation(self, params):
        print(f"solve here")
        #params_ = EEDFB_Solver(params)
        params_ = params
        self.params.update(params_)  # Update the internal parameter state
        self.update_gui_based_on_params() 
        
    def add_index_plot(self):      
        self.index_Plot(self.params['cladding_thickness'], self.params['grating_height'])
    
    def add_json_file_button(self):
        self.open_json_button = ttk.Button(self, text='Open JSON File', command=self.open_json_file)
        self.open_json_button.pack()

        # Display path in a readonly entry
        self.json_file_path_entry = ttk.Entry(self, textvariable=self.JSON_file_var, state='readonly', foreground='grey')
        self.json_file_path_entry.pack()

    def add_txt_file_button(self):
        self.open_txt_button = ttk.Button(self, text='Open TXT File', command=self.open_txt_file)
        self.open_txt_button.pack()

        # Display path in a readonly entry
        self.txt_file_path_entry = ttk.Entry(self, textvariable=self.data_file_var, state='readonly', foreground='grey')
        self.txt_file_path_entry.pack()
        
    # def open_file(self):
    #     file_path = filedialog.askopenfilename()
    #     if file_path:
    #         print(f'Selected file: {file_path}')
    #         self.file_label_var.set(file_path)

    def setup_file_loaders(self):
        json_button = ttk.Button(self, text='Open JSON File', command=self.open_json_file)
        json_button.pack()
        json_label = tk.Label(self, textvariable=self.json_file_path_var, bg='#f0f0f0')
        json_label.pack()

        comsol_button = ttk.Button(self, text='Open COMSOL File', command=self.open_txt_file)
        comsol_button.pack()
        comsol_label = tk.Label(self, textvariable=self.comsol_file_path_var, bg='#f0f0f0')
        comsol_label.pack()

    def add_parameter_entries(self):
        self.param_frame = tk.Frame(self)
        self.param_frame.pack(side=tk.BOTTOM, fill=tk.X)

        tk.Label(self.param_frame, text='Parameter 1').pack(side=tk.LEFT)
        self.param1_entry = ttk.Entry(self.param_frame)
        self.param1_entry.pack(side=tk.LEFT)

        tk.Label(self.param_frame, text='Parameter 2').pack(side=tk.LEFT)
        self.param2_entry = ttk.Entry(self.param_frame)
        self.param2_entry.pack(side=tk.LEFT)

        self.update_button = ttk.Button(self.param_frame, text='Update Parameters', command=self.update_parameters)
        self.update_button.pack(side=tk.RIGHT)

    def update_parameters(self):
        # Example for updating parameters
        self.params['Parameter1'] = self.param1_entry.get()
        self.params['Parameter2'] = self.param2_entry.get()
        
        # Save the updated parameters back to the JSON file if necessary
        save_parameters(self.params, self.default_JSON_filepath)
        
        # Decide whether to re-run simulation or just update plots
        if self.needs_recalculation(self.params):
            self.run_simulation(self.params)
        else:
            self.update_GUI()
if __name__ == '__main__':
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
