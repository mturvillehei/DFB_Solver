# DFB_Solver

Based off of the Transfer Matrix Model from S. Li et al., IEEE J. Sel. Topics. Quantum Electron. 9, 1153 (2003). Code originally written by C. Siegler in Matlab for the Mawst/Botez lab. Unless specified otherwise, units are cm.

## Main Solver Script
There are two main scripts, which are functionally identical as solvers: DFB_Finite_Mode_Solver.py and DFBsolve_NP.py. 

##### DFB_Finite_Mode_Solver.py:

In COMSOL, a 3D parametric sweep is performed - the values populate the first three columns. An additional 11 columns are populated with:

Raw data is output as a text file. Place this text file in the Data folder, and update the corresponding template JSON file with the filename.
Additionally, update any other parameters. Parametric sweeps over multiple config files/multiple parameters can be additionally handled by wrapping the Main() function.

Within DFB_Finite_Mode_Solver, data is loaded.

Data is sorted via Sort_Data_Infinite.py

Sorted_Data is then input, along with parameters (Defined in the JSON file), into Solve().

Solve() performs the Transfer Matrix Method (Li et all, 2003)
Coupled_Wave_Solver() calculates the resultant field profiles 

Data is output for each point in the parametric sweep, converted to lists, and then stored in the original JSON file.
 - If using the ND script, data is stored instead into a pickle file.
Upon repeat execution, the solve() function isn't re-executed. Data instead is loaded and input into the post-processing steps for parameter calculation.

Parametric sweeps of non-trivial values are performed inside param_Sweep, creating a new JSON output file for each point in the sweep.
Parametric sweeps of trivial values (i.e. CT, GH, DC) are performed in results_Sweep. This enables collecting a table of results for linear plotting.

##### DFBsolve_NP.py:
This is a modification to the DFB_Finite_Mode_Solver.py script that allows for variable columns. The primary change is the ability to use any number of columns in the COMSOL parametric sweep (e.g. all three cladding thicknesses, grating height, duty cycle, etc). The JSON file is slightly different for the NP simulations, and is labelled as such. The execution of the script with the new JSON file is otherwise the same. 
Please note the parameter 'AMR', which indicates if adaptive mesh refinement was used. This parameter strips the redundant AMR columns during the initial row sorting.

Data is now stored in a pickle file, since the results filesize can grow very large. Be careful of memory - the pickle file will be loaded directly into memory in the Analysis() script. 

Due to the complexity of plotting an N-dimensional set of results, plotting is handled separately in the Analysis() script below. Therefore, all of the JSON config parameters that control plotting is removed.

# Analysis.py

### Overview
The Analysis script handles post-processing calculations with different options for operations to perform on the solved dataset.

### Key Functions

- `setup_logging(log_level)`: Configures logging with specified level.
- `load_config(config_file)`: Loads JSON configuration file.
- `main(args)`: Core execution function, coordinates analysis tasks.

### Usage

```
python Analysis.py --config <config_file> --output-dir <output_directory> --log-level <log_level> [--generate-plots] [--plot-variable <variable>] [--correlation-matrix] [--trend-analysis]
```

Example CLI call:
```
python Analysis.py --config Data/EE_NP_JSON_template.json --output-dir Data/plots --log-level INFO --generate-plots --plot-variable Jth --correlation-matrix --trend-analysis
```

#### Arguments
- `--config`: Path to JSON configuration file (default: 'EE_DFB_comparing_with_text.json')
- `--output-dir`: Directory for output files (default: 'output')
- `--log-level`: Logging level (default: 'INFO')
- `--generate-plots`: Flag to generate contour plots
- `--plot-variable`: Variable to plot (default: 'Jth')
- `--correlation-matrix`: Flag to generate correlation matrix
- `--trend-analysis`: Flag to perform trend analysis

## Analysis Library (np_analysis.py)

### Key Functions

- `trend_analysis(sr, param, target_col)`: Analyzes trends across parameters.
- `find_non_constant_variables(df)`: Identifies non-constant variables in dataset.
- `correlation_matrix(data, correlation_variables, save_path, figsize)`: Generates correlation matrix heatmap.
- `contour_plot(data, x_col, y_col, z_col, ...)`: Creates contour plot for specified variables.
- `generate_contour_plots(sr, param, col_1, col_2, z_col)`: Generates multiple contour plots.
- `analyze_data(sr, ar, param, inputs, csv_filename)`: Performs initial data analysis and exports to CSV.

## Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- scipy

## Workflow
1. Load configuration and data
2. Perform initial data analysis
3. Generate requested plots and analyses (contour plots, correlation matrix, trend analysis)
4. Save results to specified output directory
