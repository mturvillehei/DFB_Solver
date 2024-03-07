# DFB_Solver

#Based off of the Transfer Matrix Model from S. Li et al., IEEE J. Sel. Topics. Quantum Electron. 9, 1153 (2003)
#Code originally written by C. Siegler in .m for the Mawst/Botez lab. Converted to Python and wrapped.
#Unless specified otherwise, znits are cm

Solver for COMSOL outputs, with some assumptions on the output shape from COMSOL. Based on the original Matlab script. Benchmarked --> 7-10x performance increase per iteration!

EEDFB_Finite_Mode_Solver is built for loading JSON files, containing parameters for the execution.

Process flow is as follows:

In COMSOL, a 3D parametric sweep is performed - the values populate the first three columns. An additional 11 columns are populated with:

Raw data is output as a text file. Place this text file in the Data folder, and update the corresponding template JSON file with the filename.
Additionally, update any other parameters. Parametric sweeps over multiple config files/multiple parameters can be additionally handled by wrapping the Main() function.

Within EEDFB_Finite_Mode_Solver, data is loaded.

Data is sorted via Sort_Data_Infinite.py

Sorted_Data is then input, along with parameters (Defined in the JSON file), into Solve().

Solve() performs the Transfer Matrix Method (Li et all, 2003)
Coupled_Wave_Solver() calculates the resultant field profiles 

Data is output for each point in the parametric sweep, converted to lists, and then stored in the original JSON file.

Upon repeat execution, the solve() function isn't re-executed. Data instead is loaded and input into the post-processing steps for parameter calculation.

Parametric sweeps of non-trivial values are performed inside param_Sweep, creating a new JSON output file for each point in the sweep.
Parametric sweeps of trivial values (i.e. CT, GH, DC) are performed in results_Sweep. This enables collecting a table of results for linear plotting.
