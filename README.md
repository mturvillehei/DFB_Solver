# EEDFB_Solver

~~Based off of the Transfer Matrix Model from S. Li et al., IEEE J. Sel. Topics. Quantum Electron. 9, 1153 (2003)~~
Code originally written by C. Siegler in .m for the Mawst/Botez lab. Converted to Python and wrapped.

Solver for COMSOL outputs, with some assumptions on the output shape from COMSOL.

EEDFB_Finite_Mode_Solver is built for loading JSON files, containing parameters for the execution.

Process flow is as follows:

In COMSOL, a 3D parametric sweep is performed - the values populate the first three columns. An additional 11 columns are populated with:
##### fill this

Raw data is output as a text file. Place this text file in the Data folder, and update the corresponding template JSON file with the filename.
Additionally, update any other parameters. Parametric sweeps over multiple config files/multiple parameters can be additionally handled by wrapping the Main() function.

Within Main, data is loaded.

Data is sorted via Sort_Data_Infinite.py

Sorted_Data is then input, along with parameters (Defined in the JSON file), into Solve().

Solve() performs two primary processes:
	Transfer Matrix Method (Li et all, 2003)
	Coupled Wave Solver.

Data is output for each point in the parametric sweep, converted to lists, and then stored in the original JSON file.

Upon repeat execution, the solve() function isn't re-executed. Data instead is loaded and input into the post-processing steps for parameter calculation.