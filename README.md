# BayesFit
version 2.8

### Description 
BayesFit allow users to fit analytical models (form factors and structure factors) to SAXS and/or SANS data using Bayesian refinement.

## Installation
Download the scripts from this repository and run BayesFit as a standard Python3 script.  
The following python scripts should be in the same folder as bayesfit.py:    
* bayesfit_functions.py
* formfactors.py

## Running the program
```
python bayesfit.py
```
the program will prompt you for options (pressing enter gives you the default option).       
Alternatively, the input can be given via an input file (a plain text file with each line being an input): 
```
python bayesfit.py < inputfile.txt
```
where each line in the inputfile corresponds to the options that are prompted for (default value are used if a line is left blank). 
For a model with 3 parameters, fitted against a single dataset, the inputfile has the form:    
```
line 1 : output directory (default: output_bayesfit)    
line 2 : number of contrasts (default: 1)    
line 3 : datafile (default: examples/cylinder/Isim_1.dat)    
line 4 : model (default: cylinder)    
line 5 : prior values for the first parameter: mean sigma (optional) min (optional) max (optional)      
line 6 : prior values for the second parameter    
line 7 : prior values for the third parameter
line 8 : logalpha values (default: -5 5 15)    
line 9 : plot data (default: yes)
line 10: plot posterior distributions (default: no)
```
for a model with 4 parameters, fitted against two dataset, and weighted with the information content, the inputfile should have the form:    
```
line 1 : output directory (default: output_bayesfit)    
line 2 : number of contrasts (default 1)
line 3 : datafile 1 (default: examples/cylinder/Isim_1.dat)
line 4 : datafile 2   
line 5 : model (default: cylinder)    
line 6 : prior values for the first parameter: mean sigma (optional) min (optional) max (optional)      
line 7 : prior values for the second parameter    
line 8 : prior values for the third parameter
line 9 : prior values for the fourth parameter
line 10: logalpha values (default: -5 5 15)    
line 11: plot data and fit (default: yes)
line 12: plot posterior distributions (default: no)
line 13: weighting scheme (default: 0)
line 14: information content of dataset 1 (only used if weighting scheme is set to 2)
line 15: information content of dataset 2 (only used if weighting scheme is set to 2)
```

## Examples

### cylinder model, single dataset
Data were simulated using [Shape2SAS](https://github.com/ehb54/GenApp-Shape2SAS) and is available in the example folder.    
To fit the simulated data of monodisperse cylinders (parameters: radius, length, scale and background), use input_cylinder.txt:     
```
output_fit_fylinder
1
examples/cylinder/Isim_1.dat
cylinder
25 10 0
110 20 0
1.0 0.5 0
1e-4 0.1
-5 5 20
yes
no
```
and run bayesapp using this inputfile:    
```
python bayesfit.py < examples/cylinder/input_cylinder.txt
```

### core-multishell model, two datasets 
Data is available in the example folder. It was simulated and fitted to investigate the best weigting scheme for simultaneous fitting of multiple SAXS or SANS datasets (Larsen, 2024).    
To fit the simulated data of a core-multishell particle, use input_coreshell.txt:
```
output_fit_coreshell
2
examples/coreshell/data/Isim1.dat
examples/coreshell/data/Isim2.dat
coreshell4_ratio_2
10 5 0
30 10 0
50 15 0
70 20 0
2 0.2
3 0.3
4 0.4
-0.1 0.01
0.1 0.01
0.05 0.005
0.5 0.05 0
1e-4 1e-2
0.8 0.08 0
1e-4 1e-2
-2 10 20
yes
no
0
```
and run bayesapp:    
```
python bayesfit.py < examples/coreshell/input_coreshell.txt
```

### If you use BayesFit, please cite 
* Larsen, Arleth and Hansen (2018), J. Appl. Cryst. 51:  1151-1161 [https://doi.org/10.1107/S1600576718008956]
* Larsen (2024) ArXiv 2311.06408 [https://arxiv.org/abs/2311.06408]
 
### Release notes     

#### BayesFit verison 1
* this version was used in Larsen et al. 2018
* written in Fortran by Steen Hansen and Andreas Larsen
* this version is no longer maintained. Archived at the old repository: github.com/Niels-Bohr-Institute-XNS-StructBiophys/BayesFit   
* 3 form factors are available: core-shell, micelle, and nanodisc
* GUI available at https://somo.chem.utk.edu/bayesfit/ (but not maintained)    
  
#### BayesFit version 2 major updates:       
* written in Python3 by Andreas Larsen.     
* improved documentation (at GitHub and in the source code).    
* simultaneous fitting of multiple datasets.
* more than 20 models available
* no GUI available, the strategy is instead to include functionality in larger packages for analysis of SAXS/SANS data, with GUIs, such as [SasView](https://sasview.org) or [WillItFit](https://sourceforge.net/projects/willitfit/).   

##### version 2.7
* reorganization of helpfunctions
* addition of examples

##### version 2.8
* addition of models
* clean-up of code
* less and more relevant output



