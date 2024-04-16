# BayesFit
version 2.6

### Description 
BayesFit allow users to fit analytical models (form factors and structure factors) to SAXS and/or SANS data using Bayesian refinement.

## Installation
Download the scripts from this repository and run BayesFit as a standard Python3 script.  
The following python scripts should be in the same folder as bayesfit.py:    
* bayesfit_functions.py
* function.py
* formfactors.py

## Running the program
```
python bayesfit.py
```
the program will prompt you for options (pressing enter gives you the default option).       
Alternatively, the input can be given via an inputfile (a plain text file): 
```
python bayesfit.py < inputfile.txt
```
where each line in the inputfile corresponds to the options that are prompted for (default value are used if line is left blank). 
for model with 3 parameters, fitted against a single dataset, the inputfile has the form:    
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
line 11: leave blank
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
data were simulated using Shape2SAS and is available in the example folder.    
to fit the simulated data of monodisperse cylinders (parameters: radius, length, scale and background) use input_cylinder.txt:     
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
python bayesfit.py < examples/input_cylinder.txt
```

### core-multishell model, two datasets 
data were simulated as described in Larsen2024 and is available in the example folder.    
to fith the simulated data of a core-multishell particle (parameters described in Larsen2024), use input_coreshell.txt:
```
output_fit_coreshell
2
examples/coreshell/Isim1.dat
examples/coreshell/Isim2.dat
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
1e-5 1e-2
0.8 0.08
1e-4 1e-2
-1.5 2 12
yes
no
0
```
and run bayesapp using this inputfile:    
```
python bayesfit.py < examples/input_coreshell.txt
```

### If you use BayesFit, please cite 
please cite:
* Larsen, Arleth and Hansen 2018 (BayesFit version 1):     
  * https://doi.org/10.1107/S1600576718008956
* Larsen 2024 (BayesFit version 2):    
  * https://arxiv.org/abs/2311.06408
 
### Release notes     

#### BayesFit version 2 major updates:       
* written in Python3 instead of Fortran     
* documented (at GitHub and in the source code)    
* allow for simultaneous fitting of multiple datasets

##### version 2.7



