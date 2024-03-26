# BayesFit
version 2.5

## Description 
BayesFit allow users to fit analytical models (form factors and structure factors) to SAXS and/or SANS data using Bayesian refinement.

## Installation
Download the scripts from this repository and run BayesFit as a standard python3 script: 

## Running the program
```
python bayesfit.py
```
the program will prompt you for options (pressing enter gives you the default option).       
Alternatively, the input can be given via an inputfile (a plain text file): 
```
python bayesfit.py < inputfile.txt
```
where each line in the inputfile corresponds to the options that are prompted for (default value are used if line is left blank):
```
line 1 : output directory (default: output_bayesfit)    
line 2 : number of contrasts (default: 1)    
line 3 : datafile (default: examples/cylinder/Isim_1.dat)    
line 4 : model (default: cylinder)    
line 5 : prior values, mean sigma (optional) min (optional) max (optional) for the first parameters    
line 6: prior values for the second parameter    
...    
line K+4: prior values for the K'th parameter    
line K+5: logalpha values (default: -5 5 15)    
line K+6: fit data (default: yes)
line K+7: fit posterior distributions (default: no)
line K+8: weight scheme (default: 0, i.e. weight by chi2r)    
```
## Examples

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
0
```
and run bayesapp using this inputfile:    
```
python bayesfit.py < input_cylinder.txt
```



### Cite 
please cite:
* Larsen, Arleth and Hansen 2018:     
  * https://doi.org/10.1107/S1600576718008956
* Larsen 2024:    
  * https://arxiv.org/abs/2311.06408





