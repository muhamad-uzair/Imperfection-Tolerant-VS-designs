# Imperfection-Tolerant-VS-designs

This repository provides a complete Python workflow for modeling, surrogate training, and optimization of imperfection-sensitive composite cylinders. It integrates digital imperfection data, PCA-based compression, surrogate modeling with Gaussian Processes, and multi-objective optimization via Genetic Algorithms (NSGA-II) and Bayesian Optimization (BoTorch/Ax).

# Repository Structure

PCA.py performs Principal Component Analysis (PCA) on measured imperfection fields and generates synthetic samples via Latin Hypercube Sampling (LHS).
Injection.py maps synthetic imperfection fields onto ABAQUS ".inp" models and updates winding angles for nonlinear and linear buckling simulations.
GPR Training.py trains Gaussian Process Regression (GPR) surrogates for mass, nonlinear buckling load (Pcr), and perfect buckling load (RL_perfect) with Kriging pretraining.
NSGA-II.py runs multi-objective optimization using NSGA-II to minimize mass while maximizing load-carrying capacity and KDF (knockdown factor).
BO.py implements Bayesian Optimization (qNEHVI, Ax/BoTorch) for efficient multi-objective search with uncertainty-aware surrogates.
VAFW8.inp / VAFW8RL.inp are the baseline ABAQUS input files for nonlinear (imperfect) and linear (perfect) finite element models.

# Key Features

- Imperfection Modeling: Real and synthetic geometric imperfections using PCA and LHS.
- Automated FE Input Generation: Programmatic modification of ABAQUS ".inp" files for batch simulations.
- Surrogate Modeling: Gaussian Process Regression with uncertainty quantification.
- Multi-objective Optimization: Both evolutionary (NSGA-II) and Bayesian (qNEHVI) algorithms.
- Sensitivity & Reliability Analysis: Automatic computation of knockdown factors and angle sensitivities.

# Workflow Overview

1. Imperfection Processing:
   PCA.py extracts dominant imperfection modes and Generate synthetic samples.
2. FE Model Preparation: 
   Injection.py applies imperfections and winding angles to ".inp" files.
3. Surrogate Training:
   GPR Training.py trains models on simulation data.
4. Optimization:  
   - NSGA-II.py uses NSGA-II multi-objective search  
   - BO.py uses Bayesian multi-objective optimization (qNEHVI)
5. Post-Processing:  
   Pareto fronts, convergence plots, and sensitivity charts are automatically generated.

# Dependencies
Install all dependencies via:
pip install numpy pandas matplotlib scikit-learn pyDOE deap torch botorch ax-platform joblib
