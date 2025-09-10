# Bayesian Inference for Nested Graphical Models (C++ Implementation)

**Politecnico di Milano – Bayesian Statistics Project 2023**  
**Authors:** Tancredi Ferrari, Luca Trizio, Fabio Zheng  
**Tutors:** Federico Castelletti, Francesco Denti  
**Professor:** Alessandra Guglielmi  
**Date:** February 14, 2024

---

## 📌 Project Description
This repository contains the **C++ implementation** of selected components of the Bayesian inference project on nested graphical models.  
While the main statistical framework was initially developed in **R**, this part of the work focuses on translating key routines into **C++** to leverage better **computational efficiency** and scalability.  

The C++ implementation aims to:  
- Accelerate **sampling procedures** (e.g., Gibbs Sampler) for high-dimensional data.  
- Improve handling of **large simulated and real datasets**.  
- Provide a foundation for integration into broader software pipelines beyond R.  

---

## ⚙️ Features Implemented
- Translation of R functions for **Bayesian nonparametric nested mixture models** into C++.  
- Efficient routines for:  
  - **Sampling from full conditionals** (π, ω, S, M, θ).  
  - Updating cluster allocations.  
  - Managing multivariate data structures.  
- Modular design to allow integration of further algorithms.  

---

## 📊 Results

On simulated data, the C++ sampler correctly recovered both distributional and observational clusters, validating the translation.

On real data (protein expression in leukemia patients), the implementation scaled efficiently, handling dozens of clusters with reduced runtime compared to R.

---

## 🚀 Future Work

Extend translation of remaining R functions into C++.

Incorporate parallelization (OpenMP / CUDA) to further improve runtime.

Package the C++ routines as a library to interface with R or Python.

---

## 🧑‍💻 Authors

Tancredi Ferrari

Luca Trizio

Fabio Zheng
