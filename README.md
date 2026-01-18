# Causal–Argumentation Explainability Framework

This repository contains the source code and experimental workflow for the paper  
**“A Causal–Argumentation Method for Explainability of Machine Learning Models”**  
by Henry Salgado, Meagan Kendall, and Martine Ceberio (The University of Texas at El Paso).

The framework integrates **causal discovery** and **argumentation theory** to produce structured, causally grounded explanations for machine learning models. It combines **constraint-based causal inference** (via the FCI or PC algorithms) with **Bipolar Argumentation Frameworks (BAFs)** and **probabilistic reasoning under semi-stable semantics**.

---

## Overview

Modern explainable AI (XAI) methods—such as SHAP or LIME—highlight feature importance but often fail to reveal **why** and **how** features interact to influence predictions.  
This framework bridges that gap by:

1. Discovering **causal dependencies** among features using constraint-based methods (e.g., FCI, PC).  
2. Translating these relationships into a **Bipolar Argumentation Framework (BAF)**, where positive correlations define *support* and negative correlations define *attack* relations.  
3. Computing **semi-stable extensions** through a probabilistic constellation approach to identify consistent explanatory pathways that reflect real data structure.

The result is a **causally interpretable map** of feature interactions that explains both *what supports* and *what opposes* an outcome.

---

## Key Features

- **Dual-run causal discovery:** Automatically handles singular matrix issues by running both drop-first and drop-last encodings, merging results via majority voting.  
- **Entropy-based binning:** Discretizes continuous variables using supervised entropy gain to capture interpretable feature intervals.  
- **Causal discovery engine:** Supports FCI or PC algorithms with configurable independence tests (e.g., Fisher’s z-test, G², kernel CI).  
- **Argumentation layer:** Converts causal edges into a BAF and then into a Classical Argumentation Framework (AF) for reasoning.  
- **Probabilistic reasoning:** Semi-stable semantics combined with the constellation approach yields acceptance probabilities for each argument.  
- **Visualization tools:** Exports causal and argumentative graphs with weighted correlations and produces acceptance probability plots.

---

## Installation

Clone this repository and install dependencies using Conda or pip.

