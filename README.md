**Mechanistic Interpretability of Arithmetic Reasoning in Transformers**

This repository investigates how arithmetic operators are represented inside transformer language models, using tools from mechanistic interpretability.
We focus on identifying, probing, and analyzing internal activation patterns associated with basic projective arithmetic (addition, subtraction, multiplication).

**Motivation**

Large language models exhibit strong arithmetic capabilities, yet the mechanisms underlying these behaviors remain poorly understood. Prior work has shown that models can perform arithmetic reliably, but it is unclear:
1. Where arithmetic information is represented
2. How operator identity is encoded
3. Whether such representations are causally involved in computation.
Arithmetic provides a controlled and interpretable testbed for studying abstraction, algorithmic behavior, and representation learning in neural networks.

**Research Questions**

This project addresses the following questions:
1.	Representational Question
Is arithmetic operator identity linearly decodable from intermediate transformer activations?
2.	Localization Question
At which layers do operator-related signals emerge?
3.	Mechanistic Question (ongoing)
Are these representations causally involved in arithmetic behavior, or merely correlational?

**Method Overview**

We study internal representations of a pretrained transformer using the following pipeline:
1.	Synthetic Arithmetic Data
   
  •  Templated arithmetic prompts with labeled operators
  
  •	 Controlled structure to minimize confounds
  
2.	Activation Collection
   
  •	Token-level residual stream activations
  
  • MLP post-activation hooks via TransformerLens
  
3.	Linear Probing
   
  •	Logistic regression probes trained to predict operator identity
  
  •	Evaluation on held-out data
  
4.	Analysis
   
  •	Train/test accuracy
  
  •	Probe coefficient norms
  
  •	Failure mode analysis
  
This pipeline is designed to be fully reproducible and extensible toward causal interventions.

**Experimental Setup**

•	Model: GPT-2 Small

•	Framework: TransformerLens

•	Layer Analyzed: MLP post-activation (default: layer 9)

•	Task: Operator classification (add / subtract / multiply)

•	Evaluation: Linear probe accuracy and generalization gap

**Results Summary**

Key findings from the current experiments:

•	Linear probes achieve high training accuracy (~88%), indicating that operator-related information is present in mid-layer activations.

•	Test accuracy is substantially lower (~22%), revealing poor generalization.

•	This gap suggests that operator information is entangled with superficial features rather than being robustly disentangled.

These results provide evidence of representational availability, but do not yet establish causal necessity.


**Limitations**

•	Probing results are correlational, not causal.

•	Token-level splitting can introduce data leakage.

•	Dataset size and diversity are limited.

•	Analysis is restricted to a single model and layer.

These limitations are discussed in detail in the accompanying documentation and motivate future work.

**Future Work**

Planned extensions include:

•	Example-level train/test splits

•	Causal interventions (activation ablation, patching)

•	Layer-wise analysis across the full model

•	Sparse Autoencoder (SAE) feature discovery

•	Scaling to more complex arithmetic and larger models

**Reproducibility**

**Installation:**
_pip install torch transformer-lens scikit-learn numpy_

**Run experiment:**
_python run_experiment.py_

Results are printed to stdout and can be extended to logging or visualization.

**Intended Audience**

This repository is intended for: 

•	Mechanistic interpretability researchers

•	ML researchers studying reasoning and abstraction

•	Students and practitioners interested in internal model analysis

________________________________________
**Disclaimer**
This project is an exploratory mechanistic study.
Findings should be interpreted as preliminary evidence, not definitive identification of arithmetic circuits.

