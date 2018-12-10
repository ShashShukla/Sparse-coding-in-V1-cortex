# Sparse-coding-in-V1-cortex

This repository contains code for the course project for CS419-Introduction to Machine Learning, IIT Bombay, 2018.

The receptive fields of simple cells in V1 were learnt as an overcomplete basis with a sparsity constraint. The code for the choice of a Relevance Vector Machine as the sparisty constraint is contained in this repository as Julia code. Other methods were also attempted, such as K-SVD, MOD, Lasso, Ridge Regression, PCA and ICA, the details of which can be found in the report.
The learnt receptive fields indeed resembled the Gabor wavelets found experimentally. A bank of Gabor wavelets was also generated and fit to the learnt responses.

The dataset was downloaded from http://www.rctn.org/bruno/sparsenet/IMAGES.mat

To run the code, download the above the dataset to the code folder. Then run the file "learnSparseBasis.jl" to learn the receptive fields.
Run "generateGaborFilters.jl" to generate a bank of Gabor wavelets and then run "fitGabor.jl" to fit these Gabor filters to the receptive fields learnt before. Finally, you can run "computeSparsity.jl" to evaluate the sparsity (quantified using the L1 norm) of the learnt representations using the various methods.
