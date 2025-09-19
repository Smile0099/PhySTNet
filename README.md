
# PhySTNet: Physics-Informed Neural Network for Spatiotemporal Modeling of Oceanic Multiphysical Fields

Accurate prediction of Marine multivariable fields, including sea surface temperature, sea surface salinity, velocity and wind stress, is crucial for understanding ocean-atmosphere interactions, enhancing climate predictability and ensuring the safety of Marine operations. However, the existing numerical models have high computational costs and are prone to error accumulation, while most deep learning methods lack physical consistency and are difficult to remain stable in the long term. Here, we propose PhySTNet, a physics-informed neural network that integrates multi-scale physical constraints and adaptive spatiotemporal memory. PhySTNet integrates a multi-dimensional feature perception module, a physically guided ocean prediction cell, and a dual-path memory mechanism with dynamic attention, achieving precise coupling between variables while maintaining physical laws. Make predictions based on HYCOM data for up to 7 days. The performance of physnet consistently outperforms 13 of the most advanced baselines, achieving outstanding directional accuracy in seawater velocity and wind stress prediction. Transfer experiments in distinct ocean basins further demonstrate strong generalization and stability. Our results establish PhySTNet as a robust and physically consistent approach for multi-physics ocean forecasting, with broad implications for climate science, autonomous navigation, and marine resource management.


# System Requirements

Hardware requirements: PhySTNet requires a standard computer with enough RAM to support in-memory operations and a high-performance GPU to support fast operations on high-dimensional data.

Software requirements:   
OS Requirements:

This package is supported for Windows and Linux. The package has been tested on the following systems:
Windows: Windows 10 22H2
Linux: Ubuntu 16.04

Python Dependencies:
PhySTNet mainly depends on the Python scientific stack.

einops==0.8.0
fbm==0.3.0
matplotlib==3.7.2
numpy==1.24.3
pandas==2.0.3
pmdarima==2.0.4
ptflops==0.7.3
pynvml==11.5.3
scikit_learn==1.5.1
scipy==1.10.1
seaborn==0.13.2
sympy==1.12
torch==2.3.1
torch_cluster==1.6.3
tqdm==4.66.4
tvm==1.0.0
xarray==2022.11.0

# Instructions to run on data

Due to the large size of the data and weight files, we host them on other data platforms, please download the relevant data from the link below.

Test data：https://drive.google.com/drive/folders/1ZZd4o9JOpWYvRGwEkDcX1xfV9DJLWJkt?usp=sharing
Checkpoint: https://drive.google.com/drive/folders/1Y-8Zjb0_xTYfOVPwc0Qo8oRXkgtKN9u3?usp=drive_link

