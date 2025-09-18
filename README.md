# PhySTNet
PhySTNet: Physics-Informed Neural Network for Spatiotemporal Modeling of Oceanic Multiphysical Fields

Accurate prediction of Marine multivariable fields, including sea surface temperature, sea surface salinity, velocity and wind stress, is crucial for understanding ocean-atmosphere interactions, enhancing climate predictability and ensuring the safety of Marine operations. However, the existing numerical models have high computational costs and are prone to error accumulation, while most deep learning methods lack physical consistency and are difficult to remain stable in the long term. Here, we propose PhySTNet, a physics-informed neural network that integrates multi-scale physical constraints and adaptive spatiotemporal memory. PhySTNet integrates a multi-dimensional feature perception module, a physically guided ocean prediction cell, and a dual-path memory mechanism with dynamic attention, achieving precise coupling between variables while maintaining physical laws. Make predictions based on HYCOM data for up to 7 days. The performance of physnet consistently outperforms 13 of the most advanced baselines, achieving outstanding directional accuracy in seawater velocity and wind stress prediction. Transfer experiments in distinct ocean basins further demonstrate strong generalization and stability. Our results establish PhySTNet as a robust and physically consistent approach for multi-physics ocean forecasting, with broad implications for climate science, autonomous navigation, and marine resource management.


Checkpoint: https://drive.google.com/drive/folders/1Y-8Zjb0_xTYfOVPwc0Qo8oRXkgtKN9u3?usp=drive_link

Test Data: https://drive.google.com/drive/folders/1ZZd4o9JOpWYvRGwEkDcX1xfV9DJLWJkt?usp=sharing
