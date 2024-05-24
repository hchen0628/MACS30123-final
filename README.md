# MACS30123-final
# Large Scale Computing Project: Personal Finance Concerns Analysis

This is the GitHub repository for the final project of MACS 30123 Large Scale Computing.

Author: JIngzhi XU Huanrui CHEN

# Social Science Questions
Market research on the labor force focuses on analyzing workers' behavioral responses to industry changes. These studies typically use empirical data to analyze the impact of industry adjustments on workers' welfare, thereby predicting the long-term effects of economic policies. For example, researchers might explore the impact of manufacturing automation on employment rates and wage structures. In such cases, researchers usually need to use real datasets and counterfactual datasets to perform causal inference analysis through comparative methods to conclude the effects of policy changes. Ultimately, the insights provided by these studies are used to formulate more effective economic policies and labor market interventions.

However, a potential bottleneck encountered in previous research is the lack of large-scale counterfactual datasets. This is because experimental design or empirical analysis makes it difficult to naturally generate such datasets without significant external influences. The lack of counterfactual datasets can lead to analysis results that do not accurately reflect the real effects of policy changes, thereby affecting the reliability and precision of policy recommendations. Therefore, generating counterfactual simulation datasets for large-scale datasets to more accurately assess the potential impacts of policy changes, helping policymakers make more informed decisions, is a problem worth exploring.

This project provides a feasible approach to using large-scale computing methods to run the Wasserstein Generative Adversarial Networks (WGAN) framework to generate high-quality counterfactual simulation data. This helps economists and policymakers better understand and predict the complex impacts of labor market policies. In this project, we use the Integrated Public Use Microdata Series (IPUMS) Current Population Survey dataset as the real dataset for simulating counterfactual data. The advantages of this dataset include comprehensive demographic variables and consistent survey methods over time, covering the period from 1962 to the present, with over 55 million samples (1.6 million of which will be used in this project), making it a high-quality dataset for this project's purposes.

Our data procession includes preparing a sample with consistent individuals from 1996 to 2018 for estimation, and a WGAN generated sample following Athey’s method for simulation. The original dataset is denoted as i-sample, from which observations unique in demographic variables over the years are drawn and stored as single period X sample, where assume the same group of individuals exist for all periods. The X sample is created by duplicating the single period one 22 times and merging with counterfactual wage and transition probability, estimated based on i-sample. The X sample is latter used in utility estimation and analysis. The goal is to create a counterfactual samples of i-sample, that means the generated sample should be able to matched with X-sample. We follows Athey’s method to generate 1000000 observations, 618977 of which can be matched with X-sample, and thus we can calculate counterfactual simulated data by averaging over the correspondents’ cost over 1995 ∼ 2018 in X sample.


# Seriel Computation Bottlenecks

- Due to the large volume of collected data (in the millions), data cleaning and processing without parallelized procedures can lead to very long computation times. This means that single-threaded processing for large-scale data analysis might take weeks or even months to complete, significantly delaying project progress and the formulation of policy recommendations. To address this issue, we plan to use the Dask framework, which can achieve parallel data processing through multi-core processing and delayed execution optimization, thereby greatly reducing processing time and enhancing the efficiency and feasibility of data handling.
- The WGAN model requires substantial computational resources and memory, relying on GPUs and specialized parallel processing techniques to perform complex mathematical calculations. Performing such operations in a local computing environment can be resource-intensive and time-consuming. Particularly, we utilize Torch to optimize memory usage and accelerate the training process during model training, which implies higher demands on computing resources. Deploying this WGAN model on a cloud computing cluster can fully leverage computing resources and parallel processing capabilities, significantly improving the speed and efficiency of model training.

# Structure of Project
- Collect real-world data: [Microdata Series Current Population Survey dataset](https://cps.ipums.org/cps/)

- Using the Dask framework to clean millions of data points: []()

- Generate counterfactual data using real-world datasets
  - WGANs framwork for training and generating data from conditional and joint distributions for the simulation. (Author: Jonas Metzger and Evan Munro): [wgan.py]()
  - Training the WGAN model and generating counterfactual data with optimized hyperparameters and configurations: [WGAN_trial_v4_tune.py]()

- Data visualization: []()

# Data
Real world data be found at [](https://cps.ipums.org/cps/), the simulated dataset could be found at []().
