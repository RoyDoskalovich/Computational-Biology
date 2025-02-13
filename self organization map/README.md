# Self-Organizing Map (SOM) for Handwritten Digit Clustering

## Overview
This project implements a **Self-Organizing Map (SOM)** to cluster handwritten digit images. A SOM is an unsupervised neural network that maps high-dimensional data into a lower-dimensional space while preserving the topological structure of the input data. In this case, we use a **10x10 hexagonal SOM** to group handwritten digits and analyze the learned feature space.

## Algorithm Description
### 1. **Initialization**
- The SOM grid is a **10x10 hexagonal lattice**, where each neuron has a weight vector of **784 dimensions** (corresponding to a **28x28 grayscale image**).
- Weights are initialized with small random values around the mean of the input dataset.

### 2. **Training Process**
- Each training iteration involves selecting an input digit and finding its **Best Matching Unit (BMU)**—the neuron with the closest weight vector.
- The BMU’s weight vector and its neighboring neurons' weights are updated using an **exponentially decaying learning rate and neighborhood function**.
- Neighborhood influence follows a **hexagonal distance metric**, where neurons closer to the BMU receive a greater update than those farther away.
- The training runs for **20 epochs**, or until a timeout of **3 minutes** is reached.

### 3. **Mapping and Evaluation**
- Each digit is mapped to its **BMU and Second BMU (SBMU)**.
- The quality of the trained SOM is measured using two metrics:
  - **Quantization Error:** The average Euclidean distance between input vectors and their BMUs.
  - **Topographic Error:** Measures whether the BMU and SBMU are adjacent in the grid.
- The dominant digit for each neuron is determined, and a heatmap is generated to visualize the learned clusters.
- The final weight vectors are reshaped into **28x28 images** to inspect how well the neurons represent handwritten digits.

## Results
- The trained SOM successfully groups similar digits together while preserving the structure of digit variations.
- Lower **quantization and topographic errors** indicate a well-formed mapping.
- Visualization of neuron weight vectors shows that the SOM effectively learns representations of handwritten digits.
- Follow this link [self organization map/results](https://github.com/RoyDoskalovich/Computational-Biology/tree/568531a40b530c6cc9a13a554a83bb17ff562a14/self%20organization%20map/results) to see the reuslts.


## Usage
To run the simulation:
1. Ensure Python 3.x with numpy, pandas and matplotlib is installed.
2. Ensure the datasets (`digits_test.csv` and `digits_keys.csv`) are in the same directory.
3. Download the GA.exe file from this [drive](https://drive.google.com/drive/folders/1YV0cZw733MaLh9Rk8jVq1_u4qel_gbZU?usp=sharing).
4. Run the SOM.exe.

  The program will output:
   - A visualization of dominant digit clusters.
   - A display of the learned neuron weight vectors.
   - Metrics evaluating the quality of the SOM.

