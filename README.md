# LSTM-PINN-IDM-Car-Following-Model

A physics-informed Long Short-Term Memory (LSTM) neural network for vehicle car-following acceleration prediction, incorporating the Intelligent Driver Model (IDM) as a physical constraint.

## Purpose

Traditional LSTM networks can learn complex relationships in vehicle trajectory data, making them useful for predicting car-following behaviour. However, purely data-driven models may struggle to generalize to unfamiliar driving conditions or produce predictions that are inconsistent with established traffic-flow models.

This project explores whether incorporating the Intelligent Driver Model (IDM) into an LSTM's training process can improve acceleration predictions.

Rather than relying entirely on observed vehicle data, the physics-informed model introduces an additional loss term that penalizes deviations from IDM-predicted acceleration. This encourages the network to learn from both measured driving behaviour and established car-following dynamics.

The project compares two approaches:

- **PUNN (Physics-Uninformed Neural Network):** A standard LSTM trained using observed vehicle acceleration data.
- **PIDL-CF (Physics-Informed Deep Learning for Car-Following):** An LSTM trained using a combination of observed acceleration data and an IDM-based physics loss.

## Model Architecture

Both models use the same underlying architecture:

- Two-layer LSTM with 64 hidden units
- Fully connected layer with 32 neurons and ReLU activation
- Dropout rate of 0.2
- Single output predicting follower acceleration
- Sequence length of 10 observations
- Adam optimizer with a learning rate of 0.001

### Input Features

1. Spacing between the following and leading vehicles
2. Relative speed (follower speed − leader speed)
3. Following vehicle speed

**Prediction target:** Future following-vehicle acceleration (m/s²).


## Physics-Informed Training

The PIDL-CF model incorporates the IDM acceleration equation:

$$
a_{\mathrm{IDM}} = a_{\max}\left[1-\left(\frac{v}{v_0}\right)^4-\left(\frac{s^*}{s}\right)^2\right]
$$

where $s^*$ represents the desired dynamic spacing.

The total training loss combines the prediction error and physics-based error:

$$
L_{\mathrm{total}} = 0.7L_{\mathrm{data}} + 0.3L_{\mathrm{physics}}
$$

This approach regularizes the LSTM during training rather than directly modifying its predictions afterward.

## Dependencies

Developed using Python and PyTorch, with optional CUDA acceleration for compatible NVIDIA GPUs.

- pandas
- numpy
- scikit-learn
- matplotlib
- torch
- scipy
- tqdm

The Python standard-library modules `math` and `warnings` are also used.

**Runtime:** Approximately 20 minutes on an NVIDIA RTX 3050 (observed during development; runtime may vary).

## Dataset: NGSIM Vehicle Trajectory Data

Source: [Next Generation Simulation (NGSIM) Vehicle Trajectories and Supporting Data](https://catalog.data.gov/dataset/next-generation-simulation-ngsim-vehicle-trajectories-and-supporting-data)

This project uses the reconstructed vehicle trajectories from Interstate 80 (I-80), Emeryville, California, recorded between 4:00 PM and 4:15 PM.

### Download Instructions

1. Open the NGSIM dataset page linked above.
2. Locate the `I-80-Emeryville-CA.zip` attachment.
3. Navigate to `i-80-vehicle-trajectory-data/vehicle-trajectory-data/0400pm-0415pm/`.
4. Select `RECONSTRUCTED trajectories-400-0415_NO MOTORCYCLES.csv`.
5. Preprocess the dataset into `processed_ngsim.csv` with the required leader-follower variables before executing the model.

Other recording intervals are available, although this implementation was developed using the 4:00–4:15 PM interval.

## Evaluation

The implementation compares acceleration prediction RMSE across five selected follower vehicles, excluding each selected vehicle from the neural-network training set.

RMSE is reported in m/s² for both the standard LSTM and physics-informed LSTM.

**Implementation note:** The original experimental code calibrates IDM parameters using the held-out evaluation vehicle. This must be changed to training-only calibration before the results can be considered an independent held-out evaluation. Sequence continuity and acceleration preprocessing should also be validated.
