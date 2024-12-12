# MLFinalProjectUIC

## Overview
This repository contains the foundational resources necessary for running various machine learning models. It includes scripts, folders, and instructions to set up and preprocess datasets, train models, and evaluate their performance.

## Repository Structure
- **Preprocessing1**: Scripts for an earlier version of preprocessing.
- **Preprocessing2**: Scripts for an earlier version of preprocessing.
- **EnsembleModels**: Contains Jupyter notebooks for ensemble models.
- **RegressionModels**: Contains Jupyter notebooks for regression models.
- **NeuralNetwork**: Includes scripts to train and test neural network models and pre-trained model files.

## Initial Setup
Before running the models, follow these steps to set up the environment and generate the necessary datasets:

1. **Install Required Dependencies**
   Run the following command to install all required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Preprocessed Datasets**
   - **Small Dataset**: The small dataset and its preprocessed version are already included in the repository.
   - **Large Datasets**: Large datasets are not included due to size constraints. To download and create these datasets, execute:
     ```bash
     ./createDB.sh
     ```
     **Note:** This process may take between 30 minutes to 1 hour. After creating the large dataset, update all scripts to use it by modifying the dataset references accordingly.
   - **Final Preprocessed Files**: To create the final preprocessed files for both small and large datasets, run:
     ```bash
     python preprocessing.py
     ```

## Using the Models

### Ensemble Models
In the `EnsembleModels` folder, you will find Jupyter notebooks dedicated to training and evaluating ensemble models. Open these notebooks in a Jupyter environment to run them.

### Regression Models
In the `RegressionModels` folder, you will find Jupyter notebooks for regression models. These notebooks provide training and evaluation steps for various regression techniques.

### Neural Network Models

1. **Train the Neural Network**
   Navigate to the `NeuralNetwork` folder and run the following script to train the model on the small dataset:
   ```bash
   python NN.py
   ```

2. **Test the Neural Network**
   Run the following script to test the trained model:
   ```bash
   python TestNN.py
   ```
   **Important:** Ensure that you run the scripts from within the `NeuralNetwork` folder. Running them from any other location will result in errors.

3. **Pre-Trained Model**
   A pre-trained neural network model is available in the `NeuralNetwork` folder for direct usage.

## Notes
- Ensure you follow the setup steps in order to avoid missing files or dependencies.
- The preprocessing and dataset generation scripts require sufficient disk space and time for completion.
- To use the large dataset, ensure all scripts referencing the dataset are updated accordingly.

## License
The used datasets can be found at:
- https://www.kaggle.com/datasets/dhruvildave/new-york-city-taxi-trips-2019
- https://www.kaggle.com/datasets/alejopaullier/new-york-city-weather-data-2019
- https://www.kaggle.com/datasets/donnetew/us-holiday-dates-2004-2021


## Contact
For issues or queries, please open an issue on the GitHub repository or contact the maintainer directly.

