# fraud_detection
Fraud detection ML usecase


For testing and running repositoy follow the steps below:

### Running analysis and baseline notebook

1. Open `analysis.ipynb`. 
2. Place data csv file in the data folder. 
3. Check that `.env` file is present in the root directory.
4. Add relative path to data file in the `.env` file under `DATA_FILE_PATH`.



### Running model training and evaluation scripts.
There are two scripts for training and evaluating models:
1. A neural network model in `src/fraud_detector/nn.py`.
2. A sklearn-like pipeline in `src/fraud_detector/sklearn_pipeline.py`.


1. #### Sklearn-like API pipeline.
