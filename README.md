# fraud_detection
Fraud detection ML usecase

## Setup
1. Clone the repository.
2. Navigate to the project directory.
3. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
4. Activate the virtual environment:
5. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
6. Ensure you have a `.env` file in the root directory with the following content:
   ```
    DATA_FILE_PATH=relative/path/to/your/data.csv
    ```
7. Place your data CSV file in the `data` folder.
8. Install project as editable:
   ```bash
   pip install -e .
   ```


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
