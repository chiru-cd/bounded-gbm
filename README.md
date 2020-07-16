# bounded-gbm

Tool for determining the boundedness of a gradient boosting machine. Generates boundaries for a set of features in a model and filters out of bound records in test dataset.

## Tool Features

* Easy preprocessing of raw training data to be used in the tool.
* Builds XGBoost model from given training data alongwith parameter tuning.
* Generates boundary values for given features or top K features.
* Can use user defined boundary values for evaluation.
* Output modes for test data:
  * Result for each record indicating number of features it is unbound on.
  * Reason matrix indicating the margin by which it is bound or unbound.
  * Evaluation score improvement after modifying the given dataset.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and evaluation purposes.

1. run `git clone https://github.com/chiru-cd/bounded-gbm.git`
2. run `cd bounded-gbm`

### Prerequisites

* Dependencies used in this tool are `xgboost`, `numpy`, `pandas`, `sklearn`, `pickle` & `matplotlib`. Can be installed by using `pip3 install -r dependencies.txt`
* Make sure the datasets downloaded from [LendingClub](https://www.lendingclub.com/statistics/additional-statistics?) are saved in the `data` folder.
* For very large datasets, use `split.py` for splitting them into smaller datasets.

### For Model Generation

1. run `cd preprocessing`
2. Enter the input file names (from data folder) in `preprocessing.py` and run it. The processed data files will be saved in the project root folder.
3. Modify the tuning parameters in `parameter_tuning.py` and run it. The optimal parameters will be displayed in the console.
4. Enter processed data file and optimal parameters in `model.py` and run it. The built model is saved as `model.dat`.
> *.dat files to be used as model input for the tool*

## Running Tool

* Modify `configuration.ini` according to your requirements.
> **Do NOT change the DEFAULT section**
* Run `main.py` script to get result according to `eval_flags` set.