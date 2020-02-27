# LSTM_Generic_explainable

This is the code associated to the paper:

**On the Explanations of Business Process Predictions of Generic KPIs**, Riccardo Galanti, Bernat Coma Puig, Josep Carmona, Massimiliano de Leoni, Nicolò Navarin. In BPM 2020: 18th International Conference on Business Process Management, to appear.

The preprint of the paper is available in [arXiv]

## Code Example

Show what the library does as concisely as possible, developers should be able to figure out **how** your project solves their problem by looking at the code example. Make sure the API you are showing off is obvious, and that your code is short and concise.

Command examples to run the code:
1) Build you neural network

python LSTM_sequence_mae.py --mandatory data/bac_1_9_1_anonimyzed_less_rows.csv 0 2 "%Y-%m-%d %H:%M:%S" remaining_time --end_date_position 3 --shap=True

python LSTM_sequence_mae.py --mandatory data/bac_1_9_1_anonimyzed_less_rows.csv 0 2 "%Y-%m-%d %H:%M:%S" ACTIVITY --end_date_position 3 --shap=True

2) Test the trained neural network on running cases

python LSTM_sequence_mae.py --mandatory data/bac_1_9_1_anonimyzed_less_rows_running.csv 0 2 "%Y-%m-%d %H:%M:%S" remaining_time --end_date_position 3 --model model/model_bac_1_9_1_less_rows_remaining_time_100_8.json

python LSTM_sequence_mae.py --mandatory data/bac_1_9_1_anonimyzed_less_rows_running.csv 0 2 "%Y-%m-%d %H:%M:%S" ACTIVITY --end_date_position 3 --model model/model_bac_1_9_1_less_rows_ACTIVITY_100_8.json --pred_attribute "ACTIVITY 11"

where there are 5 mandatory parameters: csv, case_id_position, start_date_position, timestamp, column_to_be_predicted (remaining_time or ACTIVITY as an example)

and up to 4 optional parameters: 
--shap (default False) --> if you want to calculate also the shapley values for explainability (train phase)

--end_date_position --> if the csv has also end_date for every activity

--model --> when you want to predicted real running cases you need the trained json model

--pred_attribute --> when you predict a categorical attribute for real running cases (example if a certain activity will be performed or not)


NOTE: the training of LSTM may be very slow on CPUs, so we suggest to run this code on GPUs instead (and in a Unix environment).
NOTE: Also the calculation of the shapley values in the train phase could be very slow, so we suggest to train only the model to replicate the results and then obtain explanations only for running cases

## Installation
The code requires python 3.6+ and the following libraries:

pandas

keras

tensorflow-gpu==1.15

shap==0.31

matplotlib

## Contributors

Riccardo Galanti

## License

If you use this code for research, please cite our paper.
This code is released under the Apache licence 2.0.