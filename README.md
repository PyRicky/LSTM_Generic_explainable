# LSTM_Generic_explainable

This is the code associated to the paper:

**Explainable Predictive Process Monitoring**, Riccardo Galanti, Bernat Coma Puig, Josep Carmona, Massimiliano de Leoni, Nicolò Navarin. In BPM 2020: 18th International Conference on Business Process Management, to appear.

## Code Example

Show what the library does as concisely as possible, developers should be able to figure out **how** your project solves their problem by looking at the code example. Make sure the API you are showing off is obvious, and that your code is short and concise.

Command examples to run the code:
1) Build you neural network

python LSTM_sequence_mae.py --mandatory data/bac_1_9_1_anonimyzed_less_rows.csv 0 2 "%Y-%m-%d %H:%M:%S" remaining_time --end_date_position 3 --shap True

python LSTM_sequence_mae.py --mandatory data/bac_1_9_1_anonimyzed_less_rows.csv 0 2 "%Y-%m-%d %H:%M:%S" ACTIVITY --end_date_position 3 --shap True --shap_attributes "ACTIVITY 11,ACTIVITY 3,ACTIVITY 7"

2) Test the trained neural network on running cases

python LSTM_sequence_mae.py --mandatory data/bac_running.csv 0 2 "%Y-%m-%d %H:%M:%S" remaining_time --end_date_position 3 --model model/model_bac_remaining_time_100_8.json

python LSTM_sequence_mae.py --mandatory data/bac_running.csv 0 2 "%Y-%m-%d %H:%M:%S" ACTIVITY --end_date_position 3 --model model/model_bac_ACTIVITY_100_8.json --pred_attribute "Back-Office Adjustment Requested"

where there are 5 mandatory parameters: csv, case_id_position, start_date_position, timestamp, column_to_be_predicted (remaining_time or ACTIVITY as an example)

and up to 4 optional parameters: 
--shap (default False) --> if you want to calculate also the shapley values for explainability (train phase)

--shap_attributes (default False) --> for categorical attribute decide the histograms you want to show

--end_date_position --> if the csv has also end_date column for every activity

--model --> when you want to predict real running cases you need the neural network trained json model

--pred_attribute --> when you predict a categorical attribute for real running cases (example if a certain activity will be performed or not)


NOTE: the training of LSTM may be very slow on CPUs, so we suggest to run this code on GPUs instead (and in a Unix environment).

NOTE: The calculation of the shapley values in the train phase could be very slow even on GPUs, so we suggest to train only the model and then obtain explanations only for running cases

## Installation
The code requires python 3.6+ and the following libraries:

pandas

keras

tensorflow-gpu==1.15

shap==0.31

matplotlib

## Contributors

Riccardo Galanti
Bernat Coma Puig

## License

If you use this code for research, please cite our paper.
This code is released under the Apache licence 2.0.