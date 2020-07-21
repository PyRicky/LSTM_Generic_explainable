# LSTM_Generic_explainable

This is the code associated to the paper:

**Explainable Predictive Process Monitoring**, Riccardo Galanti, Bernat Coma Puig, Massimiliano de Leoni, Josep Carmona, Nicolò Navarin. In ICPM 2020, to appear.

## Code Example

Command examples to run the code:
1) Build you neural network (Offline phase)

python init_local.py --mandatory data/bac.csv 0 2 "%Y-%m-%d %H:%M:%S" remaining_time bac_remaining_time train --end_date_position 3

python init_local.py --mandatory data/bac.csv 0 2 "%Y-%m-%d %H:%M:%S" ACTIVITY bac_activity train --end_date_position 3 --pred_attributes "Back-Office Adjustment Requested,Authorization Requested,Pending Request for acquittance of heirs"

python init_local.py --mandatory data/bac.csv 0 2 "%Y-%m-%d %H:%M:%S" case_cost bac_cost train --end_date_position 3

2) Test the trained neural network on running cases (Online phase)

python init_local.py --mandatory data/running/bac_running.csv 0 2 "%Y-%m-%d %H:%M:%S" remaining_time bac_remaining_time predict --end_date_position 3

python init_local.py --mandatory data/running/bac_running.csv 0 2 "%Y-%m-%d %H:%M:%S" ACTIVITY bac_activity predict --end_date_position 3 --pred_attributes "Back-Office Adjustment Requested,Authorization Requested,Pending Request for acquittance of heirs"

where there are 7 mandatory parameters: file, case_id_position, start_date_position, timestamp, column_to_be_predicted, experiment_name, mode

column_to_be_predicted --> (remaining_time or ACTIVITY for example)

experiment_name --> choose a name for the experiment (in the online phase this name must be the same)

There are also up to 4 optional parameters: 
--shap (default True) --> if True calculate also the shapley values for explainability (Offline phase)

--end_date_position --> if the file has also end_date column for every activity

--pred_attributes --> mandatory when predicting a categorical attribute (example specify to predict if one or more activities will be performed or not)

--num_epochs (default 500) --> set a maximum limit for training the model (Offline phase)


**IMPORTANT NOTE**: Please find attached in the repository the results obtained in the several datasets and described in the paper and in the appendix.
                    If you want to replicate the experiments (Offline phase) you should delete the relative experiment folder in the experiment_files folder and retrain the model.
					Shapley values need between half a day and a day to be calculated and have not been uploaded since they are very heavy.
				    NOTE: the training of LSTM may be very slow on CPUs, so we suggest to run this code on GPUs instead (and in a Unix environment).


**TRAIN OTHER EXPERIMENTS**

<b> Bpic 2012</b>

python init_local.py --mandatory data/bpi12_complete.csv 0 2 "%Y/%m/%d %H:%M:%S" remaining_time bpi12_complete_remaining_time train --end_date_position 3

python init_local.py --mandatory data/bpi12_complete.csv 0 2 "%Y/%m/%d %H:%M:%S" event bpi12_complete_activity train --end_date_position 3 --pred_attributes "A_ACCEPTED,A_CANCELLED,A_DECLINED"

<b> Bpic 2012-W</b>

python init_local.py --mandatory data/BPI12_anonimyzed_less_activities.csv 0 2 "%Y-%m-%d %H:%M:%S" remaining_time bpi12_less_activities_remaining_time train

<b> Bpic 2013</b>

python init_local.py --mandatory "data/VINST cases incidents.csv" 0 1 "%Y-%m-%d %H:%M:%S%z" remaining_time bpi13_remaining_time train

python init_local.py --mandatory "data/VINST_cases_incidents_2_3_line.csv" 0 1 "%Y-%m-%d %H:%M:%S%z" "Involved ST" bpi13_push_to_front train --pred_attributes "2nd_3rd line"

python init_local.py --mandatory "data/VINST cases incidents.csv" 0 1 "%Y-%m-%d %H:%M:%S%z" "Sub Status" bpi13_wait_user train --pred_attributes "Wait - User"

<b> HelpDesk 2017</b>

python init_local.py --mandatory data/Helpdesk2017_anonimyzed.csv 0 3 "%Y/%m/%d %H:%M:%S" remaining_time HelpDesk_remaining_time train

## Installation
The code requires python 3.7+ and the following libraries:

pandas

keras==2.3.1

tensorflow-gpu==1.15

shap==0.31

seaborn

## Contributors

Riccardo Galanti

## License

If you use this code for research, please cite our paper.
This code is released under the Apache licence 2.0.