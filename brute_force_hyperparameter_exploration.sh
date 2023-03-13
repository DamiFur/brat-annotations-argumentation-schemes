
for quarter in 1 2 3
do
	python3 scripts/training_scripts/train_model.py Collective --modelname roberta-base --lr 5e-06 --batch_size 16 --quarters_of_dataset ${quarter}
	python3 scripts/training_scripts/train_model.py Collective --modelname vinai/bertweet-base --lr 1e-05 --batch_size 16 --quarters_of_dataset ${quarter}
	python3 scripts/training_scripts/train_model.py Collective --modelname xlm-roberta-base --multilingual --lr 5e-06 --batch_size 16 --quarters_of_dataset ${quarter}
	python3 scripts/training_scripts/train_model.py Collective --modelname xlm-roberta-base --crosslingual --lr 2e-05 --batch_size 16 --quarters_of_dataset ${quarter}

	python3 scripts/training_scripts/train_model.py Property --modelname roberta-base --lr 5e-05 --batch_size 16 --quarters_of_dataset ${quarter}
	python3 scripts/training_scripts/train_model.py Property --modelname vinai/bertweet-base --lr 5e-05 --batch_size 16 --quarters_of_dataset ${quarter}
	python3 scripts/training_scripts/train_model.py Property --modelname xlm-roberta-base --multilingual --lr 1e-04 --batch_size 16 --quarters_of_dataset ${quarter}
	python3 scripts/training_scripts/train_model.py Property --modelname xlm-roberta-base --crosslingual --lr 1e-04 --batch_size 16 --quarters_of_dataset ${quarter}
	
	python3 scripts/training_scripts/train_model.py pivot --modelname roberta-base --lr 2e-05 --batch_size 16 --quarters_of_dataset ${quarter}
	python3 scripts/training_scripts/train_model.py pivot --modelname vinai/bertweet-base --lr 1e-05 --batch_size 16 --quarters_of_dataset ${quarter}
	python3 scripts/training_scripts/train_model.py pivot --modelname xlm-roberta-base --multilingual --lr 1e-05 --batch_size 16 --quarters_of_dataset ${quarter}
	python3 scripts/training_scripts/train_model.py pivot --modelname xlm-roberta-base --crosslingual --lr 1e-05 --batch_size 16 --quarters_of_dataset ${quarter}
	
	python3 scripts/training_scripts/train_model.py Premise1Conclusion --modelname roberta-base --lr 1e-04 --batch_size 16 --quarters_of_dataset ${quarter}
	python3 scripts/training_scripts/train_model.py Premise1Conclusion --modelname vinai/bertweet-base --lr 5e-05 --batch_size 16 --quarters_of_dataset ${quarter}
	python3 scripts/training_scripts/train_model.py Premise1Conclusion --modelname xlm-roberta-base --multilingual --lr 1e-04 --batch_size 16 --quarters_of_dataset ${quarter}
	python3 scripts/training_scripts/train_model.py Premise1Conclusion --modelname xlm-roberta-base --crosslingual --lr 1e-04 --batch_size 16 --quarters_of_dataset ${quarter}
	
	python3 scripts/training_scripts/train_model.py Premise2Justification --modelname roberta-base --lr 1e-04 --batch_size 16 --quarters_of_dataset ${quarter}
	python3 scripts/training_scripts/train_model.py Premise2Justification --modelname vinai/bertweet-base --lr 1e-04 --batch_size 16 --quarters_of_dataset ${quarter}
	python3 scripts/training_scripts/train_model.py Premise2Justification --modelname xlm-roberta-base --multilingual --lr 2e-05 --batch_size 16 --quarters_of_dataset ${quarter}
	python3 scripts/training_scripts/train_model.py Premise2Justification --modelname xlm-roberta-base --crosslingual --lr 1e-04 --batch_size 16 --quarters_of_dataset ${quarter}

	python3 scripts/training_scripts/train_model.py Premise1Conclusion --modelname roberta-base --type_of_premise True --lr 5e-05 --batch_size 16 --quarters_of_dataset ${quarter}
	python3 scripts/training_scripts/train_model.py Premise1Conclusion --modelname vinai/bertweet-base --type_of_premise True --lr 5e-05 --batch_size 16 --quarters_of_dataset ${quarter}
	python3 scripts/training_scripts/train_model.py Premise1Conclusion --modelname xlm-roberta-base --type_of_premise True --multilingual --lr 5e-05 --batch_size 16 --quarters_of_dataset ${quarter}
	python3 scripts/training_scripts/train_model.py Premise1Conclusion --modelname xlm-roberta-base --type_of_premise True --crosslingual --lr 5e-05 --batch_size 16 --quarters_of_dataset ${quarter}

	python3 scripts/training_scripts/train_model.py Premise2Justification --modelname roberta-base --type_of_premise True --lr 1e-05 --batch_size 16 --quarters_of_dataset ${quarter}
	python3 scripts/training_scripts/train_model.py Premise2Justification --modelname vinai/bertweet-base --type_of_premise True --lr 1e-05 --batch_size 16 --quarters_of_dataset ${quarter}
	python3 scripts/training_scripts/train_model.py Premise2Justification --modelname xlm-roberta-base --type_of_premise True --multilingual --lr 1e-05 --batch_size 16 --quarters_of_dataset ${quarter}
	python3 scripts/training_scripts/train_model.py Premise2Justification --modelname xlm-roberta-base --type_of_premise True --crosslingual --lr 1e-05 --batch_size 16 --quarters_of_dataset ${quarter}

	python3 scripts/training_scripts/train_model.py Argumentative --modelname roberta-base --lr 1e-04 --batch_size 16 --quarters_of_dataset ${quarter}
	python3 scripts/training_scripts/train_model.py Argumentative --modelname vinai/bertweet-base --lr 2e-05 --batch_size 16 --quarters_of_dataset ${quarter}
	python3 scripts/training_scripts/train_model.py Argumentative --modelname xlm-roberta-base --multilingual --lr 1e-05 --batch_size 16 --quarters_of_dataset ${quarter}
	python3 scripts/training_scripts/train_model.py Argumentative --modelname xlm-roberta-base --crosslingual --lr 5e-05 --batch_size 16 --quarters_of_dataset ${quarter}


done

