for batch_size in 16
    do
    for lr in 1e-05 2e-05 5e-05 1e-04 5e-06
        do
        for modelname in roberta-base
        do
            python3 scripts/training_scripts/train_model.py Collective Property Premise2Justification Premise1Conclusion pivot --modelname ${modelname} --lr ${lr} --batch_size ${batch_size}
        done
    done
done

for batch_size in 16
    do
    for lr in 1e-05 2e-05 5e-05 1e-04 5e-06
        do
        for modelname in roberta-base
        do
            python3 scripts/training_scripts/train_model.py Premise2Justification Premise1Conclusion --type_of_premise True --modelname ${modelname} --lr ${lr} --batch_size ${batch_size}
        done
    done
done

