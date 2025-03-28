bash run.sh train RotatE FB13 3 1 1024 256 1000 24.0 1.0 0.0001 150000 16 -de
bash run.sh train pRotatE FB13 3 0 1024 256 500 500.0 1.0 0.001 150000 16 -de -dr -r 0.000002
bash run.sh train ComplEx FB13 0 0 1024 256 1000 500.0 1.0 0.001 150000 16 -de -dr -r 0.000002
bash run.sh train DistMult FB13 5 0 512 256 500 500.0 1.0 0.001 150000 16 -de -dr -r 0.000002

# these configs can be better by adjusting the hyperparameters