**Train**

For example, this command train a TransE model on FB13 dataset with GPU 0.

```
python -u codes/run.py --do_train  --cuda  --do_valid  --do_test  --data_path data/FB13  --model TransE  -n 256 -b 1024 -d 1000  -g 24.0 -a 1.0 -adv  -lr 0.0001 --max_steps 200000  -save models/TransE_FB13_0 --test_batch_size 4 -de
```

Check argparse configuration at codes/run.py for more arguments and more details.

**Test**

python -u codes/run.py --do_test --cuda --test_file $test_file -init $SAVE

```
python -u codes/run.py --do_test --cuda -init models/RotatE_FB13_1/
```

```
# generate train files for llms in stage 2
python -u codes/run.py --do_test --cuda --test_file "train.txt" -init $SAVE
```