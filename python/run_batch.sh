for k in {1..1}
do 
    echo Batch $k
    #python batch_learn_mfcc.py --past 1 --future 1 --period 20 --dim 10 --init random >> data/batch_mfcc.txt
    #python batch_learn_mfcc.py --past 1 --future 1 --period 1 --dim 10 --init random
    #python batch_learn_mfcc.py --past 20 --future 20 --period 1 --dim 10 --init random >> data/batch_mfcc2.txt
    #python batch_learn.py --past 1 --future 1 --period 1 --dim 10 --init random >> data/batch_zeros_1_1_1.txt

    #python batch_learn.py --past 20 --future 5 --period 1 --dim 10 --init random >> data/batch_zeros_20_5_1.txt
    #python batch_learn.py --past 12 --future 12 --period 20 --dim 10 --init random >> data/batch_zeros_12_12_20.txt
    python batch_learn.py --past 20 --future 10 --period 5 --dim 10 --init random #>> data/batch_zeros_20_10_5.txt
    #python batch_learn.py |& tee -a data/batch/log.txt
done
