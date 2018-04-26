for k in {1..100}
do 
    echo Batch $k
    python batch_learn.py --past 12 --future 12 --period 20 --dim 10 --init random >> data/batch_zeros_12_12_20.txt
    #python batch_learn.py --past 20 --future 10 --period 5 --dim 10 --init random >> data/batch_zeros_20_10_5.txt
    #python batch_learn.py |& tee -a data/batch/log.txt
done
