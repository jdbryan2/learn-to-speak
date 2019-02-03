
for k in {1..10}
do
    echo Batch $k
    python primitive_foward_eval.py --batch $k
done
