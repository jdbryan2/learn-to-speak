for k in {1..1}
do 
    echo Random Breathing: $k
    python generate_data.py --round $k --init random --breathe random
    python generate_data.py --round $k --init zeros --breathe random
done

for k in {1..10}
do 
    echo Manual Breathing: $k
    python generate_data.py --round $k --init random --breathe manual
    python generate_data.py --round $k --init zeros --breathe manual
done
