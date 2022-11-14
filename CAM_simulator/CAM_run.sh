RUN_EXE="./build/CAM_architecture_simulation"
DATA_FOLDER1="../data/CSR/random_order"
DATA_FOLDER2="../data/CSR/force_order"

DATA_FOLDER=$DATA_FOLDER2
OUTPUT_FOLDER="../output"
TRACE_FOLDER="../output/trace/force_order"

OUTFILE="force_order"
mkdir $OUTPUT_FOLDER
mkdir $TRACE_FOLDER
for GRAPH in "citeseer" 
do
    mkdir $TRACE_FOLDER/$GRAPH
    $RUN_EXE $DATA_FOLDER/$GRAPH.bin 1 1 $TRACE_FOLDER/$GRAPH/$GRAPH >> $OUTPUT_FOLDER/$OUTFILE.stdout
done