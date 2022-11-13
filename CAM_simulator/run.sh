RUN_EXE="./build/CAM_architecture_simulation"
DATA_FOLDER1="../data/CSR/random"
DATA_FOLDER2="../data/CSR/force_test"

DATA_FOLDER=$DATA_FOLDER2
OUTPUT_FOLDER="../output"
TRACE_FOLDER="../output/trace/force_test"

OUTFILE="force_test"
mkdir $OUTPUT_FOLDER
mkdir $TRACE_FOLDER
for GRAPH in "citeseer" #"p2p"  "mico" "patent" "youtube" "astro" "email-Enron" "roadNet-PA" "roadNet-TX" "lj" #
do
    mkdir $TRACE_FOLDER/$GRAPH
    $RUN_EXE $DATA_FOLDER/$GRAPH.bin 1 1 $TRACE_FOLDER/$GRAPH/$GRAPH >> $OUTPUT_FOLDER/$OUTFILE.stdout
done