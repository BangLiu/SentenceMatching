#!/bin/sh
<<USAGE
Turn sentences into s-expression style normalized sentences.
Each line in the input file is a sentence.
Each line in the output file is a normalized sentence.
USAGE

# tools directory settings
X=../../tools/JAMR/jamr/
JAMR_PATH="$(cd "$(dirname "$X")"; pwd)/$(basename "$X")"
X=../../tools/CornellAMR/amr/
CORNELL_AMR_PATH="$(cd "$(dirname "$X")"; pwd)/$(basename "$X")"
WORK_PATH=`pwd`

# get parameters
parser=$1
input="$(cd "$(dirname "$2")"; pwd)/$(basename "$2")"
output="$(cd "$(dirname "$3")"; pwd)/$(basename "$3")"

echo "parser"
echo $parser

# parse and align sentences to AMR
if [ $parser = "JAMR" ] ; then
	echo "JAMR"
	cd $JAMR_PATH
	. scripts/config.sh
	scripts/PARSE.sh < $input > $output".aligned" 2> $output".err"
	cd $WORK_PATH
elif [ $parser = "CornellAMR" ] ; then  # seems performance not good when doing alignment.
	echo "Cornell AMR"
	cd $CORNELL_AMR_PATH
	java -Xmx8g -jar dist/amr-1.0.jar parse rootDir=`pwd` modelFile=`pwd`/amr.sp sentences=$input
	mv experiments/parse/logs/parse.out $output".CornellAMR"
	cd $WORK_PATH
	python process_cornell_amr.py -i $output".CornellAMR" -o $output".CornellAMR.processed"
	cd $JAMR_PATH
	. scripts/config.sh
	scripts/ALIGN.sh < $output".CornellAMR.processed" > $output".aligned"
	cd $WORK_PATH
else
	echo "Other parser not implemented yet. Choose JAMR or CornellAMR."
fi

# get s-expression style sentence
python normalize_sentence.py -i $output".aligned" -o $output
