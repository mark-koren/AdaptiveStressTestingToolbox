#!/bin/bash

source ~/miniconda2/bin/activate AST
cd ../..
unset PYTHONPATH
#export PYTHONPATH=$(pwd):$(pwd)/rltools:$(pwd)/garage:$PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$PWD/Toolbox/garage
export PYTHONPATH=$PYTHONPATH:$PWD/TestCases
export PYTHONPATH=$PYTHONPATH:$PWD/Toolbox
#pydot from https://github.com/nlhepler/pydot
#neet run python setup.py install
export PYTHONPATH=$PYTHONPATH:$PWD/Toolbox/pydot
cd Toolbox/scripts

while getopts ":g:r:" opt; do
  case "$opt" in
    g) group=$OPTARG ;;
    r) runner=$OPTARG ;;
  esac
done
shift $(( OPTIND - 1 ))

path="../data/$group"

for run_name in "$@"; do
  args=""
  while IFS='' read -r line || [[ -n "$line" ]]; do
    args="$args $line"
    echo "Text read from file: $line"
  done < "$path/$run_name/$run_name.txt"
  echo "python ../../TestCases/$runner $args"
  python ../../TestCases/$runner $args --log_dir $path/$run_name
  mkdir -p ~/box/Research/Data/DRL-AST/$group/$run_name
  cp -a $path/$run_name/. ~/box/Research/Data/DRL-AST/$group/$run_name/ &>/dev/null &
done

source ~/miniconda2/bin/deactivate
