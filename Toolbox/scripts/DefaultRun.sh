#!/bin/bash

while getopts ":g:" opt; do
  case "$opt" in
    g) group=$OPTARG ;;
  esac
done
shift $(( OPTIND - 1 ))

path="../data/$group"
mkdir -p "$path"

for run_name in "$@"; do
  mkdir "$path/$run_name"
  cp settings_default.txt $path/$run_name/$run_name.txt
done
