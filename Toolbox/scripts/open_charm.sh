source ~/miniconda2/bin/activate AST
cd ..
unset PYTHONPATH
export PYTHONPATH=$(pwd):$(pwd)/garage:$PYTHONPATH
cd ../..
charm AdaptiveStressTestingToolbox &
