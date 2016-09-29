#!/bin/bash
#version="$1"
#if [ $# -eq 0 ]
#	then
#		echo "Must provide command line version"
#		exit 1
#fi

# Linux wheels not allowe, but numpy has them?
#python setup.py bdist_wheel
#python setup.py sdist

# SDIST
#cp "dist/zorro-$version.tar.gz" "/mnt/qnap01/Robert_McLeod/build_zorro/"
source activate py27
python setup.py register -r pypi
python setup.py sdist upload -r pypi
# TODO: cp27-cp27m-manylinux1_x86_64.whl
#python setup.py sdist bdist_wheel upload -r pypi
