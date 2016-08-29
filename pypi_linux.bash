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

python setup.py register -r pypi
#python setup.py sdist upload -r pypi
python setup.py sdist bdist_wheel upload -r pypi
