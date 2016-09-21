#!/bin/bash
version="$1"
if [ $# -eq 0 ]
	then
		echo "Must provide command line version"
		exit 1
fi

python setup.py bdist_wheel
python setup.py sdist

# SDIST
cp "dist/zorroautomator-$version.tar.gz" "/mnt/qnap01/Robert_McLeod/build_zorro/"

# BC2
rsync -v "dist/zorroautomator-$version.tar.gz" mcleod@login2.bc2.unibas.ch:~/
ssh mcleod@login2.bc2.unibas.ch "~/bin/reinstallZorro27.bash $version"
ssh mcleod@login2.bc2.unibas.ch "~/bin/reinstallZorro34.bash $version"


# LOC-CLUSTER
rsync -v "dist/zorroautomator-$version.tar.gz" lcluster@ram.ethz.ch:~/
ssh  -t lcluster@ram.ethz.ch ssh loc-login "~/bin/reinstallZorro.bash $version"

# D-BSSE
#cp "dist/zorroautomator-$version"-cp34*.whl /mnt/qnap01/Robert_McLeod/build_zorro/
# I can't use the wheel because the RHEL is only Qt 4.8.5
ssh rmcleod@bs-gpu04 "~/bin/reinstallZorro34.bash $version"
ssh rmcleod@bs-gpu04 "~/bin/reinstallZorro27.bash $version"

