#!/usr/local/bin/bash
# Some dependencies don't support manylinux1
docker build . -f manylinux2010.Dockerfile --tag seunglab/kimimaro:manylinux2010
docker build . -f manylinux2014.Dockerfile --tag seunglab/kimimaro:manylinux2014
docker run -v $PWD/dist:/output seunglab/kimimaro:manylinux2010 /bin/bash -c "cp -r wheelhouse/* /output"
docker run -v $PWD/dist:/output seunglab/kimimaro:manylinux2014 /bin/bash -c "cp -r wheelhouse/* /output"