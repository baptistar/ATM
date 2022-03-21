#!/bin/bash

set -e

# 0) Check you are on private repository
PRIVATE_URL=$(git remote -v | grep fetch | cut -d/  -f2 | cut -d\  -f1)
if [ "$PRIVATE_URL" != "adaptivetransportmaps.git" ]; then
    echo "ERROR: You are not on the private repository adaptivetransportmaps.git"
    echo "  Current repository $PRIVATE_URL"
    exit 1
fi

# 1) Check whether we are on master branch
PRIVATE_PATH=$(pwd)
PRIVATE_BRANCH=$(git branch | grep \* | cut -d ' ' -f2)
if [ "$PRIVATE_BRANCH" == "master" ]; then
    echo "Updating from branch" $PRIVATE_BRANCH
else
    echo "You must be on branch master. Current branch" $PRIVATE_BRANCH
    exit 2
fi

# 1.a) Get version
VERSION=1.01
echo "Mirroring version $VERSION"

# 1.b) Pushing private
echo "Pushing private"
git push -f

# 2) Exclude list
declare -a EXCLUDE_LIST=("examples/ATMAlgorithmCriteria" "examples/Approximation_Theory" "examples/HighDimensionalSparseGaussian" "examples/Lorenz63" "examples/NeurIPS" "examples/RectifierPlots" "examples/Spectroscopy" "examples/ToyProblemOneDimensional" "examples/WaveletBasis" "examples/em31" "examples/icesat" "examples/PlotPBR" "testing/test_Grad_PullbackDensity.m" "examples/StochasticVolatility" "todo.md")

# 3) Check existence of ../ATM/
PUBLIC_PATH="../ATM/"
if cd "$PUBLIC_PATH" 
then 
    echo "Entered " $PUBLIC_PATH
else
    cd ..
    git clone git@github.com:baptistar/ATM.git
    cd ATM
fi

# 4) Pull ../TransportMaps/
echo "Pulling" $(pwd)
echo "git pull (public)"
git pull

# 4.a) Remove all previous version
echo "Remove old version (public)"
git ls-files -z | xargs -0 rm -f

# 5) git ls-tree -r master --name-only
cd "$PRIVATE_PATH"
echo "git ls-tree (private)"
for fname in $(git ls-tree -r master --name-only)
do
    SKIP=false
    for exp in "${EXCLUDE_LIST[@]}"
    do
        if [[ "$fname" == *$exp* ]];
        then
            SKIP=true;
            break;
        fi
    done
    if ! $SKIP;
    then
        dst=$PUBLIC_PATH$(dirname "$fname")
        mkdir -p "$dst"
        cp -a "$fname" "$dst"
    fi
done

# 6) git add all on public branch
cd "$PUBLIC_PATH" 
echo "git add (public)"
git add --all

# 7) git commit
echo "git commit (public)"
git commit -a -m "Mirroring version $VERSION"

# 8) git push
echo "git push (public)"
git push

# 9) git tag public repository
echo "git tag (public)"
git tag -f "v$VERSION" -m "v$VERSION"
git push -f --tags

# 10) git tag private repository
echo "git tag (private)"
cd "$PRIVATE_PATH"
git tag -f "v$VERSION" -m "v$VERSION"
git push -f --tags
