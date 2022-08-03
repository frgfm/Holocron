function deploy_doc(){
    if [ ! -z "$1" ]
    then
        git checkout $1
    fi
    COMMIT=$(git rev-parse --short HEAD)
    echo "Creating doc at commit" $COMMIT "and pushing to folder $2"
    # Hotfix
    if [ -d ../requirements.txt ]; then
        sed -i "s/^torchvision.*/&,<0.11.0/" ../requirements.txt
    fi
    sed -i "s/torchvision>=.*',/&,<0.11.0',/" ../setup.py
    sed -i "s/',,/,/" ../setup.py
    pip install -U ..
    git checkout ../setup.py
    if [ -d ../requirements.txt ]; then
        git checkout ../requirements.txt
    fi
    if [ ! -z "$2" ]
    then
        if [ "$2" == "latest" ]; then
            echo "Pushing main"
            mkdir build/$2
            sphinx-build source build/$2 -a
        elif [ -d build/$2 ]; then
            echo "Directory" $2 "already exists"
        else
            echo "Pushing version" $2
            cp -r _static source/ && cp _conf.py source/conf.py
            mkdir build/$2
            sphinx-build source build/$2 -a
        fi
    else
        echo "Pushing stable"
        cp -r _static source/ && cp _conf.py source/conf.py
        sphinx-build source build -a
    fi
    git checkout source/ && git clean -f source/
}

# exit when any command fails
set -e
# You can find the commit for each tag on https://github.com/frgfm/holocron/tags
if [ -d build ]; then rm -Rf build; fi
mkdir build
cp -r source/_static .
cp source/conf.py _conf.py
git fetch --all --tags --unshallow
deploy_doc "" latest
deploy_doc "e9ca768" v0.1.0
deploy_doc "9b3f927" v0.1.1
deploy_doc "59c3124" v0.1.2
deploy_doc "d41610b" v0.1.3
deploy_doc "67a50c7" v0.2.0
deploy_doc "bc0d972" # v0.2.1 Latest stable release
rm -rf _build _static _conf.py
