function deploy_doc(){
    if [ ! -z "$1" ]
    then
        git checkout $1
    fi
    COMMIT=$(git rev-parse --short HEAD)
    echo "Creating doc at commit" $COMMIT "and pushing to folder $2"
    # Hotfix
    sed -i "s/^torchvision.*/&,<0.11.0/" ../requirements.txt
    sed -i "s/torchvision>=.*',/&,<0.11.0',/" ../setup.py
    sed -i "s/',,/,/" ../setup.py
    pip install -U ..
    git checkout ../setup.py
    git checkout ../requirements.txt
    if [ ! -z "$2" ]
    then
        if [ "$2" == "latest" ]; then
            echo "Pushing main"
            sphinx-build source _build -a && mkdir build && mkdir build/$2 && cp -a _build/* build/$2/
        elif [ -d build/$2 ]; then
            echo "Directory" $2 "already exists"
        else
            echo "Pushing version" $2
            cp -r _static source/ && cp _conf.py source/conf.py
            sphinx-build source _build -a
            mkdir build/$2 && cp -a _build/* build/$2/ && rm -r source && git checkout source/
        fi
    else
        echo "Pushing stable"
        cp -r _static source/ && cp _conf.py source/conf.py
        sphinx-build source build -a && rm -r source && git checkout source/
    fi
}

# You can find the commit for each tag on https://github.com/frgfm/holocron/tags
if [ -d build ]; then rm -Rf build; fi
cp -r source/_static .
cp source/conf.py _conf.py
git fetch --all --tags --unshallow
deploy_doc "" latest
deploy_doc "e9ca768" v0.1.0
deploy_doc "9b3f927" v0.1.1
deploy_doc "59c3124" v0.1.2
deploy_doc "d41610b" v0.1.3
deploy_doc "67a50c7" # v0.2.0 Latest stable release
rm -rf _build _static _conf.py
