SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
ROOT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPTPATH")")")"
echo $ROOT_DIR
cd $ROOT_DIR
cp -r `$1` test_data
tar -xf test_data/dat_files.tar.gz -C test_data

git clone https://github.com/VulcanClimateModeling/buildenv &>/dev/null`
if [ $? -ne 0 ] ; then
    echo "Error: Could not download the buildenv (https://github.com/VulcanClimateModeling/buildenv). Aborting."
    exit 1
fi


#args:
#- #ranks
#- experiment
#- size
#- dir to store
