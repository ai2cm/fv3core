#!/bin/bash

SCRIPT=`realpath $0`
SCRIPT_DIR=`dirname $SCRIPT`
ROOT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

# activate Python environment
venv="$ROOT_DIR/venv"
if [ ! -f "${venv}/bin/activate" ] ; then
  echo "ERROR: problem with ${venv} could not find ${venv}/bin/activate"
  exit 1
fi
source ${venv}/bin/activate

# generate simple profile listing
cat > ./profile.py <<EOF
#!/usr/bin/env python3
import pstats
stats = pstats.Stats("$ROOT_DIR/fv3core_${experiment}_${backend}_0.prof")
stats.strip_dirs()
print('=================================================================')
stats.sort_stats('time')
stats.print_stats(200)
print('=================================================================')
stats.sort_stats('cumulative')
stats.print_stats(200)
print('=================================================================')
stats.sort_stats('calls')
stats.print_stats(200)
print('=================================================================')
EOF
chmod 755 ./profile.py
./profile.py > profile.txt

# convert to html
mkdir -p html
echo "<html><body><pre>" > html/index.html
cat profile.txt >> html/index.html
echo "</pre></body></html>" >> html/index.html
