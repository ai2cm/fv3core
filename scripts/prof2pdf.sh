#!/bin/bash
gprof2dot -f pstats $1 > ${1}.dot
# dot -Tpng ${1}.dot -o ${1}.png
dot -Tpdf ${1}.dot -o ${1}.pdf
rm -f ${1}.dot
echo "output: ${1}.pdf"
