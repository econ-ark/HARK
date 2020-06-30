#!/bin/sh

if [ $# -eq 0 ]
then
  echo "usage: ${0##*/} <handoutName>"
  exit 1
fi

handoutName=$1

# cd "$(dirname "$0")" # http://stackoverflow.com/questions/3349105/how-to-set-current

echo cp `kpsewhich svg-math-and-subfigures.cfg` $handoutName.cfg
echo cp `kpsewhich svg-set-size-to-1p0.mk4` $handoutName.mk4
cp `kpsewhich svg-math-and-subfigures.cfg` $handoutName.cfg
cp `kpsewhich svg-set-size-to-1p0.mk4` $handoutName.mk4

