#!/bin/sh

if [ $# -eq 0 ]
then
  echo "usage: ${0##*/} <handoutName>"
  exit 1
fi

handoutName=$1

# cd "$(dirname "$0")" # http://stackoverflow.com/questions/3349105/how-to-set-current

cmd="cp `kpsewhich svg-math-and-subfigures.cfg` $handoutName.cfg"
echo "$cmd" ; eval "$cmd"
#cp `kpsewhich svg-set-size-to-1p0.mk4` $handoutName.mk4
cmd="cp `kpsewhich svg-set-size-to-1p0.mk4` $handoutName.mk4"
#cp `kpsewhich svg-set-size-to-1p2.mk4` $handoutName.mk4
#cmd="cp `kpsewhich svg-set-size-to-1p3x1p0.mk4` $handoutName.mk4"
#cmd="cp `kpsewhich svg-set-size-to-1p2x1p0.mk4` $handoutName.mk4"
#cmd="cp `kpsewhich svg-set-size-to-1p1x1p0.mk4` $handoutName.mk4"  # Tried this to achieve bold; does not look good
echo "$cmd" ; eval "$cmd"


