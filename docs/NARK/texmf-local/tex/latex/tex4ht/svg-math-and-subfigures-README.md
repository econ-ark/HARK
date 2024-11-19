20191212 - struggled for several hours to get svg figures to work properly

The current version works

There seem to be two options:

1. If you never set a specified size of a figure (like \includegraphics[width=6in]) then the method discussed [here](https://tug.org/pipermail/tex4ht/2015q2/001166.html) works. But dies with a division by zero error whenever the size has been explicitly set. It can be resurrected by uncommenting the \Configure{graphics*}{svg} portion of the file
1. In cases where the size is set (as is frequently the case), the current version obtains the dimensions directly from the svg file
   * This might have problems with an svg file without explicit dimensions
