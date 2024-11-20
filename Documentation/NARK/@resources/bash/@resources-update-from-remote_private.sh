#!/bin/bash

# Pull down the latest @resources and replace local files
# with the same names.  Leave alone any existing files
# that do not have a counterpart in in the upstream
# written by Claude, edited by CDC
# 20240611: fix permissions; cleanup

# script should live in directory of the @resources file to be updated

# Set the GitHub repository URL and the desired subdirectory
repo_url="https://github.com/econ-ark/econ-ark-tools.git"

# directory of this script
here="$(realpath $(dirname $0))" # here=/Volumes/Data/Papers/BufferStockTheory/SolvingMicroDSOPs-Latest

# subdirectory path
repo_subdir="@resources"

repo_url_root="https://github.com/econ-ark/econ-ark-tools"
resources="@resources"
repo_dirpath="$repo_url_root/$resources"

# Set the destination directory on your macOS computer
dest_dir="$here/@resources"

# Change its permissions to allow writing
chmod -Rf u+w "$dest_dir"

# Clone the GitHub repository into the temporary directory
[[ -e /tmp/@resources ]] && rm -rf /tmp/@resources
git clone --depth 1 "$repo_url" /tmp/@resources

# Navigate to the desired subdirectory within the cloned repository
src_dir="/tmp/@resources/$repo_subdir"

# Copy the contents of the subdirectory to the destination directory,
# printing a list of the files that were changed:
echo '' ; echo rsync "$src_dir/" "$dest_dir" # tell
rsync -avh --delete --checksum --itemize-changes --out-format="%i %n%L" "$src_dir/" "$dest_dir" | grep '^>f.*c' | tee >(awk 'END { if (NR == 0) print "\nno files were changed\n"; else print NR, "files were changed\n" }')

# Remove the temporary directory
rm -rf "$temp_dir"

# # Return to the source directory
popd

# Change to read-only; edits should be done upstream
chmod u-w "$dest_dir"
