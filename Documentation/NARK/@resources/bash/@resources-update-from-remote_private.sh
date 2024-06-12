#!/bin/bash
# Pull down the latest @resources and replace local files
# with the same names.  Leave alone any existing files
# that do not have a counterpart in in the upstream
# written by Claude, edited by CDC

# script should live in directory of the @resources file to be updated

# path to the directory the script lives in

# Set the GitHub repository URL and the desired subdirectory
repo_url="https://github.com/econ-ark/econ-ark-tools.git"

# directory of this script
here="$(realpath $(dirname $0))" # here=/Volumes/Data/Papers/BufferStockTheory/BufferStockTheory-Latest/

# subdirectory path
repo_subdir="@resources"

repo_url_root="https://github.com/econ-ark/econ-ark-tools"
resources="@resources"
repo_dirpath="$repo_url_root/$resources"

# Set the destination directory on your macOS computer
dest_dir="$here/@resources"

# Create a temporary directory for cloning the repository
temp_dir=$(mktemp -d)

# Clone the GitHub repository into the temporary directory
git clone --depth 1 "$repo_url" "$temp_dir"

# Navigate to the desired subdirectory within the cloned repository
pushd . ; cd "$temp_dir/$repo_subdir"

# Copy the contents of the subdirectory to the destination directory
rsync -avh --delete --checksum --itemize-changes --out-format="%i %n%L" . "$dest_dir" | grep '^>f.*c' | tee >(awk 'END { if (NR == 0) print "\nno files were changed\n"; else print NR, "files were changed\n" }')

# Remove the temporary directory
#rm -rf "$temp_dir"

# Return to the source directory
popd
