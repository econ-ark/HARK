#!/usr/bin/env bash

# Function to process a single file
process_file() {
    input_file="$1"
    python ./split-to-sentences.py "$input_file"
}

# Check if the split-to-sentences.sh script exists in the current directory
if [ ! -f "split-to-sentences.py" ]; then
    echo "Error: split-to-sentences.py script not found in the current directory."
    exit 1
fi

# Make sure the split-to-sentences.sh script is executable
chmod +x split-to-sentences.py

# Loop through all .tex files in the current directory
for file in *.tex; do
    # Check if the file exists
    if [ -f "$file" ]; then
        echo "Processing file: $file"
        process_file "$file"
        echo "Processed file saved as: ${file%.tex}-sentenced.tex"
        echo "------------------------"
    fi
done

echo "All .tex files in the current directory have been processed."
