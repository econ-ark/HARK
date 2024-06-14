#!/usr/bin/env bash

add_newlines() {
    # Read the input text
    text="$1"

    # Regular expression pattern to match any LaTeX environment or comment
    pattern='(\\begin{.*?}.*?\\end{.*?}|%.*$)'

    # Process each line
    while IFS= read -r line; do
        # Extract LaTeX environments and comments from the line
        matches=$(printf "%s" "$line" | sed -n "s/.*\($pattern\).*/\1/p")

        # Replace LaTeX environments and comments with placeholders
        placeholder_line=$(printf "%s" "$line" | sed "s/$pattern/__PLACEHOLDER__/g")

        # Regular expression pattern to match sentence endings
        sentence_pattern='(?<=[.!?])['\''"]?(?=\s|$)'

        # Replace sentence endings with newline character
        formatted_line=$(printf "%s" "$placeholder_line" | sed "s/$sentence_pattern/\n/g")

        # Reinsert LaTeX environments and comments back into the formatted line
        while read -r match; do
            formatted_line=${formatted_line/__PLACEHOLDER__/$match}
        done <<< "$matches"

        printf "%s\n" "$formatted_line"
    done <<< "$text"
}

# Debugging line to test the number of arguments
echo "Number of arguments: $#"

# Check if the input and output file names are provided as arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_file> <output_file>"
    exit 1
fi

# Get the input and output file names from the command line arguments
input_file="$1"
output_file="$2"

# Check if the input file exists
if [ ! -f "$input_file" ]; then
    echo "Input file '$input_file' does not exist."
    exit 1
fi

# Read the contents of the input file
text=$(cat "$input_file")

# Process the text
formatted_text=$(add_newlines "$text")

# Write the formatted text to the output file
echo "$formatted_text" > "$output_file"

echo "Input file '$input_file' has been processed and the formatted text has been written to '$output_file'."
