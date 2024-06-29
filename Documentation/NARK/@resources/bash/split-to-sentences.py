import re
import sys
import os


def add_newlines(text):
    # Regular expression pattern to match any LaTeX environment or comment
    pattern = r"(\\begin{.*?}.*?\\end{.*?}|%.*$)"

    # Split the text into lines
    lines = text.split("\n")

    # Process each line
    formatted_lines = []
    inside_env = False
    for line in lines:
        # Check if the line contains a LaTeX environment
        if re.search(r"\\begin{.*?}", line):
            inside_env = True
        elif re.search(r"\\end{.*?}", line):
            inside_env = False

        # Extract LaTeX environments and comments from the line
        matches = re.findall(pattern, line)

        # Replace LaTeX environments and comments with placeholders
        placeholder_line = re.sub(pattern, "__PLACEHOLDER__", line)

        # Regular expression pattern to match sentence endings
        sentence_pattern = r"(?<=[.!?])\s+(?=\S)"

        # Replace sentence endings with newline character only if there is further non-whitespace material on the line
        # and not inside a LaTeX environment
        if not inside_env:
            formatted_line = re.sub(sentence_pattern, "\n", placeholder_line)
        else:
            formatted_line = placeholder_line

        # Reinsert LaTeX environments and comments back into the formatted line
        for match in matches:
            formatted_line = formatted_line.replace("__PLACEHOLDER__", match, 1)

        formatted_lines.append(formatted_line)

    # Join the formatted lines back into a single string
    formatted_text = "\n".join(formatted_lines)

    return formatted_text


# Check if the input file name is provided as a command-line argument
if len(sys.argv) != 2:
    print("Usage: python split_to_sentences.py <input_file>")
    sys.exit(1)

# Get the input file name from the command-line argument
input_file = sys.argv[1]

# Generate the output file name by appending "-sentenced" to the input file name
output_file = os.path.splitext(input_file)[0] + "-sentenced.tex"

# Read the input file
try:
    with open(input_file, "r") as file:
        text = file.read()
except FileNotFoundError:
    print(f"Input file '{input_file}' not found.")
    sys.exit(1)

# Process the text
formatted_text = add_newlines(text)

# Write the formatted text to the output file
with open(output_file, "w") as file:
    file.write(formatted_text)

print(f"Text has been processed and saved to '{output_file}'.")
