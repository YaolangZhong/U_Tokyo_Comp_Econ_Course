#!/bin/bash
# LaTeX Compilation Script
# Usage: ./compile_lecture.sh Lecture_9_Parameterization_and_Neural_Network
# or: ./compile_lecture.sh Lecture_9_Parameterization_and_Neural_Network.tex

# Get the filename without extension
if [ -z "$1" ]; then
    echo "Usage: $0 <lecture_name>"
    echo "Example: $0 Lecture_9_Parameterization_and_Neural_Network"
    exit 1
fi

# Remove .tex extension if provided
FILENAME="${1%.tex}"

# Check if the tex file exists
if [ ! -f "${FILENAME}.tex" ]; then
    echo "Error: ${FILENAME}.tex not found"
    exit 1
fi

echo "Compiling ${FILENAME}.tex..."
echo "Auxiliary files will be stored in build/"
echo "PDF will be generated in both build/ and root directory"
echo ""

# Compile using latexmk
latexmk -pdf "${FILENAME}.tex"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Compilation successful!"
    echo "  - Auxiliary files: build/${FILENAME}.*"
    echo "  - PDF output: ${FILENAME}.pdf (root) and build/${FILENAME}.pdf"
else
    echo ""
    echo "✗ Compilation failed. Check build/${FILENAME}.log for details."
    exit 1
fi


