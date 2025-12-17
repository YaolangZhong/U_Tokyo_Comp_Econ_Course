$out_dir = 'build';
$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 %O %S';

# This hook runs after the PDF is successfully made
# Copies the PDF to the root folder while keeping auxiliary files in build/
$success_cmd = 'cp build/%R.pdf .';
