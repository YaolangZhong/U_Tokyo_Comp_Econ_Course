$out_dir = 'build';
$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 %O %S';

# This hook runs after the PDF is successfully made
$post_system = 'cp build/%R.pdf .';
