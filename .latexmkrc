# Shared multi-pass PDF build settings. Use compile_lecture.sh to publish handouts.
$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode -halt-on-error %O %S';
# The build driver sets a separate output directory for each deck.
