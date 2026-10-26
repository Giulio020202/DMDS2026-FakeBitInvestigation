#!/usr/bin/env bash
# Regenerates project_explanation.pdf (in this folder, documentation/) from project_explanation.md.
# Needs pandoc (also the one of `pip install pypandoc_binary`: PANDOC=$(python -c "import pypandoc; print(pypandoc.get_pandoc_path())"))
# and lualatex. Fonts: DejaVu from matplotlib (inside OFF_MDS_ENV) + system "Noto Sans Symbols 2" for ✔ ✘ ⚠.
# Usage (from any folder):  PANDOC=/path/to/pandoc bash make_pdf.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PANDOC="${PANDOC:-pandoc}"
FONTS="$(ls -d "$REPO"/OFF_MDS_ENV/lib/python3*/site-packages/matplotlib/mpl-data/fonts/ttf | head -1)/"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# the copy for the PDF: without the first line (title) and the subtitle, which go in the metadata; without the 🗣 emoji
# (it is in no font); the symbols ✔ ✘ ⚠ with a font that contains them
tail -n +4 "$HERE/project_explanation.md" | sed 's/🗣 //g' \
  | perl -CSD -Mutf8 -pe 's/([✔✘⚠])/`{\\symbols $1}`{=latex}/g' > "$TMP/body.md"
printf '%s\n' '\newfontfamily\symbols{Noto Sans Symbols 2}' > "$TMP/header.tex"

cd "$HERE"
"$PANDOC" "$TMP/body.md" -o "$HERE/project_explanation.pdf" --pdf-engine=lualatex \
  --toc --toc-depth=2 --resource-path="$HERE" -H "$TMP/header.tex" \
  -M title="Capture the Mark 2026: the defense of group fakebitinvestigation" \
  -M subtitle='Version 2.0 “hybrid” (DCT + wavelet): full explanation of strategy, algorithm and code' -M lang=en \
  -V geometry:margin=2cm -V fontsize=10pt -V colorlinks=true \
  -V mainfont=DejaVuSerif -V "mainfontoptions=Path=$FONTS,Extension=.ttf,UprightFont=*,BoldFont=*-Bold,ItalicFont=*-Italic,BoldItalicFont=*-BoldItalic" \
  -V monofont=DejaVuSansMono -V "monofontoptions=Path=$FONTS,Extension=.ttf,UprightFont=*,BoldFont=*-Bold,ItalicFont=*-Oblique,BoldItalicFont=*-BoldOblique" \
  -V sansfont=DejaVuSans -V "sansfontoptions=Path=$FONTS,Extension=.ttf,UprightFont=*,BoldFont=*-Bold,ItalicFont=*-Oblique,BoldItalicFont=*-BoldOblique"
echo "written $HERE/project_explanation.pdf"
