#!/usr/bin/env bash
# Regenerates CHALLENGE_INSTRUCTIONS.pdf (in this folder, challenge/) from CHALLENGE_INSTRUCTIONS.md.
# Same tools as ../documentation/make_pdf.sh: pandoc (also the one of `pip install pypandoc_binary`:
# PANDOC=$(python -c "import pypandoc; print(pypandoc.get_pandoc_path())")) and lualatex.
# Fonts: DejaVu from matplotlib (inside OFF_MDS_ENV) + system "Noto Sans Symbols 2" for ✔ ✘ ⚠ ☐.
# Usage (from any folder):  PANDOC=/path/to/pandoc bash make_pdf.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PANDOC="${PANDOC:-pandoc}"
FONTS="$(ls -d "$REPO"/OFF_MDS_ENV/lib/python3*/site-packages/matplotlib/mpl-data/fonts/ttf | head -1)/"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# the copy for the PDF: without the first line (title, it goes in the metadata); without the 🗣 emoji
# (it is in no font); the checkboxes "- [ ]" as ☐ and the symbols ✔ ✘ ⚠ ☐ with a font that contains them
tail -n +2 "$HERE/CHALLENGE_INSTRUCTIONS.md" | sed 's/🗣 //g; s/^\( *\)- \[ \] /\1- ☐ /' \
  | perl -CSD -Mutf8 -pe 's/([✔✘⚠☐])/`{\\symbols $1}`{=latex}/g' > "$TMP/body.md"
# long commands wrap instead of going past the right margin (they must stay readable on paper)
cat > "$TMP/header.tex" <<'TEX'
\newfontfamily\symbols{Noto Sans Symbols 2}
\usepackage{fvextra}
\DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,breakanywhere,commandchars=\\\{\}}
\RecustomVerbatimEnvironment{verbatim}{Verbatim}{breaklines,breakanywhere}
TEX

cd "$HERE"
"$PANDOC" "$TMP/body.md" -o "$HERE/CHALLENGE_INSTRUCTIONS.pdf" --pdf-engine=lualatex \
  --toc --toc-depth=2 -H "$TMP/header.tex" \
  -M title="Il giorno della gara (2/11): cosa fare, passo per passo" \
  -M subtitle="Capture the Mark 2026, gruppo fakebitinvestigation" -M lang=it \
  -V geometry:margin=2cm -V fontsize=10pt -V colorlinks=true \
  -V mainfont=DejaVuSerif -V "mainfontoptions=Path=$FONTS,Extension=.ttf,UprightFont=*,BoldFont=*-Bold,ItalicFont=*-Italic,BoldItalicFont=*-BoldItalic" \
  -V monofont=DejaVuSansMono -V "monofontoptions=Path=$FONTS,Extension=.ttf,UprightFont=*,BoldFont=*-Bold,ItalicFont=*-Oblique,BoldItalicFont=*-BoldOblique" \
  -V sansfont=DejaVuSans -V "sansfontoptions=Path=$FONTS,Extension=.ttf,UprightFont=*,BoldFont=*-Bold,ItalicFont=*-Oblique,BoldItalicFont=*-BoldOblique"
echo "written $HERE/CHALLENGE_INSTRUCTIONS.pdf"
