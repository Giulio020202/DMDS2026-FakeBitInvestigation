#!/usr/bin/env bash
# Rigenera spiegazione_progetto.pdf (in questa cartella, documentazione/) da spiegazione_progetto.md.
# Serve pandoc (anche quello di `pip install pypandoc_binary`: PANDOC=$(python -c "import pypandoc; print(pypandoc.get_pandoc_path())"))
# e lualatex. Font: DejaVu di matplotlib (dentro OFF_MDS_ENV) + "Noto Sans Symbols 2" di sistema per ✔ ✘ ⚠.
# Uso (da qualunque cartella):  PANDOC=/percorso/pandoc bash genera_pdf.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PANDOC="${PANDOC:-pandoc}"
FONTS="$(ls -d "$REPO"/OFF_MDS_ENV/lib/python3*/site-packages/matplotlib/mpl-data/fonts/ttf | head -1)/"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# la copia per il PDF: senza la prima riga (titolo) e il sottotitolo, che vanno nei metadati; senza l'emoji 🗣
# (non c'e' in nessun font); i simboli ✔ ✘ ⚠ con un font che li contiene
tail -n +4 "$HERE/spiegazione_progetto.md" | sed 's/🗣 //g' \
  | perl -CSD -Mutf8 -pe 's/([✔✘⚠])/`{\\simboli $1}`{=latex}/g' > "$TMP/corpo.md"
printf '%s\n' '\newfontfamily\simboli{Noto Sans Symbols 2}' > "$TMP/header.tex"

cd "$HERE"
"$PANDOC" "$TMP/corpo.md" -o "$HERE/spiegazione_progetto.pdf" --pdf-engine=lualatex \
  --toc --toc-depth=1 --resource-path="$HERE" -H "$TMP/header.tex" \
  -M title="Capture the Mark 2026: la difesa del gruppo fakebitinvestigation" \
  -M subtitle='Versione 2.0 "ibrida" (DCT + wavelet): spiegazione completa di strategia, algoritmo e codice' -M lang=it \
  -V geometry:margin=2cm -V fontsize=10pt -V colorlinks=true \
  -V mainfont=DejaVuSerif -V "mainfontoptions=Path=$FONTS,Extension=.ttf,UprightFont=*,BoldFont=*-Bold,ItalicFont=*-Italic,BoldItalicFont=*-BoldItalic" \
  -V monofont=DejaVuSansMono -V "monofontoptions=Path=$FONTS,Extension=.ttf,UprightFont=*,BoldFont=*-Bold,ItalicFont=*-Oblique,BoldItalicFont=*-BoldOblique" \
  -V sansfont=DejaVuSans -V "sansfontoptions=Path=$FONTS,Extension=.ttf,UprightFont=*,BoldFont=*-Bold,ItalicFont=*-Oblique,BoldItalicFont=*-BoldOblique"
echo "scritto $HERE/spiegazione_progetto.pdf"
