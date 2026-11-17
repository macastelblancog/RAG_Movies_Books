# data/raw — procedencia y ubicación de los datos

**Nada de esta carpeta se versiona** (ver `.gitignore`). Los libros y guiones tienen derechos de
autor: cada persona debe obtener su propia copia legal. El repositorio publica solo código y
resultados derivados (estadísticas, figuras, métricas).

## Estructura esperada (fuente de verdad: `configs/corpora.yaml`)

```
data/raw/
├── hp/
│   ├── book/hp{2..7}_<titulo>.pdf                  # Pottermore 2015 (libro 1 no disponible)
│   ├── maven/Dialogue.csv, Characters.csv, Chapters.csv, Places.csv
│   └── kaggle/Harry Potter {1,2,3}.csv             # Character;Sentence (control)
├── lotr/
│   ├── book/LOTR - ALL.pdf                         # trilogía completa en un PDF
│   ├── screenplay/LordoftheRings{1-FOTR,2-TTT,3-ROTK}.pdf
│   └── kaggle/lotr_scripts.csv
├── dune/
│   ├── book/dune.pdf
│   └── screenplay/dune_part_one.pdf, dune_part_two.pdf
└── narnia/
    ├── book/narnia_7_books.pdf                     # ómnibus, orden de publicación
    └── screenplay/lww_2005.pdf (+ .layout.json OCR), prince_caspian_2008.pdf (no incluido)
```

## Procedencia

| Archivo | Origen | Licencia / nota |
|---|---|---|
| `hp/maven/*.csv` | Maven Analytics Data Playground, *Harry Potter Movie Scripts* | CC BY-SA 3.0, con atribución |
| `hp/kaggle/Harry Potter {1,2,3}.csv` | Kaggle, *Harry Potter Dataset* | Diálogo `Character;Sentence` de las películas 1–3; control |
| `hp/book/*.pdf` | Pottermore Publishing, 2015 (Enhanced Edition) | © J.K. Rowling. Copia propia |
| `lotr/kaggle/lotr_scripts.csv` | Kaggle, *Lord of the Rings Movies Analysis* (tnyont) | Diálogos sin escena; solo para conteos de control |
| `lotr/screenplay/*.pdf` | Guiones de rodaje de Walsh, Boyens & Jackson (New Line Cinema, 2001–2003); ROTK es "Final Revision, Oct 2003" | © New Line Cinema |
| `lotr/book/LOTR - ALL.pdf` | Edición de HarperCollins (incluye nota del 50.º aniversario) | © Tolkien Estate. Copia propia |
| `dune/screenplay/dune_part_two.pdf` | *Dune Part Two (final version)*, Final Draft 11, 119 págs. (difundido por Deadline, *Read the Screenplay*, 2024) | © Warner Bros / Legendary. Uso académico |
| `dune/screenplay/dune_part_one.pdf` | *DUNE*, borrador "Salmon Rev. 06/19/2020", Final Draft 11, 111 págs. | © Warner Bros / Legendary. Uso académico |
| `dune/book/dune.pdf` | PDF de la novela, 345 págs.; capítulos separados por "= = = = = =" | © Herbert Estate. Copia propia |
| `narnia/book/narnia_7_books.pdf` | Ómnibus de los 7 libros (PDFDrive), 664 págs. | © C.S. Lewis Pte Ltd; en Colombia sigue protegido (vida del autor + 80 años). Copia propia |
| `narnia/screenplay/lww_2005.pdf` | Guion de Markus & McFeely, "AA Rev. 12-10-03, Estate Approved", escaneado (130 págs., sin capa de texto) | © Walden Media / Disney. Uso académico |
| `narnia/screenplay/lww_2005.pdf.layout.json` | Derivado: líneas OCR con coordenada x (Tesseract 5, 240 dpi, `--psm 4`) | Contiene texto del guion: no se versiona |

### No usados
- `harry_potter_reviews.csv` (Kaggle, marcpaulo): **491 reseñas sintéticas generadas con un LLM**.
  Se excluyen del análisis.
- PDFs de HP en español: el análisis es en inglés, el idioma de las películas.

## Checklist al agregar una fuente nueva
1. Copiar el archivo a la ruta indicada en `configs/corpora.yaml`.
2. `rmb build <saga>` y luego `rmb inspect <saga>`: revisar número de capítulos/escenas y títulos.
3. Guiones: si `inspect` muestra pocas líneas de diálogo, revisar los umbrales `screenplay_layout`
   (coordenadas x en puntos PDF) con un `layout:` específico en la fuente.
4. `rmb check-questions <saga>`: ninguna evidencia debe quedar con 0 chunks relevantes.
