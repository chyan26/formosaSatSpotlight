# FormosaSatSpotlight

Notebook-based pipeline for converting raw TIF/TIFF astronomy images to FITS, stacking PAN pairs, and measuring PSF/FWHM statistics.

## What it does

- Converts raw images in `raw_data/` into FITS files.
- Aligns and stacks `PAN1` / `PAN2` pairs in `ncu-spots.ipynb`.
- Generates a PDF report with image triptychs and a final PSF distribution chart.

## Project layout

- `ncu-spots.ipynb` - main workflow notebook
- `ncu_spots_utils.py` - reusable helper functions (image I/O, alignment, stacking, PSF fitting, plotting)
- `raw_data/` - input images
- `fits_outputs/` - converted FITS files
- `fits_stacked/` - stacked FITS products and PDF report

## Run

1. Open `ncu-spots.ipynb`.
2. Run the notebook cells in order.
3. Review the outputs in `fits_outputs/` and `fits_stacked/`.

## Notes

- The notebook currently uses a correlation-based alignment workflow.
- The final PDF includes per-pair diagnostics and an outlier-cleaned PSF histogram.