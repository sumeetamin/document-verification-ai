"""Optional local OCR. Model weights are downloaded by RapidOCR on first use."""
from functools import lru_cache
from io import BytesIO
import numpy as np

@lru_cache(maxsize=1)
def engine():
    try:
        from rapidocr import RapidOCR
    except ImportError as exc:
        raise ValueError('Install requirements-ocr.txt to enable local OCR.') from exc
    return RapidOCR()

def image_text(image):
    result=engine()(np.array(image.convert('RGB')))
    if not result.txts: raise ValueError('OCR found no readable text.')
    return '\n'.join(result.txts)

def ocr_document(content,filename):
    from PIL import Image
    if filename.lower().endswith('.pdf'):
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:
            raise ValueError('Install requirements-ocr.txt to render scanned PDFs.') from exc
        pdf=pdfium.PdfDocument(content)
        if len(pdf)>20: raise ValueError('Maximum 20 pages.')
        pages=[]
        for page in pdf:
            bitmap=page.render(scale=2)
            pages.append(image_text(bitmap.to_pil()))
            bitmap.close(); page.close()
        pdf.close()
        return pages
    image=Image.open(BytesIO(content))
    if image.width*image.height>20_000_000: raise ValueError('Maximum image size is 20 megapixels.')
    return [image_text(image)]
