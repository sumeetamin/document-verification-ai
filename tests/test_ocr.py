import importlib.util
import pytest
from docverify.core import ROOT,read_document,extract

@pytest.mark.skipif(importlib.util.find_spec('rapidocr') is None,reason='Optional OCR dependencies not installed')
def test_clean_synthetic_scan():
    path=ROOT/'data/scanned_invoice.png'
    if not path.exists(): pytest.skip('Optional scan fixture is absent')
    pages=read_document(path.read_bytes(),path.name,enable_ocr=True)
    parsed=extract(pages)
    assert parsed['document']['document_id']=='INV-0042'
    assert parsed['document']['total']=='76.70'

def test_scan_without_ocr_rejected():
    with pytest.raises(ValueError): read_document(b'not an image','image.png')
