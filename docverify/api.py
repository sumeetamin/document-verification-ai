from fastapi import FastAPI, UploadFile, HTTPException
from .core import read_document, process
app = FastAPI(title='Document Verification AI', version='1.0.0')

@app.get('/health')
def health(): return {'status':'ok'}

@app.post('/verify')
async def verify_files(invoice: UploadFile, purchase_order: UploadFile, enable_ocr: bool=False):
    try:
        a=read_document(await invoice.read(10*1024*1024+1), invoice.filename or '',enable_ocr)
        b=read_document(await purchase_order.read(10*1024*1024+1), purchase_order.filename or '',enable_ocr)
        return process(a,b)
    except (ValueError, OSError) as exc:
        raise HTTPException(422,str(exc)) from exc
    except Exception as exc:
        raise HTTPException(422,'Unable to parse document. Use an unencrypted text PDF or UTF-8 text.') from exc
