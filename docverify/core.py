"""Local NLP extraction with evidence and deterministic monetary checks."""
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
from io import BytesIO
import re
import joblib
from pydantic import BaseModel, Field, ConfigDict
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['document_id', 'po_id', 'vendor', 'date', 'currency', 'subtotal', 'tax', 'total']
LABELS = {
 'document_id':['Invoice number','Document ID','Invoice reference','Bill number'],
 'po_id':['Purchase order','PO reference','Order ID','PO number'],
 'vendor':['Vendor','Supplier','Issued by','Seller'],
 'date':['Invoice date','Document date','Issued on','Billing date'],
 'currency':['Currency','Currency code','Settlement currency','Payment currency'],
 'subtotal':['Subtotal','Net amount','Amount before tax','Net subtotal'],
 'tax':['Tax','Tax amount','GST amount','Total tax'],
 'total':['Grand total','Amount due','Invoice total','Payable total'],
}

class Item(BaseModel):
    model_config = ConfigDict(extra='forbid')
    sku: str = Field(min_length=1)
    description: str
    quantity: Decimal = Field(gt=0, allow_inf_nan=False)
    unit_price: Decimal = Field(ge=0, allow_inf_nan=False)
    amount: Decimal = Field(ge=0, allow_inf_nan=False)

class Document(BaseModel):
    model_config = ConfigDict(extra='forbid')
    document_id: str | None = None
    po_id: str | None = None
    vendor: str | None = None
    date: str | None = None
    currency: str | None = None
    subtotal: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)
    tax: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)
    total: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)
    items: list[Item] = Field(default_factory=list)

def money(value):
    return Decimal(str(value).replace(',', '').strip()).quantize(Decimal('.01'), rounding=ROUND_HALF_UP)

def read_document(content: bytes, filename: str, enable_ocr=False):
    if len(content) > 10*1024*1024:
        raise ValueError('Maximum document size is 10 MB.')
    if enable_ocr and filename.lower().endswith(('.png','.jpg','.jpeg')):
        from .ocr import ocr_document
        return ocr_document(content,filename)
    if filename.lower().endswith('.pdf'):
        reader = PdfReader(BytesIO(content))
        if reader.is_encrypted:
            raise ValueError('Encrypted PDFs are not supported.')
        if len(reader.pages) > 20:
            raise ValueError('Maximum 20 pages per document.')
        pages = [p.extract_text() or '' for p in reader.pages]
        if any(not p.strip() for p in pages):
            if enable_ocr:
                from .ocr import ocr_document
                return ocr_document(content,filename)
            raise ValueError('A page has no extractable text. Run OCR first and upload the resulting searchable PDF or text.')
    elif filename.lower().endswith('.txt'):
        pages = [content.decode('utf-8-sig')]
    else:
        raise ValueError('Upload a text-based PDF or UTF-8 TXT file.')
    if sum(map(len,pages)) > 150000:
        raise ValueError('Extracted text exceeds 150,000 characters.')
    return pages

def extract(pages, method='ml', model=None):
    if method not in ('rules', 'ml'):
        raise ValueError('Unknown extraction method.')
    if model is None and method=='ml':
        path = ROOT/'artifacts/extractor.joblib'
        if not path.exists():
            raise ValueError('Run python -m docverify.train first.')
        model = joblib.load(path)
    values, evidence, warnings, items = {}, {}, [], []
    exact = {labels[0].lower(): field for field,labels in LABELS.items()}
    for page, text in enumerate(pages, 1):
        for line_no, raw in enumerate(text.splitlines(), 1):
            line = raw.strip()
            location = dict(page=page, line=line_no, text=line)
            if line.startswith('ITEM'):
                parts = re.split(r'\s*[|;]\s*', line)
                if len(parts)!=6:
                    warnings.append(f'Malformed item on page {page}, line {line_no}.')
                    continue
                try:
                    item = Item(sku=parts[1], description=parts[2], quantity=parts[3], unit_price=money(parts[4]), amount=money(parts[5]))
                    evidence[f'items.{len(items)}'] = location
                    items.append(item)
                except Exception:
                    warnings.append(f'Invalid item values on page {page}, line {line_no}.')
                continue
            match = re.match(r'^([^:=]{2,60})\s*[:=]\s*(.+)$', line)
            if not match:
                continue
            label, value = match.groups()
            confidence = 1.
            if method=='rules':
                field = exact.get(label.lower().strip())
            else:
                probabilities = model.predict_proba([label])[0]
                index = probabilities.argmax()
                confidence = float(probabilities[index])
                field = str(model.classes_[index])
                if confidence < .55 or field=='other':
                    warnings.append(f'Unrecognized field label on page {page}, line {line_no}: {label}')
                    continue
            if field is None:
                continue
            if field in values:
                warnings.append(f'Duplicate field {field}; review source documents.')
                continue
            try:
                values[field] = money(value) if field in ['subtotal','tax','total'] else value.strip()
                evidence[field] = dict(**location, confidence=round(confidence,4))
            except Exception:
                warnings.append(f'Invalid monetary field {field}.')
    doc = Document(**values, items=items)
    for field in FIELDS:
        if getattr(doc, field) is None:
            warnings.append(f'Missing {field}.')
    if not items:
        warnings.append('No structured line items found.')
    if doc.date:
        from datetime import date
        try: date.fromisoformat(doc.date)
        except ValueError: warnings.append('Date must use YYYY-MM-DD format.')
    if doc.currency and not re.fullmatch('[A-Z]{3}',doc.currency):
        warnings.append('Currency must be a three-letter uppercase code.')
    return {'document': doc.model_dump(mode='json'), 'evidence': evidence, 'warnings': warnings}

def verify(invoice: Document, purchase_order: Document, warnings=None):
    issues = []
    def add(code, message, field):
        issues.append(dict(code=code, message=message, field=field))
    for warning in warnings or []:
        add('EXTRACTION_REVIEW', warning, 'source')
    for name, doc in [('invoice', invoice), ('purchase_order', purchase_order)]:
        for field in FIELDS:
            if getattr(doc, field) is None:
                add('MISSING_FIELD', f'{name}: missing {field}', f'{name}.{field}')
        seen = set()
        for i,item in enumerate(doc.items):
            if item.sku in seen:
                add('DUPLICATE_SKU', f'{name}: duplicate SKU {item.sku}', f'{name}.items.{i}')
            seen.add(item.sku)
            if abs(money(item.quantity*item.unit_price)-item.amount)>Decimal('.01'):
                add('LINE_TOTAL', f'{name}: incorrect line total for {item.sku}', f'{name}.items.{i}')
        if not doc.items:
            add('MISSING_ITEMS', f'{name}: no line items', f'{name}.items')
        if doc.subtotal is not None and abs(sum((x.amount for x in doc.items), Decimal(0))-doc.subtotal)>Decimal('.01'):
            add('SUBTOTAL', f'{name}: subtotal differs from line items', f'{name}.subtotal')
        if all(getattr(doc,f) is not None for f in ['subtotal','tax','total']):
            if abs(doc.subtotal+doc.tax-doc.total)>Decimal('.01'):
                add('TOTAL', f'{name}: total differs from subtotal plus tax', f'{name}.total')
    for field in ['po_id','vendor','currency']:
        a,b = getattr(invoice,field), getattr(purchase_order,field)
        if a is not None and b is not None and a.casefold()!=b.casefold():
            add('HEADER_MISMATCH', f'{field} differs between documents', field)
    expected = {x.sku:x for x in purchase_order.items}
    actual = {x.sku:x for x in invoice.items}
    for sku in sorted(set(expected)|set(actual)):
        if sku not in expected or sku not in actual:
            add('UNMATCHED_ITEM', f'{sku} is not present in both documents', 'items')
        else:
            for field in ['quantity','unit_price']:
                if getattr(expected[sku],field)!=getattr(actual[sku],field):
                    add('ITEM_MISMATCH', f'{sku}: {field} differs from purchase order', 'items')
    return {'status': 'needs_review' if issues else 'checks_passed', 'issues': issues,
            'note':'Checks passed is not payment approval; human review remains required.'}

def process(invoice_pages, po_pages, method='ml'):
    inv, po = extract(invoice_pages,method), extract(po_pages,method)
    result = verify(Document(**inv['document']), Document(**po['document']), inv['warnings']+po['warnings'])
    return dict(invoice=inv, purchase_order=po, verification=result)
