"""Train an offline character n-gram field classifier; evaluate held-out label styles."""
import json
import random
import time
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from reportlab.pdfgen import canvas
from .core import ROOT, LABELS, FIELDS, extract, Document, verify

def synthetic_text(index=0, style=0, mismatch=False):
    rng = random.Random(index)
    quantity = rng.randint(2,20)
    price = rng.randint(10,90)
    subtotal = quantity*price
    values = dict(document_id=f'INV-{index:04d}', po_id=f'PO-{index:04d}', vendor=f'Demo Supplier {index%9}',
                  date='2026-01-15', currency='INR', subtotal=f'{subtotal:.2f}',tax=f'{subtotal*.18:.2f}',total=f'{subtotal*1.18:.2f}')
    lines = [f'{LABELS[field][style]}: {values[field]}' for field in FIELDS]
    lines.append(f'ITEM | SKU-{index%7} | Office supplies | {quantity+(1 if mismatch else 0)} | {price:.2f} | {subtotal:.2f}')
    return '\n'.join(lines), values

def main():
    for folder in ['artifacts','data','reports']:
        (ROOT/folder).mkdir(exist_ok=True)
    texts, labels = [], []
    # Train on first three label styles; style 4 is held out.
    for field, synonyms in LABELS.items():
        for label in synonyms[:3]:
            for variant in [label, label.upper(), label.lower(), label+' ']:
                texts.append(variant); labels.append(field)
    for label in ['Notes','Address','Telephone','Bank details','Payment terms','Customer','Discount','Shipping','Instructions']:
        for _ in range(4):
            texts.append(label); labels.append('other')
    model = make_pipeline(TfidfVectorizer(analyzer='char_wb', ngram_range=(2,5)), LogisticRegression(C=20, max_iter=1500, random_state=42))
    model.fit(texts,labels)
    joblib.dump(model, ROOT/'artifacts/extractor.joblib')
    results = []
    for style,method in [(0,'rules'),(0,'ml'),(3,'rules'),(3,'ml')]:
        correct, total, detected, false_alarm, times = 0,0,0,0,[]
        for i in range(80):
            text, expected = synthetic_text(i, style)
            start = time.perf_counter()
            parsed = extract([text], method, model)
            times.append((time.perf_counter()-start)*1000)
            for field in FIELDS:
                total += 1
                correct += parsed['document'][field]==expected[field]
            # End-to-end mismatch evaluation includes abstentions; report separately.
            po_text,_ = synthetic_text(i,style)
            bad_text,_ = synthetic_text(i,style,mismatch=True)
            po = Document(**extract([po_text],method,model)['document'])
            good = verify(Document(**parsed['document']),po,parsed['warnings'])
            bad_parsed=extract([bad_text],method,model)
            bad = verify(Document(**bad_parsed['document']),po,bad_parsed['warnings'])
            detected += any(x['code']=='ITEM_MISMATCH' for x in bad['issues'])
            false_alarm += good['status']=='needs_review'
        results.append(dict(method=method, layout='supported labels' if style==0 else 'unseen labels', documents=80, field_exact_match=correct/total,
                            quantity_mismatch_detection=detected/80, clean_document_review_rate=false_alarm/80,
                            mean_extraction_ms=sum(times)/len(times)))
    for name, mismatch in [('invoice',True),('purchase_order',False)]:
        text,_=synthetic_text(42,0,mismatch)
        (ROOT/f'data/{name}.txt').write_text(text)
        pdf=canvas.Canvas(str(ROOT/f'data/{name}.pdf'))
        pdf.setFont('Helvetica-Bold',18); pdf.drawString(50,790,'DEMO / '+name.replace('_',' ').upper())
        pdf.setFont('Helvetica',11)
        for i,line in enumerate(text.splitlines()): pdf.drawString(50,750-i*25,line)
        pdf.save()
    report=dict(dataset='80 synthetic documents with a held-out label style; not a real-world benchmark',
                training='Character TF-IDF + logistic regression on three field-label styles',
                results=results, limitations=['Optional RapidOCR for scans; OCR errors propagate to extraction.', 'Pipe/semicolon-delimited line items only.',
                'Small synthetic template split does not establish generalization to arbitrary invoices.', 'No external LLM or paid API required.'])
    (ROOT/'reports/evaluation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
