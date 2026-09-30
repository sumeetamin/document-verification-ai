from decimal import Decimal
import pytest
from fastapi.testclient import TestClient
from docverify.core import Document, extract, verify, read_document, ROOT
from docverify.train import synthetic_text
from docverify.api import app

def test_arithmetic_and_matching():
    good,_=synthetic_text(1,0)
    bad,_=synthetic_text(1,0,True)
    po=Document(**extract([good],'rules')['document'])
    inv=Document(**extract([bad],'rules')['document'])
    assert verify(po,po)['status']=='checks_passed'
    codes={x['code'] for x in verify(inv,po)['issues']}
    assert {'LINE_TOTAL','ITEM_MISMATCH'}<=codes

def test_missing_duplicate_and_evidence():
    text,_=synthetic_text(2,0)
    parsed=extract([text+'\nGrand total: 0.00'],'rules')
    assert any('Duplicate' in x for x in parsed['warnings'])
    assert parsed['evidence']['vendor']['page']==1
    assert verify(Document(),Document())['status']=='needs_review'

def test_pdf_and_api():
    content=(ROOT/'data/invoice.pdf').read_bytes()
    assert 'Invoice number' in read_document(content,'invoice.pdf')[0]
    client=TestClient(app)
    result=client.post('/verify',files={'invoice':('invoice.pdf',content,'application/pdf'),
                       'purchase_order':('po.pdf',(ROOT/'data/purchase_order.pdf').read_bytes(),'application/pdf')})
    assert result.status_code==200
    assert result.json()['verification']['status']=='needs_review'
    bad=client.post('/verify',files={'invoice':('x.exe',b'bad'),'purchase_order':('x.txt',b'bad')})
    assert bad.status_code==422

def test_schema_rejects_invalid_money_and_unknown_fields():
    with pytest.raises(ValueError): Document(total='NaN')
    with pytest.raises(ValueError): Document(total=-1)
    with pytest.raises(ValueError): Document(unexpected='x')

def test_unknown_label_does_not_silently_pass():
    text,_=synthetic_text(3,0)
    parsed=extract([text+'\nUnexpected magic: 900'],'ml')
    assert parsed['warnings']
