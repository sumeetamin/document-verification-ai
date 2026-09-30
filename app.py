import json
import pandas as pd
import streamlit as st
from docverify.core import ROOT, read_document, process, Document, verify

st.set_page_config(page_title='Document Verification AI',page_icon='📄',layout='wide')
st.caption('DOCUMENT LAB / APPLIED AI')
st.title('Document Verification AI')
st.write('Extract the evidence. Match the order. Review exceptions.')
st.info('Local NLP model · no external API · text PDFs and structured TXT · synthetic demo documents')
mode=st.sidebar.radio('Document source',['Built-in demo','Upload documents'])
method=st.sidebar.selectbox('Extraction method',['ml','rules'])
enable_ocr=st.sidebar.checkbox('Enable local OCR for scans',value=False)
st.sidebar.caption('OCR requires requirements-ocr.txt; first use downloads public model weights. Documents stay local.')
st.sidebar.caption('ML: trained character n-gram field classifier. Rules: exact-label baseline.')
if not (ROOT/'artifacts/extractor.joblib').exists():
    st.info('Run `python -m docverify.train` first.'); st.stop()
if mode=='Built-in demo':
    inv=(ROOT/'data/invoice.txt').read_text()
    po=(ROOT/'data/purchase_order.txt').read_text()
    a,b=st.columns(2)
    with a: st.download_button('Download sample invoice PDF',(ROOT/'data/invoice.pdf').read_bytes(),'invoice.pdf')
    with b: st.download_button('Download sample purchase order PDF',(ROOT/'data/purchase_order.pdf').read_bytes(),'purchase_order.pdf')
    invoice_pages,po_pages=[inv],[po]
else:
    a,b=st.columns(2)
    with a: f1=st.file_uploader('Invoice',type=['pdf','txt','png','jpg','jpeg'])
    with b: f2=st.file_uploader('Purchase order',type=['pdf','txt','png','jpg','jpeg'])
    if not f1 or not f2: st.stop()
    try:
        invoice_pages=read_document(f1.getvalue(),f1.name,enable_ocr); po_pages=read_document(f2.getvalue(),f2.name,enable_ocr)
    except Exception as e: st.error(str(e)); st.stop()
try: result=process(invoice_pages,po_pages,method)
except Exception as e: st.error(str(e)); st.stop()
tabs=st.tabs(['Review desk','Evidence','Correct & export','Evaluation'])
with tabs[0]:
    a,b,c=st.columns(3)
    a.metric('Status',result['verification']['status'].replace('_',' ').title())
    b.metric('Exceptions',len(result['verification']['issues']))
    c.metric('Invoice items',len(result['invoice']['document']['items']))
    st.dataframe(pd.DataFrame(result['verification']['issues']),hide_index=True,width='stretch')
    st.subheader('Extracted invoice')
    st.json(result['invoice']['document'])
with tabs[1]:
    for name in ['invoice','purchase_order']:
        st.subheader(name.replace('_',' ').title())
        for warning in result[name]['warnings']: st.warning(warning)
        st.dataframe(pd.DataFrame([dict(field=k,**v) for k,v in result[name]['evidence'].items()]),hide_index=True,width='stretch')
    with st.expander('Source text'): st.code('\n'.join(invoice_pages))
with tabs[2]:
    st.write('Correct extracted fields as JSON, then validate again. Original extraction warnings remain visible in the Evidence tab and exported record.')
    edited=st.text_area('Invoice JSON',json.dumps(result['invoice']['document'],indent=2),height=350)
    edited_po=st.text_area('Purchase order JSON',json.dumps(result['purchase_order']['document'],indent=2),height=250)
    if st.button('Validate corrections'):
        try:
            invoice=Document.model_validate_json(edited); order=Document.model_validate_json(edited_po)
            checked=verify(invoice,order)
            audit=dict(original=result, corrected_invoice=invoice.model_dump(mode='json'), corrected_purchase_order=order.model_dump(mode='json'),verification=checked)
            st.json(checked)
            st.download_button('Export reviewed record',json.dumps(audit,indent=2),'review.json','application/json')
        except Exception as e: st.error(str(e))
    st.download_button('Export original result',json.dumps(result,indent=2),'verification.json','application/json')
with tabs[3]:
    report=json.loads((ROOT/'reports/evaluation.json').read_text())
    st.write(report['dataset']); st.dataframe(pd.DataFrame(report['results']),hide_index=True)
    for limitation in report['limitations']: st.write('• '+limitation)
st.caption('This tool checks consistency; it does not approve payments or establish document authenticity.')
