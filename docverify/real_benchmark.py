"""Evaluate generic field-role classification on downloaded FUNSD annotations.

FUNSD has real scanned forms with question, answer, header and other labels. These
are deliberately kept separate from invoice/PO verification: FUNSD does not contain
paired purchase orders or enough structured financial fields to validate totals.
"""
import argparse, json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score
from .core import ROOT

VALID={'question','answer','header','other'}

def rows(folder):
    values=[]
    for path in Path(folder).rglob('*.json'):
        payload=json.loads(path.read_text(encoding='utf-8'))
        for item in payload.get('form',[]):
            label=item.get('label')
            text=item.get('text','').strip()
            if label in VALID and text: values.append((text,label))
    if len(values)<20: raise ValueError('Expected FUNSD annotation JSON files with at least 20 labelled form fields.')
    return values

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--train-dir',required=True,help='FUNSD training_data/annotations directory')
    parser.add_argument('--test-dir',required=True,help='FUNSD testing_data/annotations directory')
    args=parser.parse_args()
    train,test=rows(args.train_dir),rows(args.test_dir)
    model=make_pipeline(TfidfVectorizer(analyzer='char_wb',ngram_range=(2,5)),LogisticRegression(max_iter=1500,random_state=42))
    model.fit([x[0] for x in train],[x[1] for x in train])
    predicted=model.predict([x[0] for x in test])
    report={'dataset':'FUNSD real scanned-form annotations', 'source':'https://guillaumejaume.github.io/FUNSD/',
            'task':'generic form field-role classification; not invoice or purchase-order verification',
            'train_fields':len(train),'test_fields':len(test),
            'accuracy':float(accuracy_score([x[1] for x in test],predicted)),
            'macro_f1':float(f1_score([x[1] for x in test],predicted,average='macro'))}
    (ROOT/'reports/funsd_evaluation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
