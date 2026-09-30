# Measured results

Synthetic document benchmark; 80 documents per regime. Supported-label scores are smoke tests, not unseen-template generalization.

| Method | Label regime | Exact field match | Clean documents sent to review |
|---|---|---:|---:|
| rules | supported labels | 100.0% | 0.0% |
| ml | supported labels | 100.0% | 0.0% |
| rules | unseen labels | 0.0% | 100.0% |
| ml | unseen labels | 50.0% | 100.0% |

Both extractors detect the generated quantity mismatches. That result comes from the shared structured item parser and deterministic comparison, not from general document understanding. The learned extractor reaches only 50% exact field match on unseen label wording and requests review on all those documents. A broader real-document corpus and stronger extraction model are needed before claiming general invoice support.
