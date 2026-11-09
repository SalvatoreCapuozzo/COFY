# Confirmatory results

Generated 2026-11-04 08:06 by `tools/confirm_analysis.py` (pre-registered in `analysis/CONFIRMATORY_PREREG.md`). Seeds 100-109 (n = 10).

## Confirmation datasets

### sync -- `confirm_v2sync` (bn+head)

**Result: passes 90 %** (fixed sequence 85 %: pass, 90 %: pass)

| dataset | n | Fed:Cent mean | sd | 95 % CI | mAP50 fed / cent | abs. loss mean (upper 95 %) | mAP50-95 Fed:Cent (CI-low) |
|---|---|---|---|---|---|---|---|
| kfm12 | 10 | 98.6 % | 2.7 | 96.7-100.6 % | 0.724 / 0.734 | +0.010 (+0.025) | 98.6 % (96.0 %) |
| wbc | 10 | 105.4 % | 9.4 | 98.7-112.1 % | 0.527 / 0.502 | -0.025 (+0.007) | 105.3 % (98.7 %) |
| chestx | 10 | 97.8 % | 3.8 | 95.0-100.5 % | 0.305 / 0.312 | +0.007 (+0.016) | 96.2 % (92.0 %) |

### async -- `confirm_v2async` (acal_refresh)

**Result: fails 80 %** (fixed sequence 80 %: fail)

| dataset | n | Fed:Cent mean | sd | 95 % CI | mAP50 fed / cent | abs. loss mean (upper 95 %) | mAP50-95 Fed:Cent (CI-low) |
|---|---|---|---|---|---|---|---|
| kfm12 | 10 | 93.6 % | 2.0 | 92.1-95.1 % | 0.687 / 0.734 | +0.047 (+0.058) | 93.2 % (91.4 %) |
| wbc | 10 | 130.8 % | 12.3 | 122.0-139.6 % | 0.654 / 0.502 | -0.152 (-0.114) | 129.8 % (121.1 %) |
| chestx | 10 | 83.2 % | 4.9 | 79.7-86.7 % | 0.259 / 0.312 | +0.053 (+0.065) | 83.5 % (79.7 %) |

## Held-out datasets

### sync -- `confirm_v2sync` (bn+head)

**Result: passes 90 %** (fixed sequence 85 %: pass, 90 %: pass)

| dataset | n | Fed:Cent mean | sd | 95 % CI | mAP50 fed / cent | abs. loss mean (upper 95 %) | mAP50-95 Fed:Cent (CI-low) |
|---|---|---|---|---|---|---|---|
| dentex | 10 | 98.8 % | 3.7 | 96.2-101.4 % | 0.947 / 0.959 | +0.012 (+0.037) | 98.6 % (95.6 %) |

### async -- `confirm_v2async` (acal_refresh)

**Result: passes 85 %** (fixed sequence 80 %: pass, 85 %: pass)

| dataset | n | Fed:Cent mean | sd | 95 % CI | mAP50 fed / cent | abs. loss mean (upper 95 %) | mAP50-95 Fed:Cent (CI-low) |
|---|---|---|---|---|---|---|---|
| dentex | 10 | 99.7 % | 0.5 | 99.4-100.1 % | 0.956 / 0.959 | +0.003 (+0.006) | 101.4 % (100.6 %) |

## All four use cases datasets

### sync -- `confirm_v2sync` (bn+head)

**Result: passes 90 %** (fixed sequence 85 %: pass, 90 %: pass)

| dataset | n | Fed:Cent mean | sd | 95 % CI | mAP50 fed / cent | abs. loss mean (upper 95 %) | mAP50-95 Fed:Cent (CI-low) |
|---|---|---|---|---|---|---|---|
| kfm12 | 10 | 98.6 % | 2.7 | 96.7-100.6 % | 0.724 / 0.734 | +0.010 (+0.025) | 98.6 % (96.0 %) |
| wbc | 10 | 105.4 % | 9.4 | 98.7-112.1 % | 0.527 / 0.502 | -0.025 (+0.007) | 105.3 % (98.7 %) |
| chestx | 10 | 97.8 % | 3.8 | 95.0-100.5 % | 0.305 / 0.312 | +0.007 (+0.016) | 96.2 % (92.0 %) |
| dentex | 10 | 98.8 % | 3.7 | 96.2-101.4 % | 0.947 / 0.959 | +0.012 (+0.037) | 98.6 % (95.6 %) |

### async -- `confirm_v2async` (acal_refresh)

**Result: fails 80 %** (fixed sequence 80 %: fail)

| dataset | n | Fed:Cent mean | sd | 95 % CI | mAP50 fed / cent | abs. loss mean (upper 95 %) | mAP50-95 Fed:Cent (CI-low) |
|---|---|---|---|---|---|---|---|
| kfm12 | 10 | 93.6 % | 2.0 | 92.1-95.1 % | 0.687 / 0.734 | +0.047 (+0.058) | 93.2 % (91.4 %) |
| wbc | 10 | 130.8 % | 12.3 | 122.0-139.6 % | 0.654 / 0.502 | -0.152 (-0.114) | 129.8 % (121.1 %) |
| chestx | 10 | 83.2 % | 4.9 | 79.7-86.7 % | 0.259 / 0.312 | +0.053 (+0.065) | 83.5 % (79.7 %) |
| dentex | 10 | 99.7 % | 0.5 | 99.4-100.1 % | 0.956 / 0.959 | +0.003 (+0.006) | 101.4 % (100.6 %) |

