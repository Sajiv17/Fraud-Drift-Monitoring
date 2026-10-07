# Dataset: Credit Card Fraud Detection

## Source

**Kaggle:** [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

## Download Instructions

### Option 1: Kaggle Website (Recommended)

1. Go to https://www.kaggle.com/mlg-ulb/creditcardfraud
2. Click "Download" button (requires Kaggle account - free)
3. Extract `creditcard.csv` from the downloaded ZIP
4. Place `creditcard.csv` in this `data/` directory

### Option 2: Kaggle CLI

```bash
# Install Kaggle CLI
pip install kaggle

# Setup API credentials (get from kaggle.com/USERNAME/account)
# Place kaggle.json in ~/.kaggle/

# Download dataset
kaggle datasets download -d mlg-ulb/creditcardfraud

# Extract
unzip creditcardfraud.zip

# Move to data folder
mv creditcard.csv data/
```

## Dataset Details

**File:** `creditcard.csv`

**Size:** ~150 MB (uncompressed)

**Rows:** 284,807 transactions

**Columns:** 31
- `Time`: Seconds elapsed from first transaction
- `V1-V28`: PCA-transformed features (anonymized)
- `Amount`: Transaction amount
- `Class`: 0 = Legitimate, 1 = Fraud

**Class Distribution:**
- Legitimate: 284,315 (99.827%)
- Fraud: 492 (0.173%)

**Imbalance Ratio:** 577:1 (extreme!)

## Why This Dataset?

1. **Real-world imbalance:** Reflects actual fraud rates
2. **Anonymous features:** Protects customer privacy but still trainable
3. **Time component:** Enables time-based splitting and drift detection
4. **Well-documented:** Widely used in ML education and research
5. **Sufficient size:** 284K transactions for meaningful train/test split

## Privacy Note

The dataset contains only PCA-transformed features (V1-V28) to protect cardholder privacy. Original feature meanings are not available.

## Citation

Machine Learning Group - ULB (Université Libre de Bruxelles)  
Dataset collected and analyzed during a research collaboration of Worldline and the Machine Learning Group.

## Verify Your Download

After downloading, verify the file:

```bash
# Check file exists
ls -lh data/creditcard.csv

# Should show: ~150M creditcard.csv

# Check row count
wc -l data/creditcard.csv

# Should show: 284808 (including header)
```

## Troubleshooting

**Error: File not found**
- Make sure `creditcard.csv` is in the `data/` directory
- Check filename (case-sensitive!)

**Error: Kaggle authentication**
- Create free Kaggle account at kaggle.com
- Get API token from kaggle.com/USERNAME/account
- Place `kaggle.json` in `~/.kaggle/`

**Large file warning**
- File is ~150MB uncompressed
- Ensure sufficient disk space (~500MB total for project)
- Dataset is included in `.gitignore` (won't be committed to Git)

## Dataset License

Database Contents License (DbCL) v1.0  
https://opendatacommons.org/licenses/dbcl/1-0/
