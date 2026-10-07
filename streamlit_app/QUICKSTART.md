# Quick Start Guide

Get the fraud detection web interface running in 5 minutes.

## Prerequisites

1. **Models trained:**
   ```bash
   cd /Users/sajivvaila/fraud_detection_project
   python3 day2_modeling.py    # Creates models/random_forest_model.pkl
   python3 day3_drift_monitoring.py  # Creates drift results
   ```

2. **Dataset available:**
   - `data/creditcard.csv` exists

3. **Python 3.8+** installed

## Installation

```bash
cd /Users/sajivvaila/fraud_detection_project/streamlit_app
pip3 install -r requirements.txt
```

## Run Locally

```bash
streamlit run Home.py
```

**Opens automatically at:** http://localhost:8501

## Navigate the App

### Pages Available:

1. **Home** - Live fraud prediction
   - Test real transactions from dataset
   - Or enter custom amounts
   - See predictions in real-time

2. **Model Performance** - Metrics and evaluation
   - PR-AUC, ROC-AUC, F1 scores
   - Strategy comparison
   - Performance curves

3. **Drift Monitoring** - Real-time monitoring
   - System status (GREEN/AMBER/RED)
   - Window-by-window analysis
   - PSI, KS test, score drift

4. **Cost Analysis** - Business optimization
   - Cost model explanation
   - Threshold comparison
   - Savings calculation

5. **Documentation** - Technical details
   - Project overview
   - Model architecture
   - Pipeline explanation

6. **Privacy Policy** - Data handling
   - What we collect (nothing)
   - How data is used
   - Your rights

7. **Terms and Conditions** - Legal terms
   - Acceptable use
   - Limitations
   - Disclaimers

## Test the System

### Quick Test:

1. Go to **Home** page
2. Select "Test Real Transactions"
3. Click **Predict**
4. See real fraud detection in action

### Interactive Test:

1. Go to **Home** page  
2. Select "Custom Amount"
3. Enter: `$250.50`
4. Click **Predict**
5. See fraud probability

## Stop the App

Press `Ctrl+C` in terminal

## Common Issues

### Port Already in Use

```bash
# Kill existing Streamlit process
pkill -f streamlit

# Or use different port
streamlit run Home.py --server.port 8502
```

### Model Not Found

```bash
# Ensure you're in correct directory
pwd  # Should show: /Users/sajivvaila/fraud_detection_project/streamlit_app

# Check model exists
ls -lh ../models/random_forest_model.pkl
```

### Import Errors

```bash
# Reinstall dependencies
pip3 install -r requirements.txt --force-reinstall
```

### Images Not Loading

```bash
# Check figures exist
ls -lh ../results/figures/

# If missing, run drift monitoring
cd ..
python3 day3_drift_monitoring.py
```

## Customization

### Change Theme

Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#1f77b4"  # Your brand color
```

### Change Port

```bash
streamlit run Home.py --server.port 9000
```

### Add Custom Content

Edit individual page files in `pages/`

## Next Steps

1. **Test locally** - Make sure everything works
2. **Review content** - Check all pages
3. **Deploy** - Follow DEPLOYMENT.md guide
4. **Custom domain** - Setup your domain
5. **Launch** - Share with world

## Development Mode

Auto-reload on file changes:
```bash
streamlit run Home.py
# Edit any .py file
# Browser auto-refreshes
```

## Production Mode

```bash
streamlit run Home.py --server.headless true
```

## View Logs

Streamlit logs appear in terminal:
```
  You can now view your Streamlit app in your browser.
  
  Local URL: http://localhost:8501
  Network URL: http://192.168.1.x:8501
```

## Keyboard Shortcuts

While app is running:

- `c` - Clear cache
- `r` - Rerun script
- `Ctrl+C` - Stop server

## Mobile Testing

1. Find your local IP:
   ```bash
   ifconfig | grep "inet "
   # Or
   ipconfig getifaddr en0
   ```

2. On mobile browser, visit:
   ```
   http://YOUR_LOCAL_IP:8501
   ```

3. Both devices must be on same WiFi

## Performance Tips

1. **Cache is enabled** - First load is slow, subsequent fast
2. **Model loads once** - Thanks to `@st.cache_resource`
3. **Images cached** - Loaded only when needed
4. **Clear cache** - Press `c` key if stale data

## Security Note

This runs locally on your machine. To make it public:

1. **Option A:** Deploy to Streamlit Cloud (free)
2. **Option B:** Self-host with proper security

**DO NOT** expose local server to internet without:
- HTTPS/SSL
- Authentication
- Firewall
- Rate limiting

## Ready to Deploy?

See **DEPLOYMENT.md** for full production deployment guide.

## Support

For issues:
1. Check this guide
2. Review README.md
3. Check DEPLOYMENT.md
4. Review individual page code

---

**Enjoy your professional fraud detection web interface!**
