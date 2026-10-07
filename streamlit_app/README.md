# Fraud Detection System - Web Interface

Professional web interface for the fraud detection machine learning system.

## Local Development

### Prerequisites

- Python 3.8+
- Trained model (`models/random_forest_model.pkl`)
- Dataset (`data/creditcard.csv`)

### Installation

```bash
cd streamlit_app
pip install -r requirements.txt
```

### Run Locally

```bash
streamlit run Home.py
```

Opens at `http://localhost:8501`

## Deployment to Streamlit Cloud

### 1. Prepare Repository

Ensure your GitHub repository contains:
- `streamlit_app/` directory with all files
- `models/random_forest_model.pkl` (if < 100MB)
- `data/creditcard.csv` (if < 100MB)
- `.streamlit/config.toml` for custom theme

**Note:** If model/data files exceed 100MB, use Git LFS or download them in startup script.

### 2. Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set main file path: `streamlit_app/Home.py`
6. Click "Deploy"

### 3. Custom Domain Setup

**Option A: Streamlit Cloud Custom Domain (Requires Teams plan)**

1. Go to app settings on Streamlit Cloud
2. Navigate to "Domains"
3. Add your custom domain
4. Configure DNS records:
   ```
   Type: CNAME
   Name: www (or subdomain)
   Value: <your-app>.streamlit.app
   ```

**Option B: Self-Host with Custom Domain**

1. Deploy on your own server (AWS, DigitalOcean, etc.)
2. Run with:
   ```bash
   streamlit run Home.py --server.port 8501
   ```
3. Set up reverse proxy (Nginx):
   ```nginx
   server {
       listen 80;
       server_name yourdomain.com;
       
       location / {
           proxy_pass http://localhost:8501;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_set_header Host $host;
       }
   }
   ```
4. Add SSL with Let's Encrypt:
   ```bash
   sudo certbot --nginx -d yourdomain.com
   ```

### 4. Add Favicon

**Create favicon:**
1. Create a 32x32 or 64x64 PNG icon
2. Save as `streamlit_app/assets/favicon.ico`
3. Update `Home.py` page_config:
   ```python
   st.set_page_config(
       page_title="Fraud Detection System",
       page_icon="🛡️",  # Or path to .ico file
       layout="wide"
   )
   ```

**For custom favicon file:**
```python
st.set_page_config(
    page_icon="assets/favicon.ico"
)
```

### 5. Remove "Made with Streamlit" Tag

**Note:** On Streamlit Community Cloud (free tier), you cannot fully remove the "Made with Streamlit" tag. Options:

**Option A: Upgrade to Teams/Enterprise**
- Teams plan allows hiding the tag
- Settings → General → Hide "Made with Streamlit"

**Option B: Self-Host**
- Deploy on your own infrastructure
- Full control over branding
- No Streamlit attribution required

**Option C: Custom CSS (Partial)**
Add to `Home.py`:
```python
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
```

**Important:** Respect Streamlit's brand guidelines if using their free hosting.

### 6. Environment Variables

If you have sensitive config:

1. Create `.streamlit/secrets.toml` (NOT committed to git):
   ```toml
   [general]
   app_name = "Fraud Detection System"
   ```

2. Access in code:
   ```python
   import streamlit as st
   app_name = st.secrets["general"]["app_name"]
   ```

3. On Streamlit Cloud, add secrets in app settings

## Production Checklist

Before going live:

- [ ] Custom domain configured with SSL
- [ ] Favicon added
- [ ] Privacy Policy page complete
- [ ] Terms and Conditions page complete
- [ ] All documentation pages reviewed
- [ ] Real metrics from actual model (no fake numbers)
- [ ] Professional theme configured (no purple gradients)
- [ ] Removed any AI-generated placeholder content
- [ ] Tested all pages and features
- [ ] Performance tested with multiple users
- [ ] Error handling for missing files
- [ ] Mobile responsiveness checked
- [ ] All external links validated

## Project Structure

```
streamlit_app/
├── Home.py                          # Main prediction interface
├── pages/
│   ├── 1_Model_Performance.py       # Performance metrics
│   ├── 2_Drift_Monitoring.py        # Drift dashboard
│   ├── 3_Cost_Analysis.py           # Threshold optimization
│   ├── 4_Documentation.py           # Technical docs
│   ├── 5_Privacy_Policy.py          # Privacy policy
│   └── 6_Terms_Conditions.py        # Terms and conditions
├── .streamlit/
│   └── config.toml                  # Custom theme
├── assets/
│   └── favicon.ico                  # Site icon
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Custom Styling

### Clean, Professional Design Principles

**What we DO:**
- Clean white/gray backgrounds
- Standard blue accents
- Clear typography
- Data-focused visualizations
- Real metrics only

**What we DON'T do:**
- Purple gradients
- Pill-shaped buttons
- Fake metrics or reviews
- Vague hero text
- Emoji icons everywhere
- Over-the-top animations
- AI slop photos
- Cursor animations

### Theme Customization

Edit `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#1f77b4"              # Main color
backgroundColor = "#ffffff"            # Background
secondaryBackgroundColor = "#f0f2f6"  # Sidebar
textColor = "#262730"                 # Text
font = "sans serif"                   # Font family
```

## Performance Optimization

### For Large Files

If model/data files are large:

**Option 1: Git LFS**
```bash
git lfs install
git lfs track "*.pkl"
git lfs track "*.csv"
```

**Option 2: Download on Startup**

Create `streamlit_app/setup.sh`:
```bash
#!/bin/bash
if [ ! -f "models/random_forest_model.pkl" ]; then
    echo "Downloading model..."
    # Download from S3, Google Drive, etc.
fi
```

**Option 3: Cloud Storage**
```python
import boto3
s3 = boto3.client('s3')
s3.download_file('bucket', 'model.pkl', 'models/random_forest_model.pkl')
```

### Caching

Already implemented with `@st.cache_resource`:
```python
@st.cache_resource
def load_model():
    # Model loaded once and cached
    return model, scaler, df, metrics
```

## Troubleshooting

### Model Not Found

Ensure file paths are correct:
```python
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models/random_forest_model.pkl')
```

### Images Not Loading

Check relative paths:
```python
image_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results/figures/chart.png')
```

### Deployment Fails

Common issues:
- Missing `requirements.txt`
- Wrong Python version (use 3.8-3.11)
- Large files not in Git LFS
- Incorrect main file path

## Security

### Environment Variables

Never commit:
- API keys
- Database credentials
- Secret tokens

Use `.streamlit/secrets.toml` (gitignored)

### Input Validation

All user inputs are validated:
```python
amount = st.number_input("Amount", min_value=0.0, max_value=25000.0)
```

## Monitoring

### Application Logs

On Streamlit Cloud:
- View logs in app dashboard
- Monitor resource usage
- Check error rates

### Self-Hosted

Use systemd service:
```bash
sudo journalctl -u streamlit-app -f
```

## Support

For issues:
1. Check Documentation page
2. Review GitHub repository
3. Consult Streamlit docs: https://docs.streamlit.io

## License

Same as parent project (MIT or as specified)

## Academic Context

This is an educational project for AWS Student Builder Group AI/ML track.
Not intended for production use without proper validation.
