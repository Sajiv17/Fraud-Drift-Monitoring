# Deployment Guide

Complete guide for deploying the fraud detection web interface professionally.

## Pre-Deployment Checklist

### 1. Code Quality
- [ ] All pages tested locally
- [ ] No hardcoded secrets or credentials
- [ ] Error handling for missing files
- [ ] Real metrics only (no fake data)
- [ ] Professional copy (no AI slop)

### 2. Legal Pages
- [ ] Privacy Policy complete and accurate
- [ ] Terms and Conditions complete and accurate
- [ ] Contact information updated
- [ ] Last updated dates current

### 3. Branding
- [ ] Custom favicon created (32x32 or 64x64 PNG)
- [ ] Professional color scheme (no purple gradients)
- [ ] Clean typography
- [ ] Mobile responsive

### 4. Performance
- [ ] Model file optimized (<100MB or using Git LFS)
- [ ] Caching implemented
- [ ] Images optimized
- [ ] Load times acceptable

## Deployment Options

### Option 1: Streamlit Community Cloud (Free)

**Pros:**
- Free hosting
- Easy deployment
- Automatic updates from GitHub
- SSL included

**Cons:**
- "Made with Streamlit" attribution (cannot remove on free tier)
- Limited resources
- No custom domain on free tier

**Steps:**

1. **Push to GitHub:**
   ```bash
   cd fraud_detection_project
   git add streamlit_app/
   git commit -m "Add Streamlit web interface"
   git push origin main
   ```

2. **Deploy:**
   - Go to https://share.streamlit.io/
   - Sign in with GitHub
   - Click "New app"
   - Repository: `your-username/fraud_detection_project`
   - Branch: `main`
   - Main file: `streamlit_app/Home.py`
   - Click "Deploy"

3. **Configure:**
   - Add secrets in app settings (if needed)
   - Set Python version (3.8-3.11)
   - Monitor logs

### Option 2: Streamlit Cloud Teams (Paid)

**Pros:**
- Can remove "Made with Streamlit" tag
- Custom domain support
- More resources
- Private apps
- Priority support

**Cost:** $250/month (as of 2024)

**Steps:**
1. Upgrade to Teams plan
2. Deploy as in Option 1
3. In app settings:
   - Enable custom domain
   - Hide Streamlit branding
   - Configure SSL

### Option 3: Self-Hosted (Full Control)

**Pros:**
- Complete control
- No attribution required
- Custom domain
- Full branding
- No resource limits

**Cons:**
- Server costs
- Maintenance required
- SSL setup needed

**Recommended Stack:**
- DigitalOcean Droplet ($12/month)
- Ubuntu 22.04
- Nginx reverse proxy
- Let's Encrypt SSL

**Steps:**

#### A. Server Setup

1. **Create Droplet:**
   ```bash
   # On DigitalOcean, create Ubuntu 22.04 droplet
   # Choose $12/month plan (2GB RAM)
   ```

2. **Initial Setup:**
   ```bash
   ssh root@your-server-ip
   
   # Update system
   apt update && apt upgrade -y
   
   # Create user
   adduser streamlit
   usermod -aG sudo streamlit
   su - streamlit
   ```

3. **Install Dependencies:**
   ```bash
   # Python
   sudo apt install python3.10 python3-pip python3-venv -y
   
   # Nginx
   sudo apt install nginx -y
   
   # Git
   sudo apt install git -y
   ```

#### B. Deploy Application

1. **Clone Repository:**
   ```bash
   cd ~
   git clone https://github.com/your-username/fraud_detection_project.git
   cd fraud_detection_project/streamlit_app
   ```

2. **Setup Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Test Run:**
   ```bash
   streamlit run Home.py
   # Should run on http://localhost:8501
   ```

#### C. Configure Nginx

1. **Create Nginx Config:**
   ```bash
   sudo nano /etc/nginx/sites-available/fraud-detection
   ```

2. **Add Configuration:**
   ```nginx
   server {
       listen 80;
       server_name yourdomain.com www.yourdomain.com;
       
       location / {
           proxy_pass http://localhost:8501;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
           
           # WebSocket support
           proxy_read_timeout 86400;
       }
   }
   ```

3. **Enable Site:**
   ```bash
   sudo ln -s /etc/nginx/sites-available/fraud-detection /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl restart nginx
   ```

#### D. Setup SSL (Let's Encrypt)

1. **Install Certbot:**
   ```bash
   sudo apt install certbot python3-certbot-nginx -y
   ```

2. **Get Certificate:**
   ```bash
   sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com
   ```

3. **Auto-Renewal:**
   ```bash
   sudo systemctl enable certbot.timer
   sudo systemctl start certbot.timer
   ```

#### E. Systemd Service

1. **Create Service File:**
   ```bash
   sudo nano /etc/systemd/system/streamlit.service
   ```

2. **Add Service Config:**
   ```ini
   [Unit]
   Description=Streamlit Fraud Detection App
   After=network.target
   
   [Service]
   Type=simple
   User=streamlit
   WorkingDirectory=/home/streamlit/fraud_detection_project/streamlit_app
   ExecStart=/home/streamlit/fraud_detection_project/streamlit_app/venv/bin/streamlit run Home.py --server.port 8501 --server.address localhost
   Restart=always
   RestartSec=10
   
   [Install]
   WantedBy=multi-user.target
   ```

3. **Enable and Start:**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable streamlit
   sudo systemctl start streamlit
   sudo systemctl status streamlit
   ```

#### F. Domain Configuration

1. **DNS Records:**
   ```
   Type: A
   Name: @
   Value: your-server-ip
   
   Type: A  
   Name: www
   Value: your-server-ip
   ```

2. **Wait for Propagation:**
   - DNS changes take 1-24 hours
   - Check with: `dig yourdomain.com`

## Custom Domain Setup

### For Streamlit Cloud (Teams Plan)

1. **Configure in Streamlit:**
   - App Settings → Domains
   - Add custom domain: `fraud-detection.yourdomain.com`
   - Copy CNAME record details

2. **Update DNS:**
   ```
   Type: CNAME
   Name: fraud-detection
   Value: your-app.streamlit.app
   TTL: 3600
   ```

### For Self-Hosted

Already covered in Option 3 above.

## Favicon Setup

### Create Favicon

**Option A: From PNG**
1. Create 32x32 or 64x64 PNG
2. Save as `streamlit_app/assets/favicon.png`
3. Convert to .ico using online tool

**Option B: From Emoji**
Already done in `Home.py`:
```python
st.set_page_config(page_icon="🛡️")
```

**Option C: Custom File**
```python
st.set_page_config(page_icon="assets/favicon.ico")
```

## Remove "Made with Streamlit"

### Community Cloud (Free)
**Not possible** - attribution required

### Teams/Enterprise Plan
1. App Settings → General
2. Toggle "Hide Made with Streamlit"

### Self-Hosted
Add custom CSS in `Home.py`:
```python
hide_streamlit = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit, unsafe_allow_html=True)
```

## Environment Variables

### Local Development
Create `.streamlit/secrets.toml`:
```toml
[general]
environment = "development"

[api]
# Add any API keys here (NOT committed to git)
```

### Streamlit Cloud
1. App Settings → Secrets
2. Add in TOML format
3. Access via `st.secrets`

### Self-Hosted
Use environment variables:
```bash
export FRAUD_DETECTION_ENV=production
```

## Monitoring

### Streamlit Cloud
- Built-in metrics dashboard
- View logs in real-time
- Resource usage tracking

### Self-Hosted

**Application Logs:**
```bash
sudo journalctl -u streamlit -f
```

**Nginx Logs:**
```bash
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

**System Monitoring:**
```bash
# Install monitoring tools
sudo apt install htop -y

# Check resources
htop
df -h
free -h
```

## Security Hardening

### 1. Firewall
```bash
sudo ufw allow 22    # SSH
sudo ufw allow 80    # HTTP
sudo ufw allow 443   # HTTPS
sudo ufw enable
```

### 2. SSH Security
```bash
# Disable password auth, use SSH keys only
sudo nano /etc/ssh/sshd_config
# Set: PasswordAuthentication no
sudo systemctl restart ssh
```

### 3. Auto Updates
```bash
sudo apt install unattended-upgrades -y
sudo dpkg-reconfigure unattended-upgrades
```

### 4. Fail2Ban
```bash
sudo apt install fail2ban -y
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

## Backup Strategy

### Database Backup (if applicable)
Not needed - no database in this app

### Code Backup
Git repository serves as backup

### Model Files
```bash
# Backup to S3
aws s3 cp models/random_forest_model.pkl s3://your-bucket/backups/
```

## Performance Optimization

### 1. Model File
```python
# Compress model if large
import joblib
from sklearn import __version__
joblib.dump(model, 'model.pkl', compress=3)
```

### 2. Caching
Already implemented with `@st.cache_resource`

### 3. Image Optimization
```bash
# Optimize PNG files
sudo apt install optipng -y
optipng results/figures/*.png
```

### 4. CDN (Optional)
Serve static assets from CDN for faster load times

## Troubleshooting

### App Won't Start
```bash
# Check logs
sudo journalctl -u streamlit -n 50

# Check if port is in use
sudo lsof -i :8501

# Restart service
sudo systemctl restart streamlit
```

### SSL Certificate Issues
```bash
# Renew manually
sudo certbot renew

# Check certificate
sudo certbot certificates
```

### High Memory Usage
```bash
# Check memory
free -h

# Increase swap if needed
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## Post-Deployment Testing

- [ ] Load homepage at custom domain
- [ ] Test all navigation links
- [ ] Submit fraud prediction form
- [ ] View all visualizations
- [ ] Check Privacy Policy
- [ ] Check Terms and Conditions
- [ ] Test on mobile device
- [ ] Verify SSL certificate (green padlock)
- [ ] Test with slow connection
- [ ] Check error handling (missing files)

## Maintenance

### Regular Tasks

**Daily:**
- Monitor error logs
- Check resource usage

**Weekly:**
- Review security updates
- Check SSL certificate expiry

**Monthly:**
- Update dependencies
- Review performance metrics
- Backup model files

### Update Deployment

```bash
# Pull latest code
cd ~/fraud_detection_project
git pull origin main

# Update dependencies
cd streamlit_app
source venv/bin/activate
pip install -r requirements.txt --upgrade

# Restart app
sudo systemctl restart streamlit
```

## Cost Estimate

### Streamlit Cloud
- Free: $0/month (with limitations)
- Teams: $250/month

### Self-Hosted (DigitalOcean)
- Droplet: $12/month
- Domain: $12/year
- SSL: Free (Let's Encrypt)
- **Total: ~$13/month**

## Final Checklist

Before announcing to public:

- [ ] Custom domain configured and working
- [ ] SSL certificate installed (https://)
- [ ] Favicon displays correctly
- [ ] Privacy Policy accurate and complete
- [ ] Terms and Conditions accurate and complete
- [ ] No fake metrics or placeholder content
- [ ] Professional design (no vibe-coded elements)
- [ ] Mobile responsive
- [ ] All features tested
- [ ] Error handling works
- [ ] Performance acceptable
- [ ] Monitoring set up
- [ ] Backup strategy in place
- [ ] Security hardened

**DO NOT LAUNCH** until all items checked!

---

**Ready to deploy professionally!**
