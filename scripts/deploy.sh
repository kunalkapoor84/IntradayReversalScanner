#!/bin/bash
# Cloud deployment to Oracle Cloud / Google Cloud
# Usage: bash deploy.sh <vm_ip> <ssh_key_path>

VM_IP=${1:-"your-vm-ip"}
SSH_KEY=${2:-"~/.ssh/id_rsa"}

echo "Deploying to ${VM_IP}..."

# Create deployment tar
tar czf deploy.tar.gz \
    --exclude='venv' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='*.pyc' \
    .

# Copy to VM
scp -i ${SSH_KEY} deploy.tar.gz ubuntu@${VM_IP}:~/scanner/

# Remote setup
ssh -i ${SSH_KEY} ubuntu@${VM_IP} << 'EOF'
    cd ~/scanner
    tar xzf deploy.tar.gz
    chmod +x scripts/setup.sh
    ./scripts/setup.sh
    source venv/bin/activate
    
    # Create systemd service for 24/7 operation
    sudo tee /etc/systemd/system/scanner.service > /dev/null << 'SERVICEEOF'
[Unit]
Description=Intraday Reversal Scanner
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/scanner
ExecStart=/home/ubuntu/scanner/venv/bin/python /home/ubuntu/scanner/run.py --mode live
Restart=on-failure
RestartSec=30
Environment=TZ=Asia/Kolkata

[Install]
WantedBy=multi-user.target
SERVICEEOF

    sudo systemctl daemon-reload
    sudo systemctl enable scanner
    sudo systemctl start scanner
    sudo systemctl status scanner --no-pager
EOF

rm deploy.tar.gz
echo "Deployment complete! Scanner running 24/7 as systemd service."