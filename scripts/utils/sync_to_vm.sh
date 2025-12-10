#!/bin/bash

# Sync Traffic Monitoring repository to VM with temporal-aware changes

VM_IP="209.20.159.8"
VM_USER="ubuntu"
SSH_KEY="~/.ssh/lambda-cloud-ssh.pem"
REMOTE_PATH="/home/ubuntu/Traffic_Monitoring"
LOCAL_PATH="."

echo "Syncing Traffic Monitoring to VM..."

# Create directory structure on VM if it doesn't exist
ssh -i $SSH_KEY $VM_USER@$VM_IP "mkdir -p $REMOTE_PATH"

# Sync files excluding large directories and cache
rsync -avz \
  --progress \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '*.pyo' \
  --exclude '.DS_Store' \
  --exclude 'backup' \
  --exclude 'checkpoints' \
  --exclude 'docker' \
  --exclude 'HDF5' \
  --exclude 'ImpV1' \
  --exclude 'logs' \
  --exclude 'OpenEB' \
  --exclude '.git' \
  -e "ssh -i $SSH_KEY" \
  $LOCAL_PATH/ \
  $VM_USER@$VM_IP:$REMOTE_PATH/

echo "Sync complete!"
echo "Next steps on VM:"
echo "  1. cd $REMOTE_PATH"
echo "  2. python3 -c 'import torch; print(\"CUDA:\", torch.cuda.is_available())'"
echo "  3. Start training with: python3 comprehensive_training.py"

