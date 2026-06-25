#!/usr/bin/env bash
# Launches the ZyinkLabs CpGPT training EC2 instance via the AWS CLI.
#
# Prerequisites:
#   - AWS CLI installed and configured (aws configure) with credentials that can
#     create EC2 instances, security groups, and IAM roles.
#   - An existing EC2 key pair (for SSH access) -- create one if you don't have one:
#       aws ec2 create-key-pair --key-name zyinklabs-training --query 'KeyMaterial' \
#           --output text > zyinklabs-training.pem
#       chmod 400 zyinklabs-training.pem
#
# This script is intentionally explicit/copy-pasteable rather than hidden behind
# a CloudFormation/Terraform abstraction -- adapt it once you're happy with the
# manual flow, but run it manually at least once so the moving pieces are clear.
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration -- edit these for your AWS account/region
# ---------------------------------------------------------------------------
REGION="${AWS_REGION:-us-east-1}"
KEY_NAME="${KEY_NAME:-zyinklabs-training}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g6e.xlarge}"   # 1x NVIDIA L40S, 48GB VRAM -- sized for
                                                # the "large" (~100M param) CpGPT model
                                                # with real headroom above CpGPT's 24GB
                                                # minimum. g5.xlarge's A10G has only
                                                # 22.35GiB usable VRAM, under that
                                                # minimum with no real margin, so it's
                                                # deliberately not used here.
VOLUME_SIZE_GB="${VOLUME_SIZE_GB:-200}"        # ComputAgeBench (~38.6GB) + CpGPT
                                                # dependencies + checkpoints add up;
                                                # 200GB gp3 is a reasonable starting point.
SG_NAME="zyinklabs-training-sg"
ROLE_NAME="zyinklabs-training-role"
INSTANCE_PROFILE_NAME="zyinklabs-training-profile"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "== ZyinkLabs EC2 training instance launch =="
echo "Region: $REGION | Instance type: $INSTANCE_TYPE | Volume: ${VOLUME_SIZE_GB}GB"

# ---------------------------------------------------------------------------
# 1. Find the latest AWS Deep Learning AMI (Ubuntu 22.04, GPU-ready, NVIDIA
#    drivers + CUDA preinstalled -- this is what bootstrap.sh assumes).
# ---------------------------------------------------------------------------
echo "--> Looking up latest Deep Learning AMI..."
AMI_ID=$(aws ec2 describe-images \
    --owners amazon \
    --filters "Name=name,Values=Deep Learning AMI GPU PyTorch*Ubuntu 22.04*" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --region "$REGION" \
    --output text)

if [ "$AMI_ID" == "None" ] || [ -z "$AMI_ID" ]; then
    echo "ERROR: Could not find a matching Deep Learning AMI in region $REGION."
    echo "Browse available AMIs manually: https://aws.amazon.com/releasenotes/aws-deep-learning-ami/"
    exit 1
fi
echo "    Using AMI: $AMI_ID"

# ---------------------------------------------------------------------------
# 2. Create (or reuse) an IAM role + instance profile scoped to what training
#    actually needs (see iam_policy_training_instance.json).
# ---------------------------------------------------------------------------
echo "--> Setting up IAM role: $ROLE_NAME"

if ! aws iam get-role --role-name "$ROLE_NAME" &> /dev/null; then
    aws iam create-role \
        --role-name "$ROLE_NAME" \
        --assume-role-policy-document '{
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "ec2.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }]
        }' > /dev/null

    aws iam put-role-policy \
        --role-name "$ROLE_NAME" \
        --policy-name "zyinklabs-training-policy" \
        --policy-document "file://${SCRIPT_DIR}/iam_policy_training_instance.json"

    aws iam create-instance-profile --instance-profile-name "$INSTANCE_PROFILE_NAME" > /dev/null
    aws iam add-role-to-instance-profile \
        --instance-profile-name "$INSTANCE_PROFILE_NAME" \
        --role-name "$ROLE_NAME"

    echo "    Created role + instance profile. Waiting 15s for IAM propagation..."
    sleep 15
else
    echo "    Role already exists, reusing."
fi

# ---------------------------------------------------------------------------
# 3. Security group -- SSH only, locked to your current IP. Adjust if you need
#    to access from a different network later.
# ---------------------------------------------------------------------------
echo "--> Setting up security group: $SG_NAME"

VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" \
    --query 'Vpcs[0].VpcId' --region "$REGION" --output text)

SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" \
    --query 'SecurityGroups[0].GroupId' --region "$REGION" --output text 2>/dev/null || echo "None")

if [ "$SG_ID" == "None" ] || [ -z "$SG_ID" ]; then
    SG_ID=$(aws ec2 create-security-group \
        --group-name "$SG_NAME" \
        --description "ZyinkLabs CpGPT training instance -- SSH only" \
        --vpc-id "$VPC_ID" \
        --region "$REGION" \
        --query 'GroupId' --output text)

    MY_IP=$(curl -s https://checkip.amazonaws.com)
    aws ec2 authorize-security-group-ingress \
        --group-id "$SG_ID" \
        --protocol tcp --port 22 --cidr "${MY_IP}/32" \
        --region "$REGION" > /dev/null

    echo "    Created security group $SG_ID, SSH locked to ${MY_IP}/32"
else
    echo "    Security group already exists: $SG_ID"
fi

# ---------------------------------------------------------------------------
# 4. Launch the instance
# ---------------------------------------------------------------------------
echo "--> Launching instance..."

INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --iam-instance-profile "Name=$INSTANCE_PROFILE_NAME" \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":${VOLUME_SIZE_GB},\"VolumeType\":\"gp3\"}}]" \
    --user-data "file://${SCRIPT_DIR}/bootstrap.sh" \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=zyinklabs-cpgpt-training}]' \
    --region "$REGION" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "    Launched instance: $INSTANCE_ID"
echo "--> Waiting for instance to reach 'running' state..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --region "$REGION" --output text)

cat <<EOF

== Instance launched ==
Instance ID: $INSTANCE_ID
Public IP:   $PUBLIC_IP

Bootstrap (Docker, NVIDIA Container Toolkit, repo clone) runs automatically via
user-data and typically takes 3-5 minutes. Check progress with:

  ssh -i ${KEY_NAME}.pem ubuntu@${PUBLIC_IP} 'tail -f /var/log/zyinklabs-bootstrap.log'

Once bootstrap completes:

  ssh -i ${KEY_NAME}.pem ubuntu@${PUBLIC_IP}
  cd cpgpt-pipeline/docker
  cp .env.example .env && nano .env       # fill in real values
  docker compose run --rm download_dependencies
  docker compose run --rm prepare_data
  docker compose run --rm train

REMEMBER TO TERMINATE THE INSTANCE WHEN DONE -- GPU instances bill per second
while running, regardless of whether a job is actively using the GPU:

  aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION

EOF
