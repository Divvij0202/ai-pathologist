#!/usr/bin/env bash
# Launch the vision service + agent on one Graviton instance in us-west-2 (next to the CAMELYON16 bucket).
#
# Needs: AWS CLI v2 logged in to your account, and Bedrock model access for Claude in us-west-2.
# Usage:  REPO_URL=https://github.com/<you>/ai-pathologist.git ./deploy/launch_graviton.sh
# It prints the public URL when the instance is up. Tear down with ./deploy/teardown.sh.
set -euo pipefail

REGION=${AWS_REGION:-us-west-2}
INSTANCE_TYPE=${INSTANCE_TYPE:-c7g.xlarge}   # 4 Graviton3 vCPU, 8 GB; ~0.145 USD/h
NAME=ai-pathologist
REPO_URL=${REPO_URL:?set REPO_URL to your git repo}
export AWS_REGION=$REGION

AMI=$(aws ssm get-parameter --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64 \
      --query Parameter.Value --output text)

# IAM role so the instance can call Bedrock without keys on disk
if ! aws iam get-role --role-name $NAME >/dev/null 2>&1; then
  aws iam create-role --role-name $NAME --assume-role-policy-document '{
    "Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  aws iam put-role-policy --role-name $NAME --policy-name bedrock-invoke --policy-document '{
    "Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["bedrock:InvokeModel","bedrock:InvokeModelWithResponseStream","bedrock-mantle:*"],"Resource":"*"}]}'
  aws iam create-instance-profile --instance-profile-name $NAME >/dev/null
  aws iam add-role-to-instance-profile --instance-profile-name $NAME --role-name $NAME
  sleep 10
fi

VPC=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
SG=$(aws ec2 describe-security-groups --filters Name=group-name,Values=$NAME Name=vpc-id,Values=$VPC \
     --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)
if [[ -z "$SG" || "$SG" == "None" ]]; then
  SG=$(aws ec2 create-security-group --group-name $NAME --description "AI pathologist demo" --vpc-id $VPC \
       --query GroupId --output text)
  aws ec2 authorize-security-group-ingress --group-id $SG --protocol tcp --port 8000 --cidr 0.0.0.0/0 >/dev/null
fi

USERDATA=$(cat <<EOF
#!/bin/bash
dnf install -y git docker
systemctl enable --now docker
git clone $REPO_URL /opt/app
cd /opt/app
docker build -t $NAME .
docker run -d --restart unless-stopped -p 8000:8000 -e AWS_REGION=$REGION -e PATHAGENT_LLM=bedrock \
  -v /opt/app/runs:/app/runs $NAME
EOF
)

ID=$(aws ec2 run-instances --image-id $AMI --instance-type $INSTANCE_TYPE \
      --iam-instance-profile Name=$NAME --security-group-ids $SG \
      --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":30,"VolumeType":"gp3"}}]' \
      --user-data "$USERDATA" --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME}]" \
      --query 'Instances[0].InstanceId' --output text)
echo "instance $ID starting..."
aws ec2 wait instance-running --instance-ids $ID
IP=$(aws ec2 describe-instances --instance-ids $ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "Up in a few minutes (docker build) at: http://$IP:8000   health: http://$IP:8000/health"
echo $ID > deploy/.instance_id
