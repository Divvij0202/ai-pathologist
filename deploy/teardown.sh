#!/usr/bin/env bash
# Stop paying: terminate the demo instance launched by launch_graviton.sh
set -euo pipefail
export AWS_REGION=${AWS_REGION:-us-west-2}
ID=$(cat deploy/.instance_id)
aws ec2 terminate-instances --instance-ids "$ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text
