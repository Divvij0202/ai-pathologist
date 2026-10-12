#!/usr/bin/env bash
# Monthly cost budget with an email alert at 80%. Usage: EMAIL=you@example.com LIMIT=50 ./deploy/budget_alarm.sh
set -euo pipefail
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
aws budgets create-budget --account-id "$ACCOUNT" \
  --budget "{\"BudgetName\":\"ai-pathologist\",\"BudgetLimit\":{\"Amount\":\"${LIMIT:-50}\",\"Unit\":\"USD\"},\"TimeUnit\":\"MONTHLY\",\"BudgetType\":\"COST\"}" \
  --notifications-with-subscribers "[{\"Notification\":{\"NotificationType\":\"ACTUAL\",\"ComparisonOperator\":\"GREATER_THAN\",\"Threshold\":80},\"Subscribers\":[{\"SubscriptionType\":\"EMAIL\",\"Address\":\"${EMAIL:?set EMAIL}\"}]}]"
echo "budget created"
