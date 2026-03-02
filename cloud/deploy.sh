#!/bin/bash
# Cadre-AI Cloud Run Deployment Script
# Run this in Google Cloud Shell or any machine with gcloud authenticated
#
# Usage: bash deploy.sh [PROJECT_ID] [GOOGLE_API_KEY]
# Or set environment variables: GCP_PROJECT, GOOGLE_API_KEY

set -e

# Configuration
PROJECT_ID="${1:-${GCP_PROJECT:-keen-defender-485200-b9}}"
API_KEY="${2:-${GOOGLE_API_KEY}}"
REGION="us-central1"
SERVICE_NAME="cadre-ai"
REPO="https://github.com/WeberG619/cadreai.git"
MODEL="gemini-2.5-flash-native-audio-latest"

echo "=== Cadre-AI Deployment ==="
echo "Project: $PROJECT_ID"
echo "Region:  $REGION"
echo "Service: $SERVICE_NAME"
echo ""

# Validate API key
if [ -z "$API_KEY" ]; then
    echo "ERROR: GOOGLE_API_KEY is required."
    echo "Usage: bash deploy.sh [PROJECT_ID] [GOOGLE_API_KEY]"
    echo "  or:  GOOGLE_API_KEY=xxx bash deploy.sh"
    exit 1
fi

# Set project
gcloud config set project "$PROJECT_ID"

# Enable required APIs
echo ">>> Enabling APIs..."
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com

# Clone repo if not already present
if [ ! -d "cadreai" ]; then
    echo ">>> Cloning repo..."
    git clone "$REPO"
fi
cd cadreai

# Build and deploy using Cloud Build + Cloud Run source deploy
echo ">>> Building and deploying to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
    --source . \
    --region "$REGION" \
    --allow-unauthenticated \
    --cpu 2 \
    --memory 1Gi \
    --timeout 1800 \
    --session-affinity \
    --min-instances 0 \
    --max-instances 3 \
    --set-env-vars "GOOGLE_API_KEY=$API_KEY,REVIT_ENABLED=false,CADRE_MODEL=$MODEL"

# Get the service URL
echo ""
echo ">>> Deployment complete!"
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --region "$REGION" --format "value(status.url)")
echo "Service URL: $SERVICE_URL"
echo ""

# Domain mapping instructions
echo "=== Domain Mapping ==="
echo "To map cadreai.dev to this service:"
echo ""
echo "1. Run:"
echo "   gcloud run domain-mappings create --service $SERVICE_NAME --domain cadreai.dev --region $REGION"
echo ""
echo "2. Add DNS records in Squarespace:"
echo "   Type: CNAME"
echo "   Host: @"
echo "   Value: ghs.googlehosted.com"
echo ""
echo "   (For www subdomain:)"
echo "   Type: CNAME"
echo "   Host: www"
echo "   Value: ghs.googlehosted.com"
echo ""
echo "Done!"
