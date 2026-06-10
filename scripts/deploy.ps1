# Deploy the latest CI-built image to Azure Container Apps
# Run this AFTER the CI pipeline succeeds (mlops-key-auth.yml)
# Requires: az login (you're already logged in)

param(
    [Parameter(Mandatory = $false)]
    [string]$ImageTag,

    [Parameter(Mandatory = $false)]
    [string]$ResourceGroup = "mlops-rg",

    [Parameter(Mandatory = $false)]
    [string]$ContainerAppName = "water-potability-api",

    [Parameter(Mandatory = $false)]
    [string]$AcrName = "waterpotabilityacr",

    [Parameter(Mandatory = $false)]
    [string]$ModelName = "water_potability_model",

    [Parameter(Mandatory = $false)]
    [string]$ModelStage = "Production",

    [Parameter(Mandatory = $false)]
    [string]$MlflowUri
)

# Get MLflow URI if not provided
if (-not $MlflowUri) {
    $MlflowUri = az ml workspace show `
        --resource-group $ResourceGroup `
        --name water-potability-mlw `
        --query mlflow_tracking_uri -o tsv
    Write-Host "MLflow URI: $MlflowUri"
}

# Get image tag from ACR if not provided
if (-not $ImageTag) {
    $sha = git rev-parse HEAD
    $loginServer = "${AcrName}.azurecr.io"
    $ImageTag = "${loginServer}/water-potability-api:${sha}"
    Write-Host "Image tag: $ImageTag"
}

$version = Get-Date -Format "yyyy-MM-dd_HHmmss"

Write-Host "Deploying $ImageTag to $ContainerAppName..."

az containerapp update `
    --name $ContainerAppName `
    --resource-group $ResourceGroup `
    --image $ImageTag `
    --set-env-vars `
        MODEL_NAME=$ModelName `
        MODEL_STAGE_OR_ALIAS=$ModelStage `
        DATASET_VERSION=$version `
        MLFLOW_TRACKING_URI=$MlflowUri

Write-Host "Done. Checking health..."

Start-Sleep -Seconds 30

$fqdn = az containerapp show `
    --name $ContainerAppName `
    --resource-group $ResourceGroup `
    --query properties.configuration.ingress.fqdn -o tsv

Invoke-RestMethod -Uri "https://${fqdn}/health" -Method Get
