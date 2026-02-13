#!/bin/bash

# Interrompe o script se qualquer comando falhar
set -e

# Recebe os argumentos passados pelo Python
RG_NAME=$1
WORKSPACE_NAME=$2
LOCATION=$3

echo "--- Iniciando Setup de Infraestrutura Azure ---"
echo "Resource Group: $RG_NAME"
echo "Workspace: $WORKSPACE_NAME"
echo "Região: $LOCATION"

# 1. Cria o Resource Group (se nao existir)
echo "1. Verificando Resource Group..."
if az group exists --name "$RG_NAME" | grep -q "true"; then
	echo "Resource Group ja existe. Pulando criacao."
else
	echo "Criando Resource Group..."
	az group create --name "$RG_NAME" --location "$LOCATION" --output table
fi

# 2. Cria o Azure ML Workspace (se nao existir)
# O Azure CLI cria automaticamente Storage, KeyVault e AppInsights aqui!
echo "2. Verificando Azure Machine Learning Workspace..."
if az ml workspace show --name "$WORKSPACE_NAME" --resource-group "$RG_NAME" --only-show-errors > /dev/null 2>&1; then
	echo "Workspace ja existe. Pulando criacao."
else
	echo "Criando Azure Machine Learning Workspace (isso pode levar uns minutos)..."
	az ml workspace create --name "$WORKSPACE_NAME" --resource-group "$RG_NAME" --location "$LOCATION" --output table
fi

echo "--- Sucesso! Infraestrutura criada. ---"