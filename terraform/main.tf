terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.1.0"
    }
  }

  required_version = ">= 1.1.0"

  cloud {
    organization = "Endowed"
    workspaces {
      name = "customerchurn"
    }
  }
}

provider "azurerm" {
  features {}
}

data "azurerm_client_config" "current" {}

resource "azurerm_resource_group" "rg" {
  name     = var.resource_group_name
  location = "eastus"

  tags = {
    Environment = "Production"
    Team        = "ML Team"
  }
}

resource "azurerm_storage_account" "azure_storage" {
  name                     = var.azure_storage_account
  resource_group_name      = azurerm_resource_group.rg.name
  location                 = azurerm_resource_group.rg.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  
  blob_properties {
    versioning_enabled = true
  }
}

resource "azurerm_key_vault" "ml_kv" {
  name                        = var.azure_key_vault
  location                    = azurerm_resource_group.rg.location
  resource_group_name         = azurerm_resource_group.rg.name
  tenant_id                   = data.azurerm_client_config.current.tenant_id
  sku_name                    = "standard"
}

resource "azurerm_application_insights" "ml_ai" {
  name                = "${var.ml_workspace_name}-ai"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  application_type    = "web"
}

resource "azurerm_machine_learning_workspace" "ml_workspace" {
  name                = var.ml_workspace_name
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name

  storage_account_id      = azurerm_storage_account.azure_storage.id
  key_vault_id            = azurerm_key_vault.ml_kv.id
  application_insights_id = azurerm_application_insights.ml_ai.id

  sku_name = "Basic"

  identity {
  type = "SystemAssigned"
}
  tags = {
    Environment = "Production"
    Team        = "ML Team"
  }
}