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

resource "azurerm_resource_group" "rg" {
  name     = var.resource_group_name
  location = "westus2"

  tags = {
    Environment = "Production"
    Team        = "ML Team"
  }
}

resource "azurerm_storage_account" "azure_storage_name" {
  name                     = var.azure_storage_account
  resource_group_name      = azurerm_resource_group.rg.name
  location                 = azurerm_resource_group.rg.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
}
resource "azurerm_key_vault" "ml_kv" {
  name                        = var.azure_key_vault
  location                    = azurerm_resource_group.rg.location
  resource_group_name         = azurerm_resource_group.rg.name
  tenant_id                   = data.azurerm_client_config.current.tenant_id
  sku_name                    = "standard"
}

resource "azurerm_machine_learning_workspace" "ml_name" {
  name                   = var.ml_workspace_name
  location               = azurerm_resource_group.rg.location
  resource_group_name    = azurerm_resource_group.rg.name
  storage_account_name   = azurerm_storage_account.azure_storage_name.name
  key_vault_name         = azurerm_key_vault.ml_kv.name
  sku_name               = "Basic"

  tags = {
    Environment = "Production"
    Team        = "ML Team"
  }
}