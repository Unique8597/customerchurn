terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.6.0"
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

resource "azurerm_storage_account" "azure_storage_account" {
  name                     = var.azure_storage_account
  resource_group_name      = azurerm_resource_group.rg.name
  location                 = azurerm_resource_group.rg.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
}

resource "azurerm_machine_learning_workspace" "ml_workspace_name" {
  name                = var.ml_workspace_name
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  storage_account_name = azurerm_storage_account.azure_storage_account.name

  sku_name = "Basic"

  tags = {
    Environment = "Production"
    Team        = "ML Team"
  }
}