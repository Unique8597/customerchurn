terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.100"
    }
  }

  required_version = ">= 1.3.0"

  backend "remote" {
    organization = "Endowed"
    workspaces {
      name = "customerchurn"
    }
  }
}

provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "ml" {
  name     = var.resource_group_name
  location = "westus2"

  tags = {
    Environment = "Production"
    Team        = "ML Team"
  }
}

resource "azurerm_storage_account" "ml" {
  name                     = var.storage_account
  resource_group_name      = azurerm_resource_group.ml.name
  location                 = azurerm_resource_group.ml.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
}

resource "azurerm_machine_learning_workspace" "ml_workspace" {
  name                = var.ml_workspace_name
  location            = azurerm_resource_group.ml.location
  resource_group_name = azurerm_resource_group.ml.name
  storage_account_name = azurerm_storage_account.ml.name

  sku_name = "Basic"

  tags = {
    Environment = "Production"
    Team        = "ML Team"
  }
}