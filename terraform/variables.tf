variable "resource_group_name" {
  description = "The name of the resource group"
  type        = string
}

variable "azure_storage_account" {
  description = "The name of the azure storage account"
  type        = string
}

variable "ml_workspace_name" {
  description = "The name of the ML workspace"
  type        = string
}

variable "azure_key_vault" {
  description = "The name of the Azure Key Vault"
  type        = string
}

variable "azure_app_insight" {
  description = "Name of the Application Insight"
  type = string
}