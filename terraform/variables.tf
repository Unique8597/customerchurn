variable "resource_group_name" {
  description = "The name of the resource group"
  type        = string
}

variable "virtual_network_name" {
  description = "The name of the virtual network"
  type        = string
}

variable "eventhub_namespace_name" {
  description = "The name of the Event Hub namespace"
  type        = string
}
variable "eventhub_name" {
  description = "The name of the Event Hub"
  type        = string
}