ifneq (,$(wildcard key.env))
	include .env
	export $(shell sed 's/=.*//' key.env)
endif

login:
	az login --service-principal -u $(CLIENT_ID) -p $(CLIENT_SECRET) --tenant $(TENANT_ID)
	az account set --subscription $(SUBSCRIPTION_ID)

rg:
	@if [ "$$(az group exists --name $(RESOURCE_GROUP))" = "false" ]; then \
		echo "Creating resource group $(RESOURCE_GROUP)..."; \
		az group create --name $(RESOURCE_GROUP) --location $(LOCATION); \
	else \
		echo "Resource group $(RESOURCE_GROUP) already exists."; \
	fi

eh-namespace:
	@if ! az eventhubs namespace show --name $(EVENTHUB_NAMESPACE) --resource-group $(RESOURCE_GROUP) >/dev/null 2>&1; then \
		echo "Creating Event Hub namespace $(EVENTHUB_NAMESPACE)..."; \
		az eventhubs namespace create --name $(EVENTHUB_NAMESPACE) --resource-group $(RESOURCE_GROUP) --location $(LOCATION) --sku Standard; \
	else \
		echo "Event Hub namespace $(EVENTHUB_NAMESPACE) already exists."; \
	fi

eh-create:
	@if ! az eventhubs eventhub show --name $(EVENT_HUB) --namespace-name $(EH_NAMESPACE) --resource-group $(RESOURCE_GROUP) >/dev/null 2>&1; then \
		echo "Creating Event Hub $(EVENT_HUB)..."; \
		az eventhubs eventhub create --name $(EVENT_HUB) --namespace-name $(EH_NAMESPACE) --resource-group $(RESOURCE_GROUP) --partition-count 2 --message-retention 1; \
	else \
		echo "Event Hub $(EVENT_HUB) already exists."; \
	fi

install:
	pip install -r requirements-dev.txt

package:
	func pack 

deploy_function:
	func azure functionapp publish $(FUNCTION_APP_NAME) --python

install-azure-cli:
	sudo apt-get update
	sudo apt-get install apt-transport-https ca-certificates curl gnupg lsb-release
	sudo mkdir -p /etc/apt/keyrings
	curl -sLS https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor | sudo tee /etc/apt/keyrings/microsoft.gpg > /dev/null
	sudo chmod go+r /etc/apt/keyrings/microsoft.gpg
	AZ_DIST=$(lsb_release -cs)
	echo "Types: deb
	URIs: https://packages.microsoft.com/repos/azure-cli/
	Suites: ${AZ_DIST}
	Components: main
	Architectures: $(dpkg --print-architecture)
	Signed-by: /etc/apt/keyrings/microsoft.gpg" | sudo tee /etc/apt/sources.list.d/azure-cli.sources
