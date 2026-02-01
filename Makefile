
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
	@if ! az eventhubs eventhub show --name $(EVENTHUB) --namespace-name $(EVENTHUB_NAMESPACE) --resource-group $(RESOURCE_GROUP) >/dev/null 2>&1; then \
		echo "Creating Event Hub..."; \
		az eventhubs eventhub create --name $(EVENTHUB) --namespace-name $(EVENTHUB_NAMESPACE) --resource-group $(RESOURCE_GROUP) --partition-count 2 --message-retention-in-days 1; \
	else \
		echo "Event Hub $(EVENTHUB) already exists."; \
	fi

install:
	pip install -r requirements-dev.txt

package:
	func pack 

deploy_function:
	func azure functionapp publish $(FUNCTION_APP_NAME) --python
