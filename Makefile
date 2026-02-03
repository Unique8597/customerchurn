
login:
	az login --service-principal -u $(CLIENT_ID) -p $(CLIENT_SECRET) --tenant $(TENANT_ID)
	az account set --subscription $(SUBSCRIPTION_ID)

install:
	pip install -r requirements-dev.txt

package:
	func pack 