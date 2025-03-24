PROJECT_ID=burstier-review
# IMAGE_NAME=dash-app
TAG=latest
REGION=europe-west1
REPO=eu.gcr.io

DATASET?=inhibblock
APPLICATION?=review

docker-build:
	docker build --build-arg DATASET=$(DATASET) --build-arg APPLICATION=$(APPLICATION) -t $(APPLICATION)-$(DATASET):$(TAG) .

docker-run:
	docker run -p 8080:8080 -t $(APPLICATION)-$(DATASET):$(TAG)

docker-tag:
	# docker tag $(APPLICATION)-$(DATASET):$(TAG) $(REGION)-docker.pkg.dev/$(PROJECT_ID)/$(REPO)/$(APPLICATION)-$(DATASET):$(TAG)
	docker tag $(APPLICATION)-$(DATASET):$(TAG) $(REPO)/$(PROJECT_ID)/$(APPLICATION)-$(DATASET):$(TAG)

gcloud-auth:
	gcloud auth login
	gcloud auth configure-docker $(REGION)-docker.pkg.dev
	gcloud auth configure-docker europe-docker.pkg.dev

docker-push:
	# docker push $(REGION)-docker.pkg.dev/$(PROJECT_ID)/$(REPO)/$(APPLICATION)-$(DATASET):$(TAG)
	docker push $(REPO)/$(PROJECT_ID)/$(APPLICATION)-$(DATASET):$(TAG)

docker-deploy: docker-build docker-tag gcloud-auth docker-push
