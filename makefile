PROJECT_ID=burstier-review
IMAGE_NAME=dash-app
TAG=latest
REGION=europe-west1
REPO=eu.gcr.io

docker-build:
	docker build -t $(IMAGE_NAME):$(TAG) .

docker-run:
	docker run -p 8080:8080 -t $(IMAGE_NAME):$(TAG)

docker-tag:
	# docker tag $(IMAGE_NAME):$(TAG) $(REGION)-docker.pkg.dev/$(PROJECT_ID)/$(REPO)/$(IMAGE_NAME):$(TAG)
	docker tag $(IMAGE_NAME):$(TAG) $(REPO)/$(PROJECT_ID)/$(IMAGE_NAME):$(TAG)

gcloud-auth:
	gcloud auth login
	gcloud auth configure-docker $(REGION)-docker.pkg.dev
	gcloud auth configure-docker europe-docker.pkg.dev

docker-push:
	# docker push $(REGION)-docker.pkg.dev/$(PROJECT_ID)/$(REPO)/$(IMAGE_NAME):$(TAG)
	docker push $(REPO)/$(PROJECT_ID)/$(IMAGE_NAME):$(TAG)

docker-deploy: docker-build docker-tag gcloud-auth docker-push
