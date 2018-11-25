PROJECT_NAME := ntsa
SHARED := $(if $(SHARED_VOLUME), -v $(SHARED_VOLUME):$(SHARED_VOLUME),)

docker:
	docker build -t ${PROJECT_NAME}_dev -f docker/Dockerfile .

dev:
	docker rm ${PROJECT_NAME}_dev || true
	docker run -it --name ${PROJECT_NAME}_dev ${SHARED} -v $(CURDIR):/workspace/ -v /var/run/docker.sock:/var/run/docker.sock:ro -p 8888:8888 --entrypoint /bin/bash ${PROJECT_NAME}_dev

run:
	docker-compose -f docker/docker-compose.yml up -d ntsa
	docker exec -it ntsa bash

.PHONY: docker dev run