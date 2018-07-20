build:
	python main.py all 150 0.0003
build-push: build
	git add . && git commit -m "Outputs" && git push origin output
clean:
	rm -rf output/models/tmp/*
