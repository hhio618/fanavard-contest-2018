build:
	python main.py all 300 0.0001
build-push: build
	git add . && git commit -m "Outputs" && git push origin output
