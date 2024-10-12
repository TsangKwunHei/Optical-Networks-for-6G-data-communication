all:
	mkdir -p build
	python3 bundler.py -i ./src/main.cpp -o ./build/main.cpp
