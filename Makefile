CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -pthread
all: build/app
build:
	mkdir -p build
build/app: main.cpp httplib.h | build
	$(CXX) $(CXXFLAGS) main.cpp -o $@
build/core-tests: tests/core.cpp main.cpp httplib.h | build
	$(CXX) $(CXXFLAGS) tests/core.cpp -o $@
build/benchmark: benchmarks/runner.cpp main.cpp httplib.h | build
	$(CXX) $(CXXFLAGS) benchmarks/runner.cpp -o $@
test: build/core-tests build/app
	./build/core-tests
	python3 tests/api.py
benchmark: build/benchmark
	./build/benchmark
.PHONY: all test benchmark
build/metric-check: benchmarks/metric_consistency.cpp main.cpp httplib.h | build
	$(CXX) $(CXXFLAGS) benchmarks/metric_consistency.cpp -o $@
