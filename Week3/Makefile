all: f_main r

run: 
	@if [ -f "./result" ]; then \
		./result; \
	else \
		echo -e "\nError: Please compile program first!\n"; exit 1; \
	fi

file_%:
	@if [ -f "$*.cpp" ]; then \
		g++ -O3 -Wall -std=c++17 $*.cpp -o result; \
	else \
		echo -e "\nError: No such file found!\n"; exit 1; \
	fi

r: run
f_%:
	@$(MAKE) --no-print-directory file_$*

.PHONY: all run r