# Final executable name (placed in project root)
TARGET := chess_engine

# Directory where all .o files are placed
BUILD_DIR := build

# Compilers
CC  := gcc
CXX := g++

# Optimization and ARCH_FLAGS
OPT := -O3 -DNDEBUG
ARCH_FLAGS := -march=native

CFLAGS_COMMON   = $(OPT) -I./fathom
CXXFLAGS_COMMON = $(OPT) -I./fathom -std=c++20
LDFLAGS        := -lpthread

# Source files

C_SRCS := fathom/tbprobe.c

CXX_SRCS := \
    main.cpp \
    eval/eval.cpp \
    uci/uci.cpp \
    search/search.cpp \
    book/book.cpp \
    nnue/nnue.cpp

# All object files land flat in build/
C_OBJS   := $(addprefix $(BUILD_DIR)/, $(notdir $(C_SRCS:.c=.o)))
CXX_OBJS := $(addprefix $(BUILD_DIR)/, $(notdir $(CXX_SRCS:.cpp=.o)))
OBJS     := $(C_OBJS) $(CXX_OBJS)

# VPATH lets make find the .c/.cpp sources even though targets are flattened
VPATH := fathom eval uci search book nnue .

# Rules
.PHONY: all native clean

all: clean $(TARGET)

# Link step: must use g++ so the C++ runtime/stdlib is linked in correctly.
# Binary is written directly to the project root.
$(TARGET): $(OBJS)
	$(CXX) $(OPT) $(ARCH_FLAGS) -o $@ $(OBJS) $(LDFLAGS)

# Pattern rule: compile .c files with gcc, output flat into build/
$(BUILD_DIR)/%.o: %.c | $(BUILD_DIR)
	$(CC) $(CFLAGS_COMMON) $(ARCH_FLAGS) -c $< -o $@

# Pattern rule: compile .cpp files with g++, output flat into build/
$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_COMMON) $(ARCH_FLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)
