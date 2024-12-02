# Makefile for compiling and running sdp_serial.cpp

# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -Wall -O2

# Target executable
TARGET = sdp_serial

# Source file
SRC = sdp_serial.cpp

# Default target
all: $(TARGET) run

# Compile the source file into an executable
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

# Run the executable
run: $(TARGET)
	./$(TARGET)

# Clean up the build files
clean:
	rm -f $(TARGET)

.PHONY: all run clean