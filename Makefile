# Makefile for compiling and running sdp_serial.cpp

# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -Wall -O2

# Target executable
TARGET = sdp_serial

# Source file
SRC = sdp_serial.cpp

# Arguments
arg1 = 1
arg2 = 12
arg3 = 64
arg4 = 64
arg5 = 4096

# Default target
all: $(TARGET) sdp_mm_parallel sdp_bmm sdp_online_softmax sdp_fused

# Compile the source file into an executable
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)
	

# Run the executable
run: $(TARGET) sdp_mm_parallel sdp_bmm sdp_online_softmax sdp_fused
	./$(TARGET) $(arg1) $(arg2) $(arg3) $(arg4) $(arg5)
	./sdp_mm_parallel $(arg1) $(arg2) $(arg3) $(arg4) $(arg5)
	./sdp_bmm $(arg1) $(arg2) $(arg3) $(arg4) $(arg5)
	./sdp_online_softmax $(arg1) $(arg2) $(arg3) $(arg4) $(arg5)
	./sdp_fused $(arg1) $(arg2) $(arg3) $(arg4) $(arg5)

# Compile and run sdp_mm_parallel.cpp
parallel: sdp_mm_parallel

sdp_mm_parallel: sdp_mm_parallel.cpp
	$(CXX) $(CXXFLAGS) -fopenmp -o sdp_mm_parallel sdp_mm_parallel.cpp

run_parallel: sdp_mm_parallel
	./sdp_mm_parallel

bmm: sdp_bmm

sdp_bmm: sdp_bmm.cpp
	$(CXX) $(CXXFLAGS) -fopenmp -o sdp_bmm sdp_bmm.cpp

run_bmm: sdp_bmm
	./sdp_bmm

sdp_online_softmax: sdp_online_softmax.cpp
	$(CXX) $(CXXFLAGS) -fopenmp -o sdp_online_softmax sdp_online_softmax.cpp

sdp_fused: sdp_fused.cpp
	$(CXX) $(CXXFLAGS) -fopenmp -o sdp_fused sdp_fused.cpp

# Clean up the build files
clean:
	rm -f $(TARGET)

.PHONY: all run clean