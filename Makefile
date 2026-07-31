CXX      := g++
NVCC     := /usr/local/cuda/bin/nvcc

# RTX 5090 is Blackwell sm_120; -DUSE_CUDA exposes the GPU physics path
CXXFLAGS  := -std=c++17 -O3 -march=native -Wall -Wextra -Wno-deprecated-declarations \
             -DUSE_CUDA -I/usr/local/cuda/include
NVCCFLAGS := -std=c++17 -O3 -arch=sm_120 -DUSE_CUDA \
             --compiler-options "-Wall -Wno-deprecated-declarations"
LDFLAGS   := -lGLEW -lGL -lGLU -lglut -pthread \
             -L/usr/local/cuda/lib64 -lcudart

# Targets
TARGET1 := physicssim
TARGET2 := dynamicpaths
TARGET3 := plannersim.bin
SCRIPT3 := plannersim

# Source files
SRCS1 := sim.cpp
SRCS2 := dynamicpaths.cpp
SRCS3 := main.cpp particle.cpp pathFinding.cpp robot.cpp gpu_renderer.cpp particle_cuda.cu

# Object files
OBJS1 := $(SRCS1:.cpp=.o)
OBJS2 := $(SRCS2:.cpp=.o)
OBJS3 := particle.o pathFinding.o robot.o main.o gpu_renderer.o particle_cuda.o

.PHONY: all clean

# Build all executables by default
all: $(TARGET1) $(TARGET2) $(SCRIPT3)

# Generate the plannersim launcher script
$(SCRIPT3): $(TARGET3)
	@printf '#!/usr/bin/env sh\n# Auto-generated GPU launcher — edit Makefile to change\nexport MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA\nexport __NV_PRIME_RENDER_OFFLOAD=1\nexport __GLX_VENDOR_LIBRARY_NAME=nvidia\nexec "$$(dirname "$$0")/$(TARGET3)" "$$@"\n' > $@
	@chmod +x $@
	@echo "Generated ./$@ wrapper"

# Build the original physics simulation
$(TARGET1): $(OBJS1)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Build the dynamic pathfinding simulation
$(TARGET2): $(OBJS2)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Build the modular particle system with pathfinding (CUDA-enabled)
$(TARGET3): $(OBJS3)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@rm -f $(SCRIPT3)  # force script regeneration on binary rebuild

# CUDA physics object
particle_cuda.o: particle_cuda.cu particle.h particle_cuda.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

# Clean all build artifacts
clean:
	rm -f $(OBJS1) $(OBJS2) $(OBJS3) particle_cuda.o \
	      $(TARGET1) $(TARGET2) $(TARGET3) $(SCRIPT3)

# Pattern rule for C++ object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<
