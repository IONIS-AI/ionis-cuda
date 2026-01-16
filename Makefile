# Makefile for ki7mt-ai-lab-cuda
#
# Local development only - does not affect COPR/rpkg builds
#
# Usage:
#   make              # Show help
#   make build        # Compile CUDA bridge (requires nvcc)
#   make install      # Install to system (requires sudo)
#   make test         # Run verification tests
#   make distclean    # Remove all build artifacts

SHELL := /bin/bash
.PHONY: help build install uninstall test distclean check-cuda

# Package metadata
NAME     := ki7mt-ai-lab-cuda
VERSION  := $(shell cat VERSION 2>/dev/null || echo "0.0.0")
PREFIX   := /usr
LIBDIR   := $(PREFIX)/lib64/$(NAME)
INCDIR   := $(PREFIX)/include/$(NAME)
DATADIR  := $(PREFIX)/share/$(NAME)/cuda

# CUDA settings
NVCC     := nvcc
CUDA_ARCH := sm_90
NVCC_FLAGS := -c -O3 -arch=$(CUDA_ARCH)

# Build directory
BUILDDIR := build

# Source files
CUDA_SRC := src/cuda/bridge.cu
CUDA_HDR := src/cuda/bridge.h
CUDA_OBJ := src/cuda/bridge.o

# Default target
.DEFAULT_GOAL := help

help:
	@printf "\n"
	@printf "Package : $(NAME)\n"
	@printf "Version : $(VERSION)\n"
	@printf "\n"
	@printf "Usage: make [target]\n"
	@printf "\n"
	@printf "Targets:\n"
	@printf "  help        Show this help message\n"
	@printf "  build       Compile CUDA bridge (requires nvcc)\n"
	@printf "  install     Install to system (PREFIX=$(PREFIX), requires sudo)\n"
	@printf "  uninstall   Remove installed files (requires sudo)\n"
	@printf "  test        Run verification tests\n"
	@printf "  distclean   Remove all build artifacts\n"
	@printf "  check-cuda  Verify CUDA toolkit installation\n"
	@printf "\n"
	@printf "Variables:\n"
	@printf "  PREFIX      Installation prefix (default: /usr)\n"
	@printf "  DESTDIR     Staging directory for packaging\n"
	@printf "  CUDA_ARCH   CUDA architecture (default: sm_90)\n"
	@printf "\n"
	@printf "Examples:\n"
	@printf "  make check-cuda               # Verify CUDA installation\n"
	@printf "  make build                    # Compile bridge.o\n"
	@printf "  sudo make install             # Install to /usr\n"
	@printf "  make PREFIX=/usr/local install # Install to /usr/local\n"
	@printf "  DESTDIR=/tmp/stage make install # Stage for packaging\n"
	@printf "  make CUDA_ARCH=sm_86 build    # Build for RTX 3090\n"

check-cuda:
	@printf "Checking CUDA installation...\n"
	@command -v $(NVCC) >/dev/null 2>&1 || { \
		printf "ERROR: nvcc not found in PATH\n"; \
		printf "Install CUDA toolkit or add to PATH\n"; \
		exit 1; \
	}
	@printf "  nvcc:     $(shell $(NVCC) --version | grep release)\n"
	@printf "  arch:     $(CUDA_ARCH)\n"
	@printf "CUDA check passed.\n"

build: $(BUILDDIR)/.built

$(BUILDDIR)/.built: $(CUDA_SRC) $(CUDA_HDR) VERSION
	@printf "Building $(NAME) v$(VERSION)...\n"
	@mkdir -p $(BUILDDIR)/lib $(BUILDDIR)/include $(BUILDDIR)/cuda
	@# Check for nvcc
	@command -v $(NVCC) >/dev/null 2>&1 || { \
		printf "WARNING: nvcc not found, using pre-compiled bridge.o\n"; \
		cp $(CUDA_OBJ) $(BUILDDIR)/lib/bridge.o; \
	}
	@# Compile CUDA if nvcc available
	@if command -v $(NVCC) >/dev/null 2>&1; then \
		printf "  Compiling bridge.cu -> bridge.o ($(CUDA_ARCH))...\n"; \
		$(NVCC) $(NVCC_FLAGS) -o $(BUILDDIR)/lib/bridge.o $(CUDA_SRC); \
	fi
	@# Copy header
	@cp $(CUDA_HDR) $(BUILDDIR)/include/
	@printf "  %-30s -> build/include/bridge.h\n" "$(CUDA_HDR)"
	@# Copy sources for reference
	@cp src/cuda/*.cu $(BUILDDIR)/cuda/
	@printf "  %-30s -> build/cuda/\n" "src/cuda/*.cu"
	@touch $(BUILDDIR)/.built
	@printf "Build complete.\n"

install: build
	@printf "Installing to $(DESTDIR)$(PREFIX)...\n"
	install -d $(DESTDIR)$(LIBDIR)
	install -d $(DESTDIR)$(INCDIR)
	install -d $(DESTDIR)$(DATADIR)
	install -m 644 $(BUILDDIR)/lib/bridge.o $(DESTDIR)$(LIBDIR)/
	install -m 644 $(BUILDDIR)/include/bridge.h $(DESTDIR)$(INCDIR)/
	install -m 644 $(BUILDDIR)/cuda/*.cu $(DESTDIR)$(DATADIR)/
	@printf "Installed:\n"
	@printf "  Library:  $(DESTDIR)$(LIBDIR)/bridge.o\n"
	@printf "  Header:   $(DESTDIR)$(INCDIR)/bridge.h\n"
	@printf "  Sources:  $(DESTDIR)$(DATADIR)/*.cu\n"

uninstall:
	@printf "Uninstalling from $(DESTDIR)$(PREFIX)...\n"
	rm -rf $(DESTDIR)$(LIBDIR)
	rm -rf $(DESTDIR)$(INCDIR)
	rm -rf $(DESTDIR)$(PREFIX)/share/$(NAME)
	@printf "Uninstall complete.\n"

test: build
	@printf "Running tests for $(NAME) v$(VERSION)...\n"
	@printf "\n"
	@# Test 1: Check build outputs exist
	@printf "[TEST] Build outputs exist... "
	@test -f $(BUILDDIR)/lib/bridge.o && \
	 test -f $(BUILDDIR)/include/bridge.h && \
	 printf "PASS\n" || { printf "FAIL\n"; exit 1; }
	@# Test 2: Check object file is valid ELF
	@printf "[TEST] bridge.o is valid object... "
	@file $(BUILDDIR)/lib/bridge.o | grep -q "ELF\|relocatable" && \
	 printf "PASS\n" || { printf "FAIL\n"; exit 1; }
	@# Test 3: Check header file has expected content
	@printf "[TEST] Header defines WsprSpotC struct... "
	@grep -q "typedef struct.*WsprSpotC" $(BUILDDIR)/include/bridge.h && \
	 printf "PASS\n" || { printf "FAIL\n"; exit 1; }
	@# Test 4: Check header has CUDA API functions
	@printf "[TEST] Header declares CUDA functions... "
	@grep -q "cuda_host_alloc_pinned" $(BUILDDIR)/include/bridge.h && \
	 grep -q "cuda_kernel_validate_spots" $(BUILDDIR)/include/bridge.h && \
	 printf "PASS\n" || { printf "FAIL\n"; exit 1; }
	@# Test 5: Check source files copied
	@printf "[TEST] CUDA source files copied... "
	@test -f $(BUILDDIR)/cuda/bridge.cu && \
	 test -f $(BUILDDIR)/cuda/kernels.cu && \
	 printf "PASS\n" || { printf "FAIL\n"; exit 1; }
	@printf "\nAll tests passed.\n"

distclean:
	@printf "Cleaning build artifacts...\n"
	rm -rf $(BUILDDIR)
	@printf "Clean complete.\n"
