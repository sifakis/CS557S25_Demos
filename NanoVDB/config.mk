# Shared build configuration — included by each example's Makefile.
# Override compiler with: make CXX=icc

CXX      = g++
CXXSTD   = -std=c++17
OPT      = -O3
WARN     = -Wall -w
OMP      = -fopenmp
INCLUDES = -I..

CXXFLAGS = $(CXXSTD) $(OPT) $(WARN) $(OMP)
