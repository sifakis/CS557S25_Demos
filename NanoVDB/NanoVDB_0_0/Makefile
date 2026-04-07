include ../config.mk

.PHONY: all clean

all: NanoVDB_0_0

NanoVDB_0_0: main.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $<

clean:
	rm -f NanoVDB_0_0
