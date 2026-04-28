include ../config.mk

.PHONY: all clean

OBJS = main.o Utility.o

all: NanoVDB_0_2

NanoVDB_0_2: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

%.o: %.cpp Utility.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

clean:
	rm -f NanoVDB_0_2 $(OBJS)
