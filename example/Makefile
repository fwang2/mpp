CC = g++
INCLUDES = -I../include
LIBS = -L../lib -lmatrix
CFLAGS = $(INCLUDES)

EXECUTABLES = testMpp testfileIO testimageIO testMatrix testEig 

.PHONY: all clean

all: $(EXECUTABLES)

testMpp: testMpp.o
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)
testEig: testEig.o
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

testfileIO: testfileIO.o
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

testimageIO: testimageIO.o
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

testMatrix: testMatrix.o
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

clean:
	rm -f *.o $(EXECUTABLES)

%.o: %.cpp
	$(CC) -c $(CFLAGS) $< -o $@
