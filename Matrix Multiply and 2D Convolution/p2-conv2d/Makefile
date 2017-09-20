OBJS=conv2d.o readjpeg.o

%.o: %.cpp readjpeg.h
	g++ -fopenmp -O2 -c $<

all: $(OBJS)
	g++ -fopenmp $(OBJS) -o conv2d -ljpeg