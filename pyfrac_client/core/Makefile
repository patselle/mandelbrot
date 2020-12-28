CC=@gcc
CFLAGS=-fpic -v -O4 -pipe -march=native -mtune=native
LDFLAGS=-lmpfr -lgmp

OBJS=pyfrac_utils.o pyfrac_frame.o pyfrac_bmp.o pyfrac_zoom.o

all: clean $(OBJS)
	$(CC) -shared -o lib/libpyfrac.so $(addprefix obj/, $(OBJS)) $(LDFLAGS)

debug: CFLAGS += -g -DDEBUG -O0
debug: all

clean:
	@mkdir -p obj lib bin
	@rm -rf obj/* lib/* bin/*

%.o : %.c
	$(CC) -c $(CFLAGS) -o obj/$@ $<
