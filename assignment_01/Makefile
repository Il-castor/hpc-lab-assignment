UTIL_DIR = polybench
INCPATHS = -I$(UTIL_DIR)

FILE ?= durbin_originale.c

EXE = $(FILE:.c=.out)
SRC = $(FILE)
HEADERS = durbin.h

SRC += $(UTIL_DIR)/polybench.c

CC=gcc
LD=ld
OBJDUMP=objdump

OPT=-O2 -g -fopenmp
CFLAGS=$(OPT) -I. $(EXT_CFLAGS)
LDFLAGS=-lm $(EXT_LDFLAGS)

.PHONY: all exe clean

all : exe

exe : $(EXE)

$(EXE) : $(SRC)
	$(CC) $(CFLAGS) $(INCPATHS) $^ -o $@ $(LDFLAGS)

clean :
	-rm -vf -vf $(EXE) *~ *.out

run: $(EXE)
	./$(EXE)
