IDIR=../include
CC=g++
NVCC=nvcc
LDIR=/usr/local/cuda/lib64
CFLAG=-I$(IDIR) -g
ODIR=./obj

CULIBS=-lcudart -lcublas 
CUFLAG= -arch=sm_70

_DEPS= types.h
DEPS=$(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ= auxf.o gpu_utilities.o chez_util.o deep_ice_chez.o checking.o
OBJ=$(patsubst %,$(ODIR)/%,$(_OBJ))

deep_ICE_GPU: $(ODIR)/main.o $(OBJ)
	$(NVCC) -rdc=true -o $@ $^  $(CULIBS) $(CUFLAG)

$(ODIR)/%.o: %.cpp $(IDIR)/%.h $(DEPS) 
	$(CC) -c -o $@ $<  $(CFLAG)

$(ODIR)/%.o: %.cu  $(DEPS)
	$(NVCC) -c  -rdc=true -o $@ $< $(CFLAG) $(CULIBS)  $(CUFLAG)

.PHONY: clean

clean: 
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~
