CXX := gcc

NVCC := nvcc

CPPFLAGS := -std=c++11 -O2 -g

# we use math.h
LIBS := -lm

# use the right gencodes for your setup
NVCCFLAGS := -O2 -arch=sm_86 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_75,code=sm_75

LDFLAGS :=  -L/usr/local/cuda/lib -lcuda -lcudart -lcurand -lstdc++

BUILDDIR := ./build
OBJDIR := $(BUILDDIR)/obj
TARGETDIR := $(BUILDDIR)/bin

SRC := main.cpp \
	   parse_text.cpp  \
	   test.cu \
	   monkeys_kernels.cu \
	   monkeys.cu

OBJ := $(patsubst %.cu,  $(OBJDIR)/%.o, $(filter %.cu,  $(SRC))) \
       $(patsubst %.cpp, $(OBJDIR)/%.o, $(filter %.cpp, $(SRC)))

default : clean $(TARGETDIR)/monkeyTheorem

clean :
	rm -f $(TARGETDIR)/* $(OBJDIR)/*

$(OBJDIR)/%.o : %.cpp
	@mkdir -p $(dir $@)
	@$(CXX) -c $(CPPFLAGS) $< -o $@

$(OBJDIR)/%.o : %.cu
	@mkdir -p $(dir $@)
	@$(NVCC) -c $(NVCCFLAGS) $< -o $@

$(TARGETDIR)/monkeyTheorem : $(OBJ)
	@mkdir -p $(dir $@)
	@$(CXX) $(OBJ) $(LIBS) $(LDFLAGS) -o $@
