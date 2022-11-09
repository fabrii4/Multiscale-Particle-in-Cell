cxx := nvcc
#src = $(wildcard *.cu) #plasma.cu Poisson.cu
obj_dir = ./obj
src_dir = .
src = ./plasma.cu ./Poisson.cu
#src  := $(wildcard $(src_dir)/*.cu)
obj := $(patsubst $(src_dir)/%.cu,$(obj_dir)/%.o,$(src))
#obj = $(src:.c=.o)
#FLAGS
cpp := -std=c++11 -O3
cuda := -arch=sm_50 --use_fast_math -lcufft --default-stream per-thread -lcudadevrt -lcudart -w -lineinfo -DTHRUST_DEBUG #-G -g # -rdc=true 
#cppcuda = -arch=sm_50 --use_fast_math -lcufft -lineinfo --default-stream per-thread -lcudadevrt -rdc=true 
flags := $(cuda) $(cpp)

MAKEFLAGS += -j4


all: plasma
print:
	@echo $(src)
	@echo $(obj)

plasma: $(obj)
	$(cxx) $(obj) -o plasma $(flags)

$(obj_dir)/%.o: $(src_dir)/%.cu | $(obj_dir)
	$(cxx) -I. -dc $< -o $@ $(flags)

$(obj_dir):
	@mkdir -p $(obj_dir)


.phony: clean
clean:
	@rm -rf $(obj_dir)

#nvcc -dc plasma.cu Poisson.cu -arch=sm_50 --use_fast_math -std=c++11 -O3
