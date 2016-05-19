SIZE ?= 256
LIST = c1 c1_16 c2 c2_16 c3 c3_16
NVCC = nvcc -arch=sm_20 -Xptxas -v

all: $(LIST)

clean:
	rm -f $(LIST)

c1: c1.cu
	$(NVCC) c1.cu -o c1

c1_16: c1.cu
	$(NVCC) c1.cu -DBLOCK_H=16 -o c1_16

c2: c1.cu
	$(NVCC) c1.cu -DHAVE_PITCH -o c2

c2_16: c1.cu
	$(NVCC) c1.cu -DHAVE_PITCH -DBLOCK_H=16 -o c2_16

c3: c1.cu
	$(NVCC) c1.cu -DHAVE_PITCH -DHAVE_SHARED -o c3

c3_16: c1.cu
	$(NVCC) c1.cu -DHAVE_PITCH -DHAVE_SHARED -DBLOCK_H=16 -o c3_16

test: c1 c2 c1_16 c2_16 c3 c3_16
	@for a in $(LIST); do echo -n "$$a: "; ./$$a $(SIZE) $(SIZE) $(SIZE); done

check: c1 c2 c1_16 c2_16 c3 c3_16
	@for a in $(LIST); do echo -n "$$a: "; ./$$a $(SIZE) $(SIZE) $(SIZE) --check | tail -1; done
