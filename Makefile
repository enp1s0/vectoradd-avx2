CXX=g++
CXXFLAGS=-std=c++17 -fopenmp -mavx2 -O2

TARGET=vecadd.out

$(TARGET):main.cpp
	$(CXX) $< -o $@ $(CXXFLAGS)
  
clean:
	rm -f $(TARGET)
