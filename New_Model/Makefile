CXX = g++
CXXFLAGS = -std=c++11 -g -Wall 
LDFLAGS = -lutil -lboost_iostreams -lboost_system -lboost_filesystem 
OBJECTS = main.o ActivityDetection.o Misc_utils.o bal_bst.o oned_range.o node.o
PROG = exam_main

OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)

$(PROG):$(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(PROG) $(OBJECTS) $(LIBS) $(LDFLAGS)
main.o: main.cpp ActivityDetection.h 
	$(CXX) $(CXXFLAGS) -c main.cpp
ActivityDetection.o: ActivityDetection.h ActivityDetection.cpp 
	$(CXX) $(CXXFLAGS) -c ActivityDetection.cpp
Misc_utils.o: Misc_utils.h Misc_utils.cpp gnuplot-iostream.h
	$(CXX) $(CXXFLAGS) -c Misc_utils.cpp
bal_bst.o: bal_bst.h bal_bst.cpp Misc_utils.cpp node.cpp
	$(CXX) $(CXXFLAGS) -c bal_bst.cpp
oned_range.o: oned_range.cpp oned_range.h node.cpp node.h
	$(CXX) $(CXXFLAGS) -c oned_range.cpp
node.o: node.h node.cpp
	$(CXX) $(CXXFLAGS) -c node.cpp
clean:	
	rm exam_main $(OBJECTS) 
