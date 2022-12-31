all: CoFIM CELF Degree SingleDiscount

CoFIM : CoFIM.o Graph.o Node.o
	g++ -O3 -o CoFIM CoFIM.o Graph.o Node.o

CELF : CELF.o Graph.o Node.o
	g++ -O3 -o CELF CELF.o Graph.o Node.o

Degree : Degree.o Graph.o Node.o
	g++ -O3 -o Degree Degree.o Graph.o Node.o

SingleDiscount : SingleDiscount.o Graph.o Node.o
	g++ -O3 -o SingleDiscount SingleDiscount.o Graph.o Node.o

CoFIM.o : CoFIM.cpp
	g++ -O3 -c CoFIM.cpp

CELF.o : CELF.cpp
	g++ -O3 -c CELF.cpp

Degree.o : Degree.cpp
	g++ -O3 -c Degree.cpp

SingleDiscount.o : SingleDiscount.cpp
	g++ -O3 -c SingleDiscount.cpp

Graph.o : Graph.cpp
	g++ -O3 -c Graph.cpp

Node.o : Node.cpp
	g++ -O3 -c Node.cpp

clean :
	rm Node.o Graph.o CoFIM.o CoFIM CELF.o CELF Degree.o Degree SingleDiscount.o SingleDiscount
