/*
 * test.cpp
 *
 * Tests simplegraph and illustrates how to do basic network operations.
 * Generates a basic ER G(N,p) type of random graph to show how to add vertices and edges.
 * Uses basic (rubiish) random number generator built into C++ but that 
 * might be good enough.
 *
 *  Created on: 28 Feb 2014
 *      Author: time
 */
#include "simplegraph.h"
#include <vector>
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <iostream>
#include <iomanip>
#include <fstream> /* Needed for file output */
//#include <set>

using namespace std;


int main(int argc, char *argv[]) {

	cout << "Arguments are: N <k> (d) where arguments are separated by spaces and put after command name" << endl; 
	cout << "               N  = number of vertices" << endl; 
	cout << "              <k> = average degree" << endl; 
	cout << "               d  = if letter d is third argument then debugging options will be turned on" << endl; 

// *************************************************************
// User defined variables - default values then command line values are processed

    // Number of vertices
	int N=10;

    // Average degree required
	double Kexp=3.0;

    // if true, then some extra debugging stuff is swicthed on.
    // First random numbers will always be the same every time you run it (fixed seed).
    // Also produces some output on screen, use only for small networks.
    bool debugon = true;

// End of user defined section
// *************************************************************

// First process command line

	for (int a=1; a<argc; a++){
		switch (a) {
		case 1: 
            N=atoi(argv[1]);
			break;
		case 2:
			Kexp=atoi(argv[2]);
			break;
		case 3: 
			debugon= (*argv[3] == 'd');
			break;
		default:
			cout << " Too many arguments" << endl;
		}
	}

	
	// for use of simple built in C random number generator see
    // http://www.cplusplus.com/reference/cstdlib/rand/
	/* initialize random seed: */
	if (debugon){
		srand(0); // use the fixed seed for debugging
	}
	else{
		srand (time(NULL)); // Use the time version for real runs
	}
	
   // start by defining an empty graph 
	simplegraph g;

    // find the necessary probability we are creating a G(N,p) type of random graph
	double p = Kexp/(N-1);
	double numberedges = N*Kexp/2.0; // always use floating point for constants including 1 and 0 unless want integer arithmetic 
	for (int v=0; v<N; v++){	
		g.addVertex();
	}
	if (debugon){
		cout << " adding edges " << endl;
	}
	for (int e=0; e<numberedges; e++){
		int source= rand() % N;
		// next part selects random target vertex which is not the same as the source
		int target= source;
		while (target==source) {
			target=rand() % N;
		}
		int i = g.addEdgeSlowly(source,target);
		if (debugon){
		   cout << source << " - " << target << endl;
		}
	}
	cout << "Network has " << g.getNumberVertices() << " vertices" << endl;
	cout << "Network has " << g.getNumberEdges() <<  " edges" << endl;

	// Write out list of edges to file if you want
	// WARNING with file names and all strings in C++ \ has a special meaning.
	// so for directories on Windows use either \\ for a single backslash or forwards slash / may work
	//g.write("c:/DATA/CandN/edgelist.txt");
	//g.write("c:\\DATA\\CandN\\edgelist.txt");
    // Thie output files will appear in same directory as source code (this file) if you give no directories in filename
    g.write("edgelist.txt");


	// Studying the degree distribution
	vector<int> dd;
	g.getDegreeDistribution(dd);
	
    // output on screen
    cout << "k \t n(k)" << endl;
	for (int k=0; k<dd.size(); k++){
		cout << k << " \t " << dd[k] << endl; 
	}

	// Write degree distribution to a file.
	// This declares fout to be like cout but everything sent to fout goes to the file
	// named in the declation
    //ofstream fout("c:\DATA\CandN\degreedistribution.dat");
	ofstream fout("degreedistribution.dat");
	fout << "k \t n(k)" << endl;
	for (int k=0; k<dd.size(); k++){
		fout << k << " \t " << dd[k] << endl; 
	}
	fout.close();

	return 0;
}
