
#include <iostream>
#include <vector>
#include <cmath>
//#include <cstdlib>
#include <algorithm>
#include <new>
#include <fstream>


#include <stdio.h>
#include <stdlib.h>
#include <time.h> 
#include <pthread.h>
#include <sys/mman.h>
#include <assert.h>
#include <float.h>
#include <unistd.h>


#include <cstdlib>
#include <cstdio>
#include <errno.h>
#include <fstream>
#include <sstream>
#include <string.h>
#include <linux/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <iomanip>
#include <sys/time.h>
#include <sys/resource.h>

using namespace std;

typedef struct Node{
        vector< float *>v;
        float median;
        int dimension;
        int is_leaf;
        Node *left;
        Node *right;
}Node;


typedef struct QUERY{
  float *qpoint;
  vector< float*>neighbours;
  vector< float>distances;
  float min_distance;   //unnecessary
  float max_distance;
  int points_attatched;
}QUERY;


typedef struct param{
  int count;
  Node *node;
  vector< float* > v;
}param;

typedef struct q_param{
  QUERY *query_array;
  long int begin, end;
  Node *node;
  int count;
}q_param;


float findMedian(vector< float *> vec, int dimension);

void * insertThread(void *input);

void insert(Node *node,  vector< float* > &v);

void *queryThreadFunc(void *input);

void findClosestNeighbour(QUERY *query, Node* node);

void assignNearestNeighbours( QUERY *q , Node *n);

int IndexToBeRemoved(float sum, QUERY *query);



typedef char byte;
  static unsigned long int totalDimension;
  static unsigned long int totalPoints;
  static unsigned long int num_queries;
  static unsigned long int k;
  static int county;        //num of threads


int main(int argc, char*argv[]) {
    if(argc != 5){
      cout << "Usage: ./k-nn n_cores training_file query_file result_file" << endl;
      return 0;
    }

    srand (time(NULL));

    
//#######################################################  Input file read #########
  static int n_cores = atoi(argv[1]);
  county = n_cores;   //num of threads
  char file_type_string[9];     //Just having a byte at the end as a safety measure. Might use file_type_string as a string later.
  file_type_string[8] = '\0';

 int fd = open(argv[2], O_RDONLY); if(fd < 0) cout << "Error opening training file" << endl;
 struct stat status;
 int foo = fstat(fd, &status); assert(foo == 0);

 void *ip_file_begin = mmap(NULL, status.st_size, PROT_READ, MAP_SHARED|MAP_POPULATE, fd, 0);       //Get the training file in memory
  // Check for mmap error 
  foo = madvise(ip_file_begin, status.st_size, MADV_SEQUENTIAL|MADV_WILLNEED); assert(foo == 0);
  
  foo = close(fd); assert(foo == 0);

  char *iterator = (char *)ip_file_begin;

  for (int j = 0; j < 8; j++)   //File name
    file_type_string[j] = *(iterator + j);
  

  iterator = iterator + 8;
  unsigned long int id = *((unsigned long int *)iterator) ;
  iterator += 8;        //Now the iterator points to number of points
  totalPoints = *((unsigned long int *)iterator);
  iterator += 8;
  totalDimension = *((unsigned long int *)iterator) ;
  iterator += 8;

  cout << "Input file " << id << " read with dimension " << totalDimension << "and " << totalPoints << "points " << endl;


//################################################################################ Assigning the data points to vector

  vector< float* >vectors;     //making a vector of floating pointers

  float *iter = (float *)iterator;
  for(int i=0; i< totalPoints  ; i++){                 //Putting fake data into our lovely vector
    vectors.push_back( iter );
    iter = iter + totalDimension;
  }

//#############################################################################  Creating the nodes. This step is taking a long time.

Node *NODES = new Node[totalPoints+1];    //Create Node array. Size is one extra because we will start from 1st index

NODES[1].dimension = 0;    //The root

for(int i =1; i<=totalPoints; i++){

  if(2*i > totalPoints)
  {
//    cout << "i = " << i << endl;
    NODES[i].left  = NULL;
    NODES[i].right = NULL;
    NODES[i].is_leaf = 1;
    continue; 
  }

  if((2*i + 1) > totalPoints)
  {
    
    NODES[i].left  = &NODES[2*i];
    NODES[i].left->dimension  = NODES[i].dimension + 1;
    if(NODES[i].dimension == totalDimension )
        NODES[i].left->dimension = 0; 
    
    NODES[i].right = NULL;
    continue;
  }

  NODES[i].left  = &NODES[2*i];
  NODES[i].right = &NODES[2*i + 1];

  NODES[i].right->dimension = NODES[i].dimension + 1;
  if(NODES[i].dimension == totalDimension )
    NODES[i].right->dimension = 0; 
    
  NODES[i].left->dimension  = NODES[i].dimension + 1;
  if(NODES[i].dimension == totalDimension )
    NODES[i].left->dimension = 0; 

  NODES[i].is_leaf = 0;

}

//###########################################################   Split the vector and create two new ones, l and r.

float median = findMedian(vectors, NODES[1].dimension);    //Find the median on dimension 0. 
                            //Important: Max dimension value passed to the findMedian should be totalDimension - 1

NODES[1].median = median; //root

vector<  float* > right;
vector<  float* > left;           // declare two sub arrays

for (int i=0; i<vectors.size(); i++){
    if(*(vectors[i]+ NODES[1].dimension) <= median )
        left.push_back(vectors[i]);
  
     else
       right.push_back(vectors[i]);

  }


//###########################################################        Create threads

pthread_t threadOne; 
pthread_t threadTwo; 

param *par1 = new param();
param *par2 = new param();


if(county == 1){                ////Sequential Construction
  insert(NODES[1].left, left);
  insert(NODES[1].right, right);
  delete par1;
  delete par2;
}
else{                     ////Parallel Construction
int totalThreads = county/2;
//param *par1 = new param();
par1->v = right;
par1->count = totalThreads;
par1->node = NODES[1].right;

if(pthread_create(&threadOne, NULL, insertThread, (void *)par1)) {
cout<< "Error creating thread\n";
return 1;
}

//param *par2 = new param();
par2->v = left;
par2->count = county - totalThreads;
par2->node = NODES[1].left;

if(pthread_create(&threadTwo, NULL, insertThread, (void *)par2)) {
cout<< "Error creating thread\n";
return 1;
}

}


//########################################################### Create query structs here


//Do the thread join after this

char file_type_string_query[9];     //Just having a byte at the end as a safety measure. Might use file_type_string as a string later.
  file_type_string_query[8] = '\0';

 int fd_q = open(argv[3], O_RDONLY); if(fd < 0) cout << "Error opening query file" << endl;
 struct stat status_q;
 int bar = fstat(fd_q, &status_q); assert(bar == 0);

 void *q_file_begin = mmap(NULL, status_q.st_size, PROT_READ, MAP_SHARED|MAP_POPULATE, fd_q, 0);       //Get the training file in memory
  // Check for mmap error 
  bar = madvise(q_file_begin, status_q.st_size, MADV_SEQUENTIAL|MADV_WILLNEED); assert(bar == 0);
  
  bar = close(fd_q); assert(foo == 0);

  char *iterator_q = (char *)q_file_begin;

  for (int j = 0; j < 8; j++)   //File name
    file_type_string_query[j] = *(iterator_q + j);
  
  iterator_q = iterator_q + 8;
  unsigned long int id_q = *((unsigned long int *)iterator_q) ;
  iterator_q += 8;        //Now the iterator points to number of points
  num_queries = *((unsigned long int *)iterator_q);
  iterator_q += 8;
  totalDimension = *((unsigned long int *)iterator_q) ;
  iterator_q += 8;
  k = *((unsigned long int *)iterator_q)  ;
  iterator_q += 8;


  cout << "Query file " << id_q << " read with dimension " << totalDimension << "and  k =" << k
  << " num of queries = " << num_queries << endl;


QUERY *query_array = new QUERY[num_queries];

iter = (float *)iterator_q;   //Data starts here

                                        
for(int i=0; i<num_queries ; i++){    //Initialize the query structs
  query_array[i].qpoint = iter;
  query_array[i].min_distance = FLT_MAX;
  query_array[i].max_distance = 0.0;
  query_array[i].points_attatched = 0;

  iter = iter + totalDimension;
} 



//########################################################### Joining the construction threads

if(county > 1){
if(pthread_join(threadOne, NULL)) {
fprintf(stderr, "Error joining thread\n");
return 2;
}

if(pthread_join(threadTwo, NULL)) {
fprintf(stderr, "Error joining thread\n");
return 2;
}

delete par1;

delete par2;

}

//###########################################################     searching


if(county == 1){								////Sequential searching
	for(int i =0; i<num_queries; i++){
		QUERY *queryp = &query_array[i];
		findClosestNeighbour(queryp , &NODES[1]);
	}
}

else{       ////Parallel searching
//###########################################################     Handle for case where query is less than number of threads

int threads = county/2;

q_param *parA = new q_param();
parA->begin = 0;
parA->end = num_queries/2;
parA->node = &NODES[1];
parA->query_array = query_array;
parA->count = threads;


q_param *parB = new q_param();
parB->begin = num_queries/2 +1;
parB->end = num_queries-1;
parB->node = &NODES[1];
parB->query_array = query_array;
parB->count = county - threads;


pthread_t threadA; 
if(pthread_create(&threadA, NULL, queryThreadFunc, (void *)parA)) {
cout<< "Error creating thread\n";
return 1;
}


pthread_t threadB; 
if(pthread_create(&threadB, NULL, queryThreadFunc, (void *)parB)) {
cout<< "Error creating thread\n";
return 1;
}


if(pthread_join(threadA, NULL)) {
fprintf(stderr, "Error joining thread\n");
return 2;
}

if(pthread_join(threadB, NULL)) {
fprintf(stderr, "Error joining thread\n");
return 2;
}

delete parA;
delete parB;

}

///////////////////////////////////// Result file creation



  char resultFileName[] = "RESULT00";

  unsigned long int resultFileID = rand() % 1000;

  ofstream out(argv[4], ofstream::binary); 
  
  if(out.good() == 0) 
    cout << "Write operation failed" << endl;

  out.write(reinterpret_cast<const char*>(&resultFileName), sizeof(unsigned long int));
  
  out.write(reinterpret_cast<const char*>(&id), sizeof(unsigned long int));
  
  out.write(reinterpret_cast<const char*>(&id_q), sizeof(unsigned long int));
  
  out.write(reinterpret_cast<const char*>(&resultFileID), sizeof(unsigned long int)); 
  
  out.write(reinterpret_cast<const char*>(&num_queries), sizeof(unsigned long int));
  
  out.write(reinterpret_cast<const char*>(&totalDimension), sizeof(unsigned long int));
  
  out.write(reinterpret_cast<const char*>(&k), sizeof(unsigned long int));
  

  QUERY *queryPoint;
  float y;

  for(int i=0 ; i<num_queries ; i++)
  {
    queryPoint = &query_array[i]; //Iterate through queries
    for(int j=0; j<k ; j++)
    {
      for(int m=0; m<totalDimension; m++){
          y = *(queryPoint->neighbours[j] + m);
          out.write(reinterpret_cast<const char*>(&y), sizeof(float));      
      }
    }
  }

  ///Read and check result file

//######################################### Memory deallocation
  delete[] NODES;
  delete[] query_array;
//munmap 
munmap(ip_file_begin, status.st_size);
munmap(q_file_begin, status_q.st_size);

return 0;

}


///////////////////////////////////////////// Function declarations





void *queryThreadFunc(void *input){
  
  q_param *par = (q_param *)input;
  par->count = par->count -1;			//one thread used up

  if(par->count <= 0){			//Only this thread left
  for(int i = par->begin ; i<=par->end ; i++){
    findClosestNeighbour((par->query_array + i) , par->node);
    }
  }

  else{
    //#########################     do (par->end - par->begin +1 ) for symmetrical distribution of work

  	int partition = (par->end - par->begin +1) / (par->count+1);
  	q_param *parnew = new q_param();
	parnew->begin = par->begin + partition;
	parnew->end = par->end;
	parnew->node = par->node;
	parnew->query_array = par->query_array;
	parnew->count = par->count;


	pthread_t threadnew; 

	if(pthread_create(&threadnew, NULL, queryThreadFunc, (void *)parnew)) {
		cout<< "Error creating thread\n";
	return NULL;
	}

	for(int i = par->begin ; i<=(par->begin + partition-1) ; i++)
    	findClosestNeighbour((par->query_array + i) , par->node);

    if(pthread_join(threadnew, NULL)) {
		fprintf(stderr, "Error joining thread\n");
		return NULL;
	}
  
  delete parnew;

  }

  return NULL;
}





void findClosestNeighbour(QUERY *query, Node* node){

  if(node == NULL)
    return;

  if(node->is_leaf == 1){
      assignNearestNeighbours(query , node);
      return;
    }

  if(*(query->qpoint + node->dimension) > node->median){
      findClosestNeighbour(query , node->right);
      float hyperplaneDistance = sqrt(fabs( (*(query->qpoint + node->dimension)) - (node->median) ) ) ;
      if( (query->points_attatched < k) ||  (query->max_distance > hyperplaneDistance )  )
        	{
            findClosestNeighbour(query , node->left);
          }
      return;
  }
  else{
      findClosestNeighbour(query , node->left);
      float hyperplaneDistance = sqrt(fabs( (*(query->qpoint + node->dimension)) - (node->median) ) ) ;
      if( (query->points_attatched < k) ||  (query->max_distance > hyperplaneDistance )  )
          {
            findClosestNeighbour(query , node->right);
          }
      return;
  }

  return;

}



void assignNearestNeighbours( QUERY *q, Node *n){
  float sum;
  
  //cout << "Entering assignnode " << endl;
  for(int i=0; i < n->v.size() ; i++)
  {
    sum = 0.0;    //each data point starts with sum = 0
    
    for(int j=0; j< totalDimension ; j++)
      sum += ( ( *(n->v[i] + j) - *(q->qpoint + j) ) * ( *(n->v[i] + j) - *(q->qpoint + j) )  ); 

    float dist = sqrt(sum);
    
    if(q->points_attatched >= k){           //Check if the query already has k neighbours
        int index = IndexToBeRemoved(dist , q);
        if(index == -1) continue;   //Dont add this point
        
        q->neighbours[index] = n->v[i];
        q->distances[index]  = dist;
        if(dist < q->min_distance)
          q->min_distance = dist;

        float max = q->distances[0];
        for(int alpha=0; alpha<k; alpha++)
        {  
          if(q->distances[alpha] >= max)
                    {
                      max = q->distances[alpha];
                    }
        }
        q->max_distance = max;
        continue;   //loop over
    }
    q->neighbours.push_back(n->v[i]);   //Put the neighbour, add its distance, add the num of neighbours 
    q->distances.push_back(dist);  //and assign the min distance
    q->points_attatched++;
    if(dist < q->min_distance)
      q->min_distance = dist;   //Initial condition satisfied as min_distance is initialized to FLT_MAX
    
    float max = q->distances[0];
    for(int alpha=0; alpha < int(q->distances.size()) ; alpha++)
    {  
          if(q->distances[alpha] >= max)
                   { 
                    max = q->distances[alpha]; 
                   }
    }
    q->max_distance = max;
  }
  return;
}

int IndexToBeRemoved(float dist, QUERY *query){
  int ind = -1;
  float new_dist = dist;
  for(int i=0; i<k ; i++){ //If there is at least one or more point available, check if
    if(query->distances[i] >= new_dist)                 //the given distance is less than all the distances previously calculated.
    {  
      ind = i;
      new_dist = query->distances[i];
    }
  }

  return ind;      //Exhausted all the neighbours, cant add more.
}





float findMedian(vector< float *> vec, int dimension){    
int num_data_points = vec.size();
int vecsize = vec.size();
if(num_data_points > 5000)
  num_data_points = 5000;

float arr[num_data_points];



for(int i=0; i<num_data_points; i++)
  arr[i] = *(vec[rand() % vecsize] + dimension) ;
//arr[num_data_points] = 0;   //just being defensive

sort(&arr[0], &arr[num_data_points]);
float median = num_data_points % 2 ? arr[num_data_points / 2] : arr[num_data_points / 2 - 1] ;

return median;
}







void insert(Node *node,  vector< float* > &v)
{

if(v.size() <= k*100){                             //Terminating condition
        node->left = NULL;
        node->right = NULL;
        node->v = v;
        node->is_leaf = 1;
        node->median = -1;
        return;
}

float median = findMedian(v, node->dimension);

node->median = median;

vector<  float* > right_child;
vector<  float* > left_child;           // declare two sub arrays

for(int i=0; i<v.size(); i++){
    if(*(v[i]+ node->dimension) <= median )
        left_child.push_back(v[i]);
  
     else
      right_child.push_back(v[i]);
  }

if(left_child.size() == 0) node->left = NULL;   //Left side is empty (Unbalanced)
else {
  if(node->left == NULL ){
        node->v = v;
        node->median = -1;
        node->is_leaf = 1;
        return;
  }
  insert(node->left, left_child);
}

if(right_child.size() == 0) node->right = NULL;   //Right side is empty (Unbalanced)
else {
  if(node->right == NULL ){
        node->v = v;
        node->median = -1;
        node->is_leaf = 1;
        return;
  }
  insert(node->right, right_child); //Assign right child
}

return;
}


void * insertThread(void *input){

  param *pOutside = (param *)input;    
  pOutside->count = pOutside->count - 1;    // One more thread used up

 // cout << "Thread in dimension :" << pOutside->node->dimension << endl;

  if(pOutside->count > 0){    //Create another thread
//    cout << "Thread : id :" << pthread_self() << endl;
    float median = findMedian( pOutside->v, pOutside->node->dimension);

    vector<  float* > right;
    vector<  float* > left;           // declare two sub arrays

    for (int i=0; i < pOutside->v.size(); i++){
      if(*(pOutside->v[i] + pOutside->node->dimension) <= median )
        left.push_back(pOutside->v[i]);
  
      else
        right.push_back(pOutside->v[i]);
    }

    param *pInside = new param();
    pInside->node = pOutside->node->right;
    pInside->v = right;
    pInside->count = pOutside->count;

    void *status;
    pthread_t newTid;
    pthread_create(&newTid, NULL, insertThread, (void *)pInside);

    insert(pOutside->node->left, left);


    if(pthread_join(newTid, NULL)) {
    fprintf(stderr, "Error joining thread\n");
    return NULL;
    }

    delete pInside;

    return NULL;
  }

    insert( pOutside->node ,pOutside->v );
    return NULL;
}




