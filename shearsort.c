/**********************************************************************
 * Parallelized shearsort
 * Usage: mpirun -np p shearsort N inputfile outputfile
 **********************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

void printMatrix(double* M, int row, int col);
void shearsort(double *data, int rows, int col, int rank, int size);
void quicksort(double *data, int left, int right, int direction);
int partition_a(double *data, int left, int right, int pivotIndex);
int partition_d(double *data, int left, int right, int pivotIndex);

int main(int argc, char *argv[]){
  int N, size, rank, workload;
  double *data, *local_data, start_time, execution_time, max_time;
  char *input_filename, *output_filename;

  if(argc != 4){ 
    printf("ERROR! Expected input: shearsort N inputfile outputfile\n");
    exit(0);
  }
  N = atoi(argv[1]); // matrix size (NxN)
  input_filename = argv[2];
  output_filename = argv[3];

  MPI_Init(&argc, &argv);               // Initialize MPI 
  MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the number of processors
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get my number

  // Check assumptions
  if(N < 1){
    if(rank == 0) printf("ERROR! N must be larger than 0\n");
    MPI_Finalize();
    exit(0);
  } else if(N%size != 0){
    if(rank == 0) printf("ERROR! N must be divisible by # of processes\n");
    MPI_Finalize();
    exit(0);
  }

  // Fill 2D array
  if(rank == 0){
    data=(double*)malloc(N*N*sizeof(double));
    
    // Open inputfile for reading
    FILE *stream_in;
    stream_in = fopen(input_filename, "r");
    if(stream_in == NULL){
      printf("Error: Unable to open file: %s\n", input_filename);
      fclose(stream_in);
      MPI_Finalize();
      exit(0);
    }

    // Read 
    for(int i = 0; i < N*N; i++){
      fscanf(stream_in, "%lf ", &data[i]); 
    }
    fclose(stream_in);

    /*for(int i = 0; i < N*N; i++){
      data[i] = drand48();
    }*/

    // Print initial list
    /*printf("Initial List:\n");
    printMatrix(data, N, N);*/
  }

  // Start timer
  start_time = MPI_Wtime(); 

  // Distribute list to processes
  workload = N/size; // number of rows per process
  local_data = (double*)malloc(N*workload*sizeof(double));
  MPI_Scatter(&data[0], N*workload, MPI_DOUBLE, &local_data[0], N*workload, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Sort local list
  shearsort(local_data, workload, N, rank, size);

  // Put all sorted local lists together
  MPI_Gather(&local_data[0], N*workload, MPI_DOUBLE, &data[0], N*workload, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Compute time
	execution_time = MPI_Wtime()-start_time; // stop timer
	MPI_Reduce(&execution_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); 

  if(rank == 0){
    // Display time result
    printf("%f\n", max_time);

    // Print sorted matrix
    /*printf("\nFinal List:\n");
    printMatrix(data, N, N);*/

    // Check results
    int OK = 1;
    double prev = data[0];
    for (int i = 0; i < N; i++) {
      if(i % 2 == 0){ // if even row -> left to right
        for(int j = 0; j < N; j++){
          if(data[i*N+j] < prev){
            OK = 0;
          }
        } 
      } else { // off row -> right to left
        for(int j = N-1; j >= 0; j--){
          if(data[i*N+j] < prev){
            OK = 0;
          }
        }
      }
    }

    if(OK){
      printf("Data sorted correctly!\n");
    } else {
      printf("Data NOT sorted correctly...\n");
    }

    // Write to output file
    FILE *stream_out;
    stream_out = fopen(output_filename, "wb"); // wb
    if(stream_out == NULL)
    {
      printf("Error: unable to open file: %s\n", output_filename);
      fclose(stream_out);
      MPI_Finalize();
      exit(0);
    }
     
    // Write output
    for(int i = 0; i < N*N; i++){
      fprintf(stream_out, "%lf\n", data[i]);
    }
    fclose(stream_out);
  }
  
  // Clean up
  if(rank == 0) {
    free(data);
  }
  free(local_data);
  MPI_Finalize();

  return 0;
}

void printMatrix(double* M, int row, int col){
  for(int i = 0; i < row; i++){
    for(int j = 0; j < col; j++){
      printf("%10f ", M[i*col+j]);
    }
    printf("\n");
  }
}

void shearsort(double *data, int rows, int col, int rank, int size){
  int d = log(col)/log(2);

  for(int l = 0; l < d+1; l++){
    if(rank%2 == 0){ // first row even
      for(int k = 0; k < rows; k += 2){ // even rows
        quicksort(&data[k*col+0], 0, col-1, 0); // ascending order
      }
      for(int k = 1; k < rows; k += 2){ // odd rows
        quicksort(&data[k*col+0], 0, col-1, 1); // descending order
      }
    } else { // first row odd
      for(int k = 0; k < rows; k += 2){ // odd rows
        quicksort(&data[k*col+0], 0, col-1, 1); // descending order
      }
      for(int k = 1; k < rows; k += 2){ // even rows
        quicksort(&data[k*col+0], 0, col-1, 0); // ascending order
      }
    }

    if(l <= d){
      // Create size # of blocks (of size rows*rows)
      double *temp = (double*)malloc(rows*col*sizeof(double));
      for(int i = 0; i < rows; i++){
        for(int block = 0; block < size; block++){
          for(int j = 0; j < rows; j++){
            temp[block*rows*rows+i*rows+j] = data[i*col+(block*rows+j)];
          }
        }
      }

      // Transpose local blocks
      for(int block = 0; block < size; block++){
        for(int i = 0; i < rows; i++){
          for(int j = 0; j < rows; j++){
            data[block*rows*rows+i*rows+j] = temp[block*rows*rows+j*rows+i];
          }
        }
      }

      // All to all communication
      MPI_Alltoall(data, rows*rows, MPI_DOUBLE, temp, rows*rows, MPI_DOUBLE, MPI_COMM_WORLD);

      // Merge 
      for(int i = 0; i < rows; i++){
        for(int block = 0; block < size; block++){
          for(int j = 0; j < rows; j++){
            data[i*col+(block*rows+j)] = temp[block*rows*rows+i*rows+j];
          }
        }
      }

      // Sort local data (now columns) in ascending order
      for(int k = 0; k < rows; k++){ // columns
        quicksort(&data[k*col+0], 0, col-1, 0);
      }

      // TRANSPOSE BACK
      // Create size # of blocks (of size rows*rows)
      for(int i = 0; i < rows; i++){
        for(int block = 0; block < size; block++){
          for(int j = 0; j < rows; j++){
            temp[block*rows*rows+i*rows+j] = data[i*col+(block*rows+j)];
          }
        }
      }

      // Transpose local blocks
      for(int block = 0; block < size; block++){
        for(int i = 0; i < rows; i++){
          for(int j = 0; j < rows; j++){
            data[block*rows*rows+i*rows+j] = temp[block*rows*rows+j*rows+i];
          }
        }
      }

      // All to all communication
      MPI_Alltoall(data, rows*rows, MPI_DOUBLE, temp, rows*rows, MPI_DOUBLE, MPI_COMM_WORLD);

      // Merge 
      for(int i = 0; i < rows; i++){
        for(int block = 0; block < size; block++){
          for(int j = 0; j < rows; j++){
            data[i*col+(block*rows+j)] = temp[block*rows*rows+i*rows+j];
          }
        }
      }
      free(temp);
    }
  }
}

void quicksort(double *data, int left, int right, int direction){
    int pivotIndex, pivotNewIndex;
    
    if(direction == 0){ // ascending order
      if (right > left){
        pivotIndex = left+(right-left)/2;
        pivotNewIndex = partition_a(data, left, right, pivotIndex);

        quicksort(data, left, pivotNewIndex - 1, 0);
        quicksort(data, pivotNewIndex + 1, right, 0);
      }
    } else { // descending order
      if (right > left){
        pivotIndex = left+(right-left)/2;
        pivotNewIndex = partition_d(data, left, right, pivotIndex);

        quicksort(data, left, pivotNewIndex - 1, 1);
        quicksort(data, pivotNewIndex + 1, right, 1);
      }
    }
}

int partition_a(double *data, int left, int right, int pivotIndex){
  double pivotValue, temp;
  int storeIndex, i;
  pivotValue = data[pivotIndex];

  temp = data[pivotIndex]; 
  data[pivotIndex] = data[right]; 
  data[right] = temp;

  storeIndex = left;
  for (i = left; i < right; i++)
    if (data[i] <= pivotValue){
      temp = data[i];
      data[i] = data[storeIndex];
      data[storeIndex] = temp;

      storeIndex = storeIndex + 1;
    }

  temp = data[storeIndex];
  data[storeIndex] = data[right]; 
  data[right] = temp;

  return storeIndex;
}

int partition_d(double *data, int left, int right, int pivotIndex){
  double pivotValue, temp;
  int storeIndex, i;
  pivotValue = data[pivotIndex];

  temp = data[pivotIndex]; 
  data[pivotIndex] = data[right]; 
  data[right] = temp;

  storeIndex = left;
  for (i = left; i < right; i++)
    if (data[i] > pivotValue){ 
      temp = data[i];
      data[i] = data[storeIndex];
      data[storeIndex] = temp;

      storeIndex = storeIndex + 1;
    }

  temp = data[storeIndex];
  data[storeIndex] = data[right]; 
  data[right] = temp;

  return storeIndex;
}