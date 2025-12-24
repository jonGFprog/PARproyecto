
  // EXEC: knn embeddins.dat similarities.dat [numwords]   

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>


#define VOCAB_SIZE 10000     // Hitz kopuru maximoa -- Maximo num. de palabras
#define EMB_SIZE 50  	     // Embedding-en kopurua hitzeko -- Nº de embedding-s por palabra


// Bi bektoreen arteko biderketa eskalarra kalkulatzeko funtzioa
// Función para calcular el producto escalar entre dos vectores
__device__ double dot_product(float* a, float* b, int size) {
  double result = 0;
  for (int i = 0; i < size; i++) {
    result += a[i] * b[i];
  }
  return result;
}

// Bi bektoreen arteko norma (magnitudea) kalkulatzeko funtzioa
// Función para calcular la norma (magnitud) de un vector
__device__ float magnitude(float* vec, int size) {
  float sum = 0;
  for (int i = 0; i < size; i++) {
    sum += vec[i] * vec[i];
  }
  return sqrtf(sum);
}

// Bi bektoreen arteko kosinu antzekotasuna kalkulatzeko funtzioa
// Función para calcular la similitud coseno entre dos vectores
__device__ float cosine_similarity(float* vec1, float* vec2, int size) {
  float mag1,mag2;
  mag1 = magnitude(vec1, size);
  mag2 = magnitude(vec2, size);
  float eps = 1e-8f;  // Evitar div por 0
  return dot_product(vec1, vec2, size) / (mag1 * mag2 + eps);
}


// kNN hitz guztietarako -- kNN para todas las palabras
__global__ void knn_complet(float *words, int numwords, float *similarities) {
/******************************************************************
    // Hitz bakoitzak beste guztien duen antzekotasuna kalkulatu 
    // Calcula la similitud de cada palabra con todas las demas

    //    OSATZEKO - PARA COMPLETAR
******************************************************************/
  int i,j, total_iteraciones;
  int idx, stride;

  idx=threadIdx.x+blockIdx.x*blockDim.x;
  stride=gridDim.x*blockDim.x;
  total_iteraciones=(numwords*(numwords-1))/2;
  for(int it=idx;it<total_iteraciones;it+=stride){
    i=(sqrtf(1+8*it)-1)/2 + 1;
    j=it - (i*(i-1))/2;
    similarities[i*numwords+j]=cosine_similarity(words+i*EMB_SIZE,words+j*EMB_SIZE,EMB_SIZE);
    similarities[j*numwords+i]=similarities[i*numwords+j];
  }
}



int main(int argc, char *argv[]) 
{
    int		i, j, numwords;//, row, col;
    float 	*words;
    FILE    	*f1, *f2;
    float 	*similarities;
    float *d_similarities, *d_words;
    struct timespec  t0, t1;
    double tej;
    int block_size=128;
    int block_amount=128;

   if (argc < 3) {
     printf("Deia-Llamada: knn embeddins.dat similarities.dat [numwords]\n");
     printf("\t(in) embeddings.dat\n");
     printf("\t(out) similarities.dat\n");
     printf("\t(hautazkoa) prozesatu nahi den hitz kopurua -- (opcional) num de palabras que se quieren procesar\n");
     exit (-1);;
   }  
   
  // Irakurri datuak sarrea-fitxategietatik
  // ====================================== 

  f1 = fopen (argv[1], "r");
  if (f1 == NULL) {
    printf ("Errorea %s fitxategia irekitzean -- Error abriendo fichero\n", argv[1]);
    exit (-1);
  }

  fscanf (f1, "%d", &numwords);	       
  if (argc >= 4) numwords = atoi (argv[3]);
  printf ("numwords = %d\n", numwords);
  if (argc >= 5) block_size= atoi(argv[4]);
  printf("block_size = %d\n", block_size);
  if(argc >= 6) block_amount=atoi(argv[5]);
  printf("block_amount = %d\n",block_amount);
/******************************************************************
    // Memoria dinamikoki esleitu words eta similarities datu-egiturei
    // Asignar memoria dinámica a las estructuras de datos words y similarities

    //    OSATZEKO - PARA COMPLETAR
******************************************************************/
  words=(float*)malloc(numwords*EMB_SIZE*sizeof(float));
  similarities=(float*)malloc(numwords*numwords*sizeof(float));
  
  for (i=0; i<numwords; i++) {
   for (j=0; j<EMB_SIZE; j++) {
    fscanf (f1, "%f", &(words[i*EMB_SIZE+j]));
   }
  }
  printf ("Embeddingak irakurrita\n");
  
  printf("Hitz guztien auzokideak kalkulatzera zoaz\n");
  clock_gettime (CLOCK_REALTIME, &t0);
/******************************************************************
    // Deitu funtzioari
    // Llamar a la función
    
    //    OSATZEKO - PARA COMPLETAR
******************************************************************/

  cudaMalloc(&d_words,numwords*EMB_SIZE*sizeof(float));
  cudaMalloc(&d_similarities,numwords*numwords*sizeof(float));

  cudaMemcpy(d_words,words,numwords*EMB_SIZE*sizeof(float), cudaMemcpyHostToDevice);

  knn_complet <<< block_amount, block_size>>> (d_words,numwords,d_similarities);

  cudaMemcpy(similarities,d_similarities,numwords*numwords*sizeof(float), cudaMemcpyDeviceToHost);

  clock_gettime (CLOCK_REALTIME, &t1);
   
  tej = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / (double)1e9;
  printf("\n Tej. (serie) = %1.3f ms\n\n", tej*1000);

// Idatzi antzekotasunak similarities fitxategietan -- Escribe las similitudes en el fichero similarities
  f2 = fopen (argv[2], "w");
  if (f2 == NULL) {
    printf ("Errorea %s fitxategia irekitzean -- Error abriendo fichero\n", argv[2]);
    exit (-1);
  }

  fprintf (f2, "%d\n", numwords);
  for (i=0; i<numwords; i++) {
    for (j=0; j<numwords; j++) {
     fprintf (f2, "%f\t", similarities[i*numwords+j]);
    } 
    fprintf  (f2, "\n"); 
  }
  printf ("antzekotasunak idatzita fitxategietan -- similitudes escritas en el fichero\n");

  fclose (f1);
  fclose (f2);

  free(words);
  free(similarities); 

  cudaFree(d_words);
  cudaFree(d_similarities);
  return 0;
}

