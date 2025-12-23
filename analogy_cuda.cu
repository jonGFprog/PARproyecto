  // KONPILATZEKO - PARA COMPILAR: (C: -lm) (CUDA: -arch=sm_61)
  // EXEC: analogy embeddings.dat dictionary.dat 
  // Ej., king – man + woman = queen

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define VOCAB_SIZE 10000     // Hitz kopuru maximoa -- Maximo num. de palabras
#define EMB_SIZE 50  	     // Embedding-en kopurua hitzeko -- Nº de embedding-s por palabra
#define TAM 25		     // Hiztegiko hitzen tamaina maximoa -- Tamaño maximo del diccionario


// Hitz baten indizea kalkulatzeko funtzioa
// Función para calcular el indice de una palabra 
int word2ind(char* word, char** dictionary, int numwords) {
    for (int i = 0; i < numwords; i++) {
        if (strcmp(word, dictionary[i]) == 0) {
            return i;
        }
    }
    return -1;  // if the word is not found
}

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

// Analogia kalkulatzeko funtzioa
// Función para calcular la analogía
__global__ void perform_analogy(float *words, int idx1, int idx2, int idx3, float *result_vector) { //da igual block_size y block_amount
  /*****************************************************************
       result_vector = word1_vector - word2_vector + word3_vector
       OSATZEKO - PARA COMPLETAR
  *****************************************************************/
  int idx, stride;

  idx=threadIdx.x+blockIdx.x*blockDim.x;
  stride=gridDim.x*blockDim.x;
  for(int i=idx;i<EMB_SIZE;i+=stride){
      result_vector[i]=words[idx1*EMB_SIZE+i]-words[idx2*EMB_SIZE+i]+words[idx3*EMB_SIZE+i];
  }
  
 } 
__global__ void find_closest_word(float *result_vector, float *words, int numwords, int idx1, int idx2, int idx3, float *out_sim,int *out_pos){
  //block_size*block_amount debe ser al menos numwords
  int tid, idx, stride, stride2;
  tid=threadIdx.x;
  idx=threadIdx.x+blockIdx.x*blockDim.x;
  stride=gridDim.x*blockDim.x;

  extern __shared__ char shared_mem[];
  int *pos = (int*)shared_mem;
  float *sim = (float*)&pos[blockDim.x];
  float max_sim_local = -1000.0f;
  int max_pos_local = -1;

  /*__shared__ float local_result[EMB_SIZE];
  if (tid < EMB_SIZE) {
    local_result[tid] = result_vector[tid];
  }
  __syncthreads();*/
  for(int i = idx; i < numwords; i += stride){
    if(i == idx1 || i == idx2 || i == idx3) continue;
    float sim_val = cosine_similarity(result_vector, words+i*EMB_SIZE, EMB_SIZE);
    if(sim_val > max_sim_local){
      max_sim_local = sim_val;
      max_pos_local = i;
    }
  }

  pos[tid] = max_pos_local;
  sim[tid] = max_sim_local;
  __syncthreads ();
  unsigned int active_threads = blockDim.x;

  while (active_threads > 1) {
    unsigned int half = (active_threads + 1) >> 1;

    if (tid < half && (tid + half) < blockDim.x) {
      if (sim[tid] < sim[tid + half]) {
        sim[tid] = sim[tid + half];
        pos[tid] = pos[tid + half];
      }
    }
    __syncthreads();
    active_threads = half;
  }
  if(tid==0){
    out_sim[blockIdx.x]= sim[0];
    out_pos[blockIdx.x]= pos[0];
  }
}


int main(int argc, char *argv[]) 
{
    int		i, j, numwords, idx1, idx2, idx3;
    int 	closest_word_idx;
    float	max_similarity;
    float 	*words;
    FILE    	*f1, *f2;
    char 	**dictionary;  
    char	target_word1[TAM], target_word2[TAM], target_word3[TAM];
    float	*result_vector;
    float	*sim_cosine;
        
    struct timespec  t0, t1;
    double tej;

    int block_size=128;
    int block_amount;
    int *d_out_pos;
    int *out_pos;
    float *out_sim;
    float *d_words, *d_result_vector,*d_out_sim;

   if (argc < 3) {
     printf("Deia: analogia embedding_fitx hiztegi_fitx\n");
     exit (-1);;
   }  
   
   
  // Irakurri datuak sarrea-fitxategietatik
  // ====================================== 
  f1 = fopen (argv[1], "r");
  if (f1 == NULL) {
    printf ("Errorea %s fitxategia irekitzean\n", argv[1]);
    exit (-1);
  }

  f2 = fopen (argv[2], "r");
  if (f2 == NULL) {
    printf ("Errorea %s fitxategia irekitzean\n", argv[2]);
    exit (-1);
  }
  
 
  fscanf (f1, "%d", &numwords);	       // prozesatu behar den hitz kopurua fitxategitik jaso
  if (argc >= 4) numwords = atoi (argv[3]);   // 3. parametroa = prozesatu behar diren hitzen kopurua
  printf ("numwords = %d\n", numwords);
  if (argc >=5) block_size=atoi(argv[4]);
  printf("block_size = %d\n", block_size);
  block_amount = (numwords+block_size-1)/block_size;
  printf("block_amount = %d\n",block_amount);
  out_pos=(int*)malloc(block_amount*sizeof(int));
  out_sim=(float*)malloc(block_amount*sizeof(float));
  words = (float*)malloc (numwords*EMB_SIZE*sizeof(float));
  dictionary = (char**)malloc (numwords*sizeof(char*));
  for (i=0; i<numwords;i++){
   dictionary[i] = (char*)malloc(TAM*sizeof(char));
  }
  sim_cosine = (float*)malloc (numwords*sizeof(float));
  result_vector = (float*)malloc (EMB_SIZE*sizeof(float));
  
  for (i=0; i<numwords; i++) {
   fscanf (f2, "%s", dictionary[i]);
   for (j=0; j<EMB_SIZE; j++) {
    fscanf (f1, "%f", &(words[i*EMB_SIZE+j]));
   }
  }
  printf("Sartu analogoak diren bi hitzak eta analogia bilatu nahi diozun hitza: \n");
  printf("Introduce las dos palabras analogas y la palabra a la que le quieres buscar la analogia: \n");
   scanf ("%s %s %s",target_word1, target_word2, target_word3);

  /*********************************************************************
    OSATZEKO - PARA COMPLETAR
    Sartutako hitzen indizeak kalkulatu (idx1, idx2 & idx3) word2ind funtzioa erabilita
    Calcular los indices de las palabras introducidas (idx1, idx2 & idx3) con la funcion word2ind     
  **********************************************************************/
  idx1=word2ind(target_word1,dictionary,numwords);//si eso mas tarde paralelizar esto
  idx2=word2ind(target_word2,dictionary,numwords);
  idx3=word2ind(target_word3,dictionary,numwords);

  if (idx1 == -1 || idx2 == -1 || idx3 == -1) {
     printf("Errorea: Ez daude hitz guztiak hiztegian / No se encontraron todas las palabras en el vocabulario.\n");
     return -1;
  }
   
  clock_gettime (CLOCK_REALTIME, &t0);
    /***************************************************/
    //    OSATZEKO - PARA COMPLETAR
    //     1. call perform_analogy function
    //     2. call find_closest_word function   
   /***************************************************/ 
   cudaMalloc(&d_words,numwords*EMB_SIZE*sizeof(float));
   cudaMalloc(&d_result_vector,EMB_SIZE*sizeof(float));
   //cudaMalloc(&d_max_similarity,sizeof(float));
   cudaMalloc(&d_out_sim,block_amount*sizeof(float));

   //cudaMalloc(&d_closest_word_idx,sizeof(int));
   cudaMalloc(&d_out_pos,block_amount*sizeof(int));

   cudaMemcpy(d_words,words,numwords*EMB_SIZE*sizeof(float), cudaMemcpyHostToDevice);

   perform_analogy <<< (EMB_SIZE+block_size-1)/block_size,block_size >>> (d_words, idx1, idx2, idx3, d_result_vector);
   cudaDeviceSynchronize();

   //cudaMemcpy(result_vector,d_result_vector,EMB_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
   //for(int i=0;i<EMB_SIZE;i++) printf("result_vector[%d]= %f\n",i,result_vector[i]);
   find_closest_word <<< block_amount,block_size,block_size*(sizeof(float)+sizeof(int)) >>> (d_result_vector, d_words, numwords, idx1,idx2,idx3,d_out_sim,d_out_pos);
  //(float *result_vector, float *words, int numwords, int idx1, int idx2, int idx3, int *closest_word_idx, float *max_similarity,float *out_sim,int *out_pos)

   cudaMemcpy(out_sim,d_out_sim,block_amount*sizeof(float), cudaMemcpyDeviceToHost);
   cudaMemcpy(out_pos,d_out_pos,block_amount*sizeof(int), cudaMemcpyDeviceToHost);
   max_similarity=out_sim[0];
   closest_word_idx=out_pos[0];
   printf("i=0 similarity=%f, pos=%d\n",out_sim[0],out_pos[0]);
   for(int i=1;i<block_amount;i++){
     if(out_sim[i]>max_similarity){
      max_similarity=out_sim[i];
      closest_word_idx=out_pos[i];
    }
    printf("i=%d similarity=%f, pos=%d\n",i,out_sim[i],out_pos[i]);
   }
   clock_gettime (CLOCK_REALTIME, &t1);
   
    if (closest_word_idx != -1) {
        printf("\nClosest_word: %s (%d), sim = %f \n", dictionary[closest_word_idx],closest_word_idx, max_similarity);
    } else printf("No close word found.\n");
 
  
  tej = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / (double)1e9;
  printf("\n Tej. (serie) = %1.3f ms\n\n", tej*1000);

  fclose (f1);
  fclose (f2);
  cudaFree(d_words);
  cudaFree(d_result_vector);
  cudaFree(d_out_pos);
  cudaFree(d_out_sim);

  free(words);
  free(sim_cosine);
  free(result_vector);
  for (i=0; i<numwords;i++) free (dictionary[i]);
  free(dictionary); 
  free(out_pos);
  free(out_sim);
  return 0;
}

