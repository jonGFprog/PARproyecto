
  // EXEKUTATZEKO: kmeans embeddings.dat dictionary.dat myclusters.dat [numwords]    // numwords: matrize txikiekin probak egiteko 

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>


#define VOCAB_SIZE  10000   	// Hitz kopuru maximoa -- Maximo num. de palabras
#define EMB_SIZE    50  	// Embedding-en kopurua hitzeko -- Nº de embedding-s por palabra
#define TAM         25		// Hiztegiko hitzen tamaina maximoa -- Tamaño maximo del diccionario
#define MAX_ITER    1000    	// konbergentzia: iterazio kopuru maximoa -- Convergencia: num maximo de iteraciones
#define K	    20 		// kluster kopurua -- numero de clusters
#define DELTA       0.5		// konbergentzia (cvi) -- convergencia (cvi)
#define NUMCLUSTERSMAX 100	// cluster kopuru maximoa -- numero máximo de clusters

__device__ double atomicAddDouble(double* address, double val) //funcion sacada de https://stackoverflow.com/questions/37566987/cuda-atomicadd-for-doubles-definition-error
{
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

__device__ double media_global=0;
struct clusterinfo	 // clusterrei buruzko informazioa -- informacion de los clusters
{
   int  elements[VOCAB_SIZE]; 	// osagaiak -- elementos
   int  number;       		// osagai kopurua -- número de elementos
};

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


// Distantzia euklidearra: bi hitzen kenketa ber bi, eta atera erro karratua
// Distancia euclidea: raiz cuadrada de la resta de dos palabras elevada al cuadrado
// Adi: double
__device__ double word_distance (float *word1, float *word2)
{
    /****************************************************************************************
      OSATZEKO - PARA COMPLETAR
    ****************************************************************************************/
  double res=0;
  for(int i=0;i<EMB_SIZE;i++){
    res+=pow(word1[i]-word2[i],2);
  }
  return sqrt(res);
}

// Zentroideen hasierako balioak ausaz -- Inicializar centroides aleatoriamente
void initialize_centroids(float *words, float *centroids, int n, int numclusters, int dim) {
    int i, j, random_index;
    for (i = 0; i < numclusters; i++) {
        random_index = rand() % n;
        for (j = 0; j < dim; j++) {
            centroids[i*dim+j] = words[random_index*dim+j];
        }
    }
}

// Zentroideak eguneratu -- Actualizar centroides
__global__ void update_centroids(float *words, float *centroids, int *wordcent, int numwords, int numclusters, int dim, int *cluster_sizes) {
    
    int i, j, cluster;
    int idx, stride;

    idx=threadIdx.x+blockIdx.x*blockDim.x;
    stride=gridDim.x*blockDim.x;
    for(i=idx;i<numclusters;i+=stride){
        cluster_sizes[i]=0;
        for (j = 0; j < dim; j++) {
            centroids[i*dim+j] = 0.0; // Zentroideak berrasieratu -- Reinicia los centroides
        }
    }
    __syncthreads();
    for (i = idx; i < numwords; i+=stride) {
        cluster = wordcent[i];
        atomicAdd(&cluster_sizes[cluster],1);
        //cluster_sizes[cluster]++;
        for (j = 0; j < dim; j++) {
            atomicAdd(&centroids[cluster*dim+j],words[i*dim+j]);
            //centroids[cluster*dim+j] += words[i*dim+j];
        }
    }
    __syncthreads();
    for (i = idx; i < numclusters; i+=stride) {
        if (cluster_sizes[i] > 0) {
            for (j = 0; j < dim; j++) {
                centroids[i * dim + j] = centroids[i * dim + j] / cluster_sizes[i];
            }
        }
    }
}

// K-Means funtzio nagusia -- Función principal de K-Means
__global__ void k_means_calculate(float *words, int numwords, int dim, int numclusters, int *wordcent, float *centroids, int *changed)
{  
/****************************************************************************************    
           OSATZEKO - PARA COMPLETAR
           - Hitz bakoitzari cluster gertukoena esleitu cosine_similarity funtzioan oinarrituta
           - Asignar cada palabra al cluster más cercano basandose en la función cosine_similarity       
****************************************************************************************/
  float cossim,cossimMax;
  int pos;
  int idx, stride;

  idx=threadIdx.x+blockIdx.x*blockDim.x;
  stride=gridDim.x*blockDim.x;
  for(int i=idx;i<numwords;i+=stride){//en cuda que cada hilo haga una palabra
    cossimMax=-100;
    for(int j=0;j<numclusters;j++){
      cossim=cosine_similarity(words+i*dim,centroids+j*dim,dim);
      if(cossim>cossimMax){
        cossimMax=cossim;
        pos=j;
      }
    }
    if(wordcent[i]!=pos){
      wordcent[i]=pos;
      *changed=1;
    }
  }
}

__device__ void cluster_homogeneity(float *words, struct clusterinfo *members, int i, int numclusters, int number,char *shared_mem)
{
    /****************************************************************************************
      OSATZEKO - PARA COMPLETAR
       Kideen arteko distantzien batezbestekoa - Media de las distancias entre los elementos del cluster
       Cluster bakoitzean, hitz bikote guztien arteko distantziak - En cada cluster, las distancias entre todos los pares de elementos
       Adi, i-j neurtuta, ez da gero j-i neurtu behar  / Ojo, una vez calculado el par i-j no hay que calcular el j-i
    ****************************************************************************************/
    int tid, idx, stride;
    tid=threadIdx.x;
    idx=threadIdx.x+blockIdx.x*blockDim.x;
    stride=gridDim.x*blockDim.x;
    double media_local=0;
    for(int j=idx;j<members[i].number;j+=stride){ //en cuda repartir las i entre los hilos + reduction
      for(int k=0;k<j;k++){
        media_local+=word_distance(words+members[i].elements[j]*EMB_SIZE,words+members[i].elements[k]*EMB_SIZE);
      }
    }
    //reduccion
    double *media = (double*)shared_mem;
    media[tid]=media_local;
    __syncthreads();
    for(int s = blockDim.x / 2; s > 0; s >>= 1) {
      if(tid < s) {
        media[tid] += media[tid + s];
      }
      __syncthreads();
    }
    if(tid==0){
      atomicAddDouble(&media_global,media[0]);
    }

}

__device__ void centroid_homogeneity(float *centroids, int i, int numclusters, char *shared_mem)
{
    /****************************************************************************************
      OSATZEKO - PARA COMPLETAR
    ****************************************************************************************/
    int tid, idx, stride;
    tid=threadIdx.x;
    idx=threadIdx.x+blockIdx.x*blockDim.x;
    stride=gridDim.x*blockDim.x;
    double media_local=0;
    for(int j=idx;j<numclusters;j+=stride){ //en cuda repartir las i entre los hilos + reduction
      media_local+=word_distance(centroids+i*EMB_SIZE,centroids+j*EMB_SIZE);
    }
    //reduccion
    double *media = (double*)shared_mem;

    media[tid]=media_local;
    __syncthreads();
    for(int s = blockDim.x / 2; s > 0; s >>= 1) {
      if(tid < s) {
        media[tid] += media[tid + s];
      }
      __syncthreads();
    }
    if(tid==0){
      atomicAddDouble(&media_global,media[0]);
    }

}

__global__ void validation (float *words, struct clusterinfo *members, float *centroids, int numclusters, float *cent_homog, float *clust_homog, double *cvi)
{
  int     i, number;
  //float   cent_homog[NUMCLUSTERSMAX];
  double  disbat;
  //float   clust_homog[NUMCLUSTERSMAX];	// multzo bakoitzeko trinkotasuna -- homogeneidad de cada cluster
  extern __shared__ char shared_mem[];

  // Kalkulatu clusterren trinkotasuna -- Calcular la homogeneidad de los clusters
  // Cluster bakoitzean, hitz bikote guztien arteko distantzien batezbestekoa. Adi, i - j neurtuta, ez da gero j - i neurtu behar
  // En cada cluster las distancias entre todos los pares de palabras. Ojo, una vez calculado i - j, no hay que calcular el j - i
  int tid, stride;
  int bid=blockIdx.x;
  tid=threadIdx.x;
  stride=gridDim.x;
  for(i=bid%numclusters;i<numclusters;i+=stride) //repartimos las i por bloques
  {
    disbat = 0.0;
    number = members[i].number; 
    if (number > 1)     // min 2 members in the cluster
    {
       cluster_homogeneity(words, members, i, numclusters, number,shared_mem);
       __syncthreads();

       if(tid==0){
          disbat = media_global/members[i].number;
          media_global=0;
          clust_homog[i] = disbat/(number*(number-1)/2);
      }
       // zati bikote kopurua -- div num de parejas
    }
    else clust_homog[i] = 0;

  __syncthreads();
  // Kalkulatu zentroideen trinkotasuna -- Calcular la homogeneidad de los centroides
  // clusterreko zentroidetik gainerako zentroideetarako batez besteko distantzia 
  // dist. media del centroide del cluster al resto de centroides
    centroid_homogeneity(centroids, i, numclusters,shared_mem);
    __syncthreads();
    if(tid==0){
        disbat = media_global/members[i].number;
        media_global=0;
        cent_homog[i] = disbat / (numclusters-1);
    }
    	// 5 multzo badira, 4 distantzia batu dira -- si son 5 clusters, se han sumado 4 dist.
    __syncthreads(); //quizas sobra
  }
  
  // cvi index
    /****************************************************************************************
      OSATZEKO - PARA COMPLETAR
      fmaxf: max of 2 floats --> maximoa kalkulatzeko -- para calcular el máximo
    ****************************************************************************************/
    //repartir hilos + reduccion


}

__global__ void calculo_cvi(int numclusters,double *cvi, float *cent_homog, float *clust_homog){
  int idx, stride;
  stride=gridDim.x*blockDim.x;
  idx=threadIdx.x+blockIdx.x*blockDim.x;
  for(int i=idx; i<numclusters;i+=stride){
    atomicAddDouble(cvi,(cent_homog[i]-clust_homog[i])/fmaxf(clust_homog[i],cent_homog[i]));
    //*cvi+=(cent_homog[i]-clust_homog[i])/fmaxf(clust_homog[i],cent_homog[i]);
  }
}
int main(int argc, char *argv[])
{
    int i, j, numwords, k, iter, *changed, end_classif;
    int cluster, zenb, numclusters = 20;
    double *cvi, cvi_old,valor_cvi;
    float *words;
    FILE *f1, *f2, *f3;
    char **hiztegia;
    int *wordcent;
    int block_amount;
    int block_size=128;
    int block_size2=128;
    struct clusterinfo  members[NUMCLUSTERSMAX];

    struct timespec  t0, t1;
    double tej;

    float *d_words, *d_centroids,*d_cent_homog,*d_clust_homog;
    //double *d_cvi;
    int *d_wordcent, *d_cluster_sizes;
    struct clusterinfo *d_members;

   if (argc < 4) {
     printf("\nCall: kmeans embeddings.dat dictionary.dat myclusters.dat [numwords]\n\n");
     printf("\t(in) embeddings.dat and dictionary.dat\n");
     printf("\t(out) myclusters.dat\n");
     printf("\t(numwords optional) prozesatu nahi den hitz kopurua -- num de palabras a procesar\n\n");
     exit (-1);;
   }  
   
  // Irakurri datuak sarrea-fitxategietatik -- Leer los datos de los ficheros de entrada
  // =================================================================================== 

  f1 = fopen (argv[1], "r");
  if (f1 == NULL) {
    printf ("Errorea %s fitxategia irekitzean -- Error abriendo fichero\n", argv[1]);
    exit (-1);
  }

  f2 = fopen (argv[2], "r");
  if (f2 == NULL) {
    printf ("Errorea %s fitxategia irekitzean -- Error abriendo fichero\n", argv[2]);
    exit (-1);
  }
  
  fscanf (f1, "%d", &numwords);	       
  if (argc == 5) numwords = atoi (argv[4]);  
  printf ("numwords = %d\n", numwords);
  if (argc == 6) block_size=atoi(argv[5]);
  printf("block_size = %d\n",block_size);
  if (argc == 7) block_size2=atoi(argv[6]);
  printf("block_size2 = %d\n",block_size2);
  words = (float*)malloc (numwords*EMB_SIZE*sizeof(float));
  hiztegia = (char**)malloc (numwords*sizeof(char*));
  for (i=0; i<numwords;i++){
   hiztegia[i] = (char*)malloc(TAM*sizeof(char));
  }
  
  for (i=0; i<numwords; i++) {
   fscanf (f2, "%s", hiztegia[i]);
   for (j=0; j<EMB_SIZE; j++) {
    fscanf (f1, "%f", &(words[i*EMB_SIZE+j]));
   }
  }
  printf ("Embeddingak eta hiztegia irakurrita -- Embeddings y dicionario leidos\n");

  wordcent = (int *)malloc(numwords * sizeof(int));
  for (int i = 0; i < numwords; i++) wordcent[i] = -1;

  k = NUMCLUSTERSMAX;   // hasierako kluster kopurua (20) -- numero de clusters inicial
  end_classif = 0; 
  cvi_old = -1;
  float *centroids = (float *)malloc(k * EMB_SIZE * sizeof(float));
  //float *centroidss = (float *)malloc(k * EMB_SIZE * sizeof(float));

  int *cluster_sizes = (int *)calloc(k, sizeof(int));
  
  cudaMalloc(&d_words,numwords*EMB_SIZE*sizeof(float));
  cudaMalloc(&d_wordcent,numwords*sizeof(int));
  cudaMalloc(&d_centroids,k*EMB_SIZE*sizeof(float));
  cudaMalloc(&d_cluster_sizes,k*sizeof(int));
  cudaMallocManaged(&changed,sizeof(int));
  cudaMalloc(&d_members,NUMCLUSTERSMAX*sizeof(clusterinfo));
  cudaMallocManaged(&cvi,sizeof(double));
  cudaMalloc(&d_cent_homog,NUMCLUSTERSMAX*sizeof(float));
  cudaMalloc(&d_clust_homog,NUMCLUSTERSMAX*sizeof(float));
  *cvi=-1;
  cudaMemcpy(d_words,words,numwords*EMB_SIZE*sizeof(float), cudaMemcpyHostToDevice);
//cudaMemcpy(d_cluster_sizes,cluster_sizes,k*sizeof(int), cudaMemcpyHostToDevice);
/******************************************************************/
  // A. kmeans kalkulatu -- Calcular kmeans
  // =========================================================
  printf("K_means\n");
  clock_gettime (CLOCK_REALTIME, &t0);
  
  while (numclusters < NUMCLUSTERSMAX && end_classif == 0)
  {
    
    initialize_centroids(words, centroids, numwords, numclusters, EMB_SIZE); //no paralelizar
    //memcpy HostToDevice
    /*printf("Inicializacion de centroides:\nnumclusters = %d\n",numclusters);
    for (int cent=0;cent<numclusters;cent++){
      printf("\ncentroide %d: ",cent);
      for(int k=0;k<EMB_SIZE;k++){
        printf("%f||",centroids[cent*EMB_SIZE+k]);
      }
    }*/
    cudaMemcpy(d_centroids,centroids,numclusters*EMB_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(centroidss,d_centroids,numclusters*EMB_SIZE*sizeof(float),cudaMemcpyDeviceToHost);
    /*for (int cent=0;cent<numclusters;cent++){
      printf("\ncentroide %d: ",cent);
      for(int k=0;k<EMB_SIZE;k++){
        printf("%f||",centroidss[cent*EMB_SIZE+k]);
      }
    }*/

    block_amount=numwords/block_size;
    for (iter = 0; iter < MAX_ITER; iter++) {
      *changed = 0;
    /****************************************************************************************
      OSATZEKO - PARA COMPLETAR
       deitu k_means_calculate funtzioari -- llamar a la función k_means_calculate
    ****************************************************************************************/
      //k_means_calculate(float *words, int numwords, int dim, int numclusters, int *wordcent, float *centroids, int *changed)

      k_means_calculate<<< block_amount,block_size >>>(d_words,numwords,EMB_SIZE,numclusters,d_wordcent,d_centroids,changed);//paralelizar
      //sync
      cudaDeviceSynchronize();
      printf("it %d  changed =%d\n",iter,*changed);
      if (*changed==0) break; // Aldaketarik ez bada egon, atera -- Si no hay cambios, salir
      //void update_centroids(float *words, float *centroids, int *wordcent, int numwords, int numclusters, int dim, int *cluster_sizes)

      update_centroids<<< block_amount,block_size >>>(d_words, d_centroids, d_wordcent, numwords, numclusters, EMB_SIZE, d_cluster_sizes);//paralelizar

    }



  // B. Sailkatzearen "kalitatea" -- "Calidad" del cluster
  // =====================================================
    printf("Kalitatea -- Calidad\n");   
    for (i=0; i<numclusters; i++)  members[i].number = 0;
    // cluster bakoitzeko hitzak (osagaiak) eta kopurua -- palabras de cada clusters y cuantas son
    for (i=0; i<numwords; i++)  {
      cluster = wordcent[i];
      zenb = members[cluster].number;
      members[cluster].elements[zenb] = i;	// clusterreko hitza -- palabra del cluster
      members[cluster].number ++; 
    }
    //memcpy members HostToDevice
    cudaMemcpy(d_members,members,numclusters*sizeof(clusterinfo), cudaMemcpyHostToDevice);
    /****************************************************************************************
      OSATZEKO - PARA COMPLETAR
        cvi = validation (OSATZEKO - PARA COMPLETAR);
   	if (cvi - cvi_old < DELTA) end classification;
        else  continue classification;	
    ****************************************************************************************/
    //void validation (float *words, struct clusterinfo *members, float *centroids, int numclusters, double *cvi)
    block_amount=numclusters*(1024/block_size2); //creo que tiene sentido ns
    validation<<< block_amount,block_size2, block_size2*sizeof(double) >>>(d_words,d_members,d_centroids,numclusters,d_cent_homog,d_clust_homog,cvi); //paralelizar
    block_amount=numclusters/block_size;
    cudaDeviceSynchronize();

    calculo_cvi <<<block_amount,block_size>>> (numclusters,cvi,d_cent_homog,d_clust_homog);

    valor_cvi = *cvi/(double)numclusters;
    if(valor_cvi-cvi_old < DELTA){
      end_classif=1;
    }
    else{
      numclusters+=10;
      cvi_old=valor_cvi;
    }

  }

  //memcpy
    
  clock_gettime (CLOCK_REALTIME, &t1);
  cudaMemcpy(cluster_sizes,d_cluster_sizes,k*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(wordcent,d_wordcent,numwords*sizeof(int), cudaMemcpyDeviceToHost);
  /******************************************************************/

  for (i=0; i<numclusters; i++){
    printf ("%d. cluster, %d words \n", i, cluster_sizes[i]);
  }

  tej = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / (double)1e9;
  printf("\n Tej. (serie) = %1.3f ms\n\n", tej*1000);

// Idatzi clusterrak fitxategietan -- Escribir los clusters en el fichero
  f3 = fopen (argv[3], "w");
  if (f3 == NULL) {
    printf ("Errorea %s fitxategia irekitzean -- Error abriendo fichero\n", argv[3]);
    exit (-1);
  }

  for (i=0; i<numwords; i++){
    fprintf (f3, "%s \t\t -> %d cluster\n", hiztegia[i], wordcent[i]);
  }
  printf ("clusters written\n");

  fclose (f1);
  fclose (f2);
  fclose (f3);

  free(words);
  for (i=0; i<numwords;i++) free (hiztegia[i]);
  free(hiztegia); 
  free(cluster_sizes);
  free(centroids);
  cudaFree(d_centroids);
  cudaFree(d_cluster_sizes);
  cudaFree(cvi);
  cudaFree(d_members);
  cudaFree(d_wordcent);
  cudaFree(d_words);
  return 0;
}

