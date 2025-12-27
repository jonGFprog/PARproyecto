
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


struct clusterinfo	 // clusterrei buruzko informazioa -- informacion de los clusters
{
   int  elements[VOCAB_SIZE]; 	// osagaiak -- elementos
   int  number;       		// osagai kopurua -- número de elementos
};


// Bi bektoreen arteko biderketa eskalarra kalkulatzeko funtzioa
// Función para calcular el producto escalar entre dos vectores
float dot_product(float* a, float* b, int dim) {
    float result = 0;
    for (int i = 0; i < dim; i++) {
        result += a[i] * b[i];
    }
    return result;
}

// Bi bektoreen arteko norma (magnitudea) kalkulatzeko funtzioa
// Función para calcular la norma (magnitud) de un vector
float magnitude(float* vec, int dim) {
    float sum = 0;
    for (int i = 0; i < dim; i++) {
        sum += vec[i] * vec[i];
    }
    return sqrt(sum);
}

// Bi bektoreen arteko kosinu antzekotasuna kalkulatzeko funtzioa
// Función para calcular la similitud coseno entre dos vectores
float cosine_similarity(float* vec1, float* vec2, int dim) {
    float mag1, mag2;
    
    mag1 = magnitude(vec1, dim);
    mag2 = magnitude(vec2, dim);
    if (mag1 == 0 || mag2 == 0) return 0; // Bektoreren bat 0 bada -- Si alguno de los vectores es nulo: cosine_similarity = 0
    else return dot_product(vec1, vec2, dim) / (mag1 * mag2);
}

// Distantzia euklidearra: bi hitzen kenketa ber bi, eta atera erro karratua
// Distancia euclidea: raiz cuadrada de la resta de dos palabras elevada al cuadrado
// Adi: double
double word_distance (float *word1, float *word2)
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
void update_centroids(float *words, float *centroids, int *wordcent, int numwords, int numclusters, int dim, int *cluster_sizes) {
    
    int i, j, cluster;

    for (int i = 0; i < numclusters; i++) {
        cluster_sizes[i]=0;
        for (int j = 0; j < dim; j++) {
            centroids[i*dim+j] = 0.0; // Zentroideak berrasieratu -- Reinicia los centroides
        }
    }

    for (i = 0; i < numwords; i++) {
        cluster = wordcent[i];
        cluster_sizes[cluster]++;
        for (j = 0; j < dim; j++) {
            centroids[cluster*dim+j] += words[i*dim+j];
        }
    }

    for (i = 0; i < numclusters; i++) {
        if (cluster_sizes[i] > 0) {
            for (j = 0; j < dim; j++) {
                centroids[i * dim + j] = centroids[i * dim + j] / cluster_sizes[i];
            }
        }
    }
}

// K-Means funtzio nagusia -- Función principal de K-Means
void k_means_calculate(float *words, int numwords, int dim, int numclusters, int *wordcent, float *centroids, int *changed) 
{  
/****************************************************************************************    
           OSATZEKO - PARA COMPLETAR
           - Hitz bakoitzari cluster gertukoena esleitu cosine_similarity funtzioan oinarrituta
           - Asignar cada palabra al cluster más cercano basandose en la función cosine_similarity       
****************************************************************************************/
  float cossim,cossimMax;
  int pos;
  for(int i=0;i<numwords;i++){ //en cuda que cada hilo haga una palabra
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

double cluster_homogeneity(float *words, struct clusterinfo *members, int i, int numclusters, int number)
{
    /****************************************************************************************
      OSATZEKO - PARA COMPLETAR
       Kideen arteko distantzien batezbestekoa - Media de las distancias entre los elementos del cluster
       Cluster bakoitzean, hitz bikote guztien arteko distantziak - En cada cluster, las distancias entre todos los pares de elementos
       Adi, i-j neurtuta, ez da gero j-i neurtu behar  / Ojo, una vez calculado el par i-j no hay que calcular el j-i
    ****************************************************************************************/
    double media=0;
    for(int j=1;j<members[i].number;j++){
      for(int k=0;k<j;k++){
        media+=word_distance(words+members[i].elements[j]*EMB_SIZE,words+members[i].elements[k]*EMB_SIZE);
      }
    }
    /*
    for(int i=1;i<members->number;i++){ //en cuda repartir las i entre los hilos
      for(int j=0;j<i;j++){
        media+=word_distance(words+members->elements[i]*EMB_SIZE,words+members->elements[j]*EMB_SIZE);
      }
    }*/
    return media/members[i].number;
}

double centroid_homogeneity(float *centroids, int i, int numclusters)
{
    /****************************************************************************************
      OSATZEKO - PARA COMPLETAR
    ****************************************************************************************/
   double media=0;
    for(int j=1;j<numclusters;j++){ //en cuda repartir las i entre los hilos
      media+=word_distance(centroids+i*EMB_SIZE,centroids+j*EMB_SIZE);
    }
    return media/numclusters;
}

double validation (float *words, struct clusterinfo *members, float *centroids, int numclusters)
{
  int     i, j, k, number, word1, word2;
  float   cent_homog[numclusters];
  double  disbat, max, cvi;
  float   clust_homog[numclusters];	// multzo bakoitzeko trinkotasuna -- homogeneidad de cada cluster


  // Kalkulatu clusterren trinkotasuna -- Calcular la homogeneidad de los clusters
  // Cluster bakoitzean, hitz bikote guztien arteko distantzien batezbestekoa. Adi, i - j neurtuta, ez da gero j - i neurtu behar
  // En cada cluster las distancias entre todos los pares de palabras. Ojo, una vez calculado i - j, no hay que calcular el j - i

 for (i=0; i<numclusters; i++)
  {
    disbat = 0.0;
    number = members[i].number; 
    if (number > 1)     // min 2 members in the cluster
    {
       disbat = cluster_homogeneity(words, members, i, numclusters, number);
       clust_homog[i] = disbat/(number*(number-1)/2);	// zati bikote kopurua -- div num de parejas
    }
    else clust_homog[i] = 0;


  // Kalkulatu zentroideen trinkotasuna -- Calcular la homogeneidad de los centroides
  // clusterreko zentroidetik gainerako zentroideetarako batez besteko distantzia 
  // dist. media del centroide del cluster al resto de centroides
  
    disbat = centroid_homogeneity(centroids, i, numclusters);
    cent_homog[i] = disbat / (numclusters-1);	// 5 multzo badira, 4 distantzia batu dira -- si son 5 clusters, se han sumado 4 dist.
  }
  
  // cvi index
    /****************************************************************************************
      OSATZEKO - PARA COMPLETAR
      fmaxf: max of 2 floats --> maximoa kalkulatzeko -- para calcular el máximo
    ****************************************************************************************/
   for(i=0; i<numclusters;i++){
    cvi+=(cent_homog[i]-clust_homog[i])/fmaxf(clust_homog[i],cent_homog[i]);
   }
   cvi=cvi/numclusters;
  return (cvi);
}


int main(int argc, char *argv[]) 
{
    int		i, j, numwords, k, iter, changed, end_classif;
    int		cluster, zenb, numclusters = 20;
    double  	cvi, cvi_old, dif;
    float 	*words;
    FILE    	*f1, *f2, *f3;
    char 	**hiztegia;  
    int     	*wordcent;

    struct clusterinfo  members[NUMCLUSTERSMAX];

    struct timespec  t0, t1;
    double tej;


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
  cvi=-1;
  float *centroids = (float *)malloc(k * EMB_SIZE * sizeof(float));
  int *cluster_sizes = (int *)calloc(k, sizeof(int));
  


/******************************************************************/
  // A. kmeans kalkulatu -- Calcular kmeans
  // =========================================================
  printf("K_means\n");
  clock_gettime (CLOCK_REALTIME, &t0);
  
  while (numclusters < NUMCLUSTERSMAX && end_classif == 0)
  {
    
    initialize_centroids(words, centroids, numwords, numclusters, EMB_SIZE);
    for (iter = 0; iter < MAX_ITER; iter++) {
      changed = 0;
    /****************************************************************************************
      OSATZEKO - PARA COMPLETAR
       deitu k_means_calculate funtzioari -- llamar a la función k_means_calculate
    ****************************************************************************************/
      k_means_calculate(words,numwords,EMB_SIZE,numclusters,wordcent,centroids,&changed);
      printf("changed = %d\n",changed);
      if (changed==0) break; // Aldaketarik ez bada egon, atera -- Si no hay cambios, salir
      update_centroids(words, centroids, wordcent, numwords, numclusters, EMB_SIZE, cluster_sizes);
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
    
    /****************************************************************************************
      OSATZEKO - PARA COMPLETAR
        cvi = validation (OSATZEKO - PARA COMPLETAR);
   	if (cvi - cvi_old < DELTA) end classification;
        else  continue classification;	
    ****************************************************************************************/
    
    cvi=validation(words,members,centroids,numclusters);
    if(cvi-cvi_old < DELTA){
      end_classif=1;
    }
    else{
      numclusters+=10;
      cvi_old=cvi;
    }
  }
  //} 
    
  clock_gettime (CLOCK_REALTIME, &t1);
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
  return 0;
}

