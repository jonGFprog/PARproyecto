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
double dot_product(float* a, float* b, int size) {
    double result = 0;
    for (int i = 0; i < size; i++) {
        result += a[i] * b[i];
    }
    return result;
}

// Bi bektoreen arteko norma (magnitudea) kalkulatzeko funtzioa
// Función para calcular la norma (magnitud) de un vector
float magnitude(float* vec, int size) {
    float sum = 0;
    for (int i = 0; i < size; i++) {
        sum += vec[i] * vec[i];
    }
    return sqrt(sum);
}

// Bi bektoreen arteko kosinu antzekotasuna kalkulatzeko funtzioa
// Función para calcular la similitud coseno entre dos vectores
float cosine_similarity(float* vec1, float* vec2, int size, float mag1) {
    float mag2;

    mag2 = magnitude(vec2, size);
    return dot_product(vec1, vec2, size) / (mag1 * mag2);
}

// Analogia kalkulatzeko funtzioa
// Función para calcular la analogía
void perform_analogy(float *words, int idx1, int idx2, int idx3, float *result_vector) {
  /*****************************************************************
       result_vector = word1_vector - word2_vector + word3_vector
       OSATZEKO - PARA COMPLETAR
  *****************************************************************/
  for(int i=0;i<EMB_SIZE;i++){
    result_vector[i]=words[idx1*EMB_SIZE+i]-words[idx2*EMB_SIZE+i]+words[idx3*EMB_SIZE+i];
  }
 } 

// Lortutako bektorearen gertukoen hitza bilatzeko funtzioa 
// Función para encontrar la palabra más cercana al vector resultante
void find_closest_word(float *result_vector, float *words, int numwords, int idx1, int idx2, int idx3, int *closest_word_idx, float *max_similarity) {
  /*******************************************************
        OSATZEKO - PARA COMPLETAR
        find closest word using cosine_similarity function
  ********************************************************
  float resArriba,resAbajo,resAbajoA, sim;
  resAbajoA=0;
  for(int j=0;j<EMB_SIZE;j++){
    resAbajoA+= pow(result_vector[j],2);
  }
  resAbajoA= sqrt(resAbajoA);
  *max_similarity=0;
  for(int i=0;i<numwords;i++){
    if(i==idx1||i==idx2||i==idx3){
      continue;
    }
    resArriba=0;
    resAbajo=0;
    for(int j=0;j<EMB_SIZE;j++){
      resAbajo+= pow(words[i*EMB_SIZE+j],2);
    }
    resAbajo= sqrt(resAbajo) * resAbajoA;
    for(int j=0;j<EMB_SIZE;j++){
      resArriba+= result_vector[j]*words[i*EMB_SIZE+j];
    }
    sim=resArriba/resAbajo;
    if(sim>*max_similarity){
      *max_similarity=sim;
      *closest_word_idx=i;
    }
  }*/
  float sim;
  float resAbajoA=magnitude(result_vector, EMB_SIZE);
  *max_similarity=0;
  for(int i=0;i<numwords;i++){
    if(i==idx1||i==idx2||i==idx3){
      continue;
    }
    
    sim=cosine_similarity(result_vector,words+i*EMB_SIZE,EMB_SIZE,resAbajoA);
    if(sim>*max_similarity){
      *max_similarity=sim;
      *closest_word_idx=i;
    }
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
  if (argc == 4) numwords = atoi (argv[3]);   // 3. parametroa = prozesatu behar diren hitzen kopurua  
 printf ("numwords = %d\n", numwords);
  
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
  idx1=word2ind(target_word1,dictionary,numwords);
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
   perform_analogy(words,idx1,idx2,idx3,result_vector); 
   find_closest_word(result_vector,words,numwords,idx1,idx2,idx3,&closest_word_idx,&max_similarity);
  clock_gettime (CLOCK_REALTIME, &t1);   
   
    if (closest_word_idx != -1) {
        printf("\nClosest_word: %s (%d), sim = %f \n", dictionary[closest_word_idx],closest_word_idx, max_similarity);
    } else printf("No close word found.\n");
 
  
  tej = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / (double)1e9;
  printf("\n Tej. (serie) = %1.3f ms\n\n", tej*1000);

  fclose (f1);
  fclose (f2);
  
  free(words);
  free(sim_cosine);
  free(result_vector);
  for (i=0; i<numwords;i++) free (dictionary[i]);
  free(dictionary); 

  return 0;
}

