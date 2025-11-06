// clang++ -O3  -fvectorize -fslp-vectorize nuss.cpp  -lgomp -fopenmp -march=rv64gcv -Rpass=loop-vectorize 

#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <omp.h>



#include <cstdlib>

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

using namespace std;

#define paired(a1, a2) \
(((a1) == 'A' && (a2) == 'U') || \
((a1) == 'U' && (a2) == 'A') || \
((a1) == 'G' && (a2) == 'C') || \
((a1) == 'C' && (a2) == 'G'))



int N = 10240;
const int bb = 16;

short** S;
char *seqq;



inline void s1(int i, int j, int k) {
  S[i][j] = max(S[i][k] + S[k+1][j], S[i][j]);
}

inline void s1(int i, int j, int k, short *s) {
  *s = max(S[i][k] + S[k+1][j], *s);
}

inline void s2(int i, int j) {
    S[i][j] = max(S[i][j], S[i+1][j-1] + paired(seqq[i], seqq[j]));
}

int main(int argc, char *argv[]){
    
    int num_proc=8;

    if(argc > 1)
        num_proc = atoi(argv[1]);


  string seq = "GUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUAC";

 N += bb - N % bb;
 //N = seq.length();


 int n = N, i,j,k;

  seqq = new char[N+1];
  if(N>1) // no debug
   {
    char znaki[] = {'C', 'G', 'U', 'A'};
    srand(static_cast<unsigned short>(time(0)));

    for (short i = 0; i < N; i++) {
      seqq[i] = znaki[rand() % 4];  // Losowy wybór z zestawu 'C', 'G', 'U', 'A'
    }
   }
   cout << seqq << endl;
  std::strcpy(seqq, seq.c_str());          // Copy the string content   // use random data for given big N, comment this

  short* flatArray_S = new short[n * n];
  short* flatArray_S_CPU = new short[n * n];

  // Allocate 2D host array for CPU and GPU
  S = new short*[n];
  short** S_CPU = new short*[n];

  for(short i = 0; i < n; i++) {
    S[i] = &flatArray_S[i * n];
    S_CPU[i] = &flatArray_S_CPU[i * n];
  }
  // initialization
  for(i=0; i<N; i++) {
    for(j=0; j<N; j++){
      S[i][j] = 0;
      S_CPU[i][j] = 0;
    }
  }
  for(i=0; i<N; i++){
    S[i][i] = 0;
    S_CPU[i][i] = 0;
    if(i+1 < N) {
      S[i][i + 1] = 0;
      S[i+1][i] = 0;
      S_CPU[i][i+1] = 0;
      S_CPU[i+1][i] = 0;
    }
  }
  // -----------------------------


  double start_time = omp_get_wtime();
  


   for (int c0 = 0; c0 <= (N - 1)/bb; c0 += 1){  // serial loop
 //   printf("%d\n ", (N - 1)/bb - c0);
   #pragma omp parallel for shared(c0) num_threads(num_proc)
       for (int c1 = c0; c1 <= N / bb; c1 += 1)
        {
           
            short C[bb][bb] = {0};
            short A_elements[bb][bb] = {0};
            short B_elements[bb][bb] = {0};
            short _ii = c1-c0;
            short _jj = c1;

            for (short kk = _ii + 1; kk < _jj; kk++) {
                
                for(int row=0; row<bb; row++)
                    for(int col=0; col<bb; col++){
                        A_elements[row][col] = S[bb * _ii + row][bb * kk + col - 1];
                        B_elements[row][col] = S[bb * kk + row][bb * _jj + col];
                    }
                
                for (int row = 0; row < bb; row++) 
                    
                    //    #pragma clang loop vectorize(enable)
                    for (int col = 0; col < bb; col++)  // simd
                    for (int k = 0; k < bb; k++) 
                           //s1(bb * _ii + row, bb * _jj + col, bb * kk + k - 1, &C[row][col]);  
                           C[row][col] = max(C[row][col], A_elements[row][k] + B_elements[k][col]);
            }
                                                                    
            if (_jj >= _ii + 1) {
              for (int c2 = bb * _ii + (bb - 1); c2 >= bb * _ii; c2--)
                for (int c3 = bb * _jj; c3 <= min(N - 1, bb * _jj + (bb - 1)); c3 += 1){
                  for (int c4 = c2; c4 < c2 + bb-1; c4 += 1)
                    s1(c2, c3, c4);
                   
                  S[c2][c3] = max(C[c2 - bb*_ii][c3 - bb*_jj],S[c2][c3]);
                   
                  for (int c4 = bb * _jj - 1; c4 < c3; c4 += 1)
                    s1(c2, c3, c4);
                  s2(c2, c3);
                }
            } else if (_jj == _ii) {
              for (int c2 = min(N - 2, bb * _jj + (bb - 2)); c2 >=  bb * _jj; c2--)
                for (int c3 = c2 + 1; c3 <= min(N - 1, bb * _jj + (bb - 1)); c3 += 1){
                  for (int c4 = c2; c4 < c3; c4 += 1)
                    s1(c2, c3, c4);
                  s2(c2, c3);
                }
            }                                  
        }
   }


    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;
    printf("Time taken: %f seconds\n", elapsed_time);

    printf("cpu ended\n");

exit(0);

    for (i = N-1; i >= 0; i--) {
        for (j = i+1; j < N; j++) {
            for (k = i; k < j; k++) {
                S_CPU[i][j] = max(S_CPU[i][k] + S_CPU[k+1][j], S_CPU[i][j]);
            }

            S_CPU[i][j] = max(S_CPU[i][j], S_CPU[i+1][j-1] + paired(seqq[i],seqq[j]));

        }
    }

    for(i=0; i<N; i++)
        for(j=0; j<N; j++)
            if(S[i][j] != S_CPU[i][j]){
                cout << i <<" " <<  j << ":" << S[i][j] << " " << S_CPU[i][j] << endl;
                cout << "error" << endl;
               exit(1);

            }
   if(1==0)
    for(i=0; i<N; i++){
        for(j=0; j<N; j++){
            if(S[i][j] < 0)
                cout << "";
            else
                cout << S[i][j];
            cout << "\t";
        }
        cout << "\n";
    }
    cout << endl;
if(1==0)
    for(i=0; i<N; i++){
        for(j=0; j<N; j++){
            if(S[i][j] < 0)
                cout << "";
            else
                cout << S_CPU[i][j];
            cout << "\t";
        }
        cout << "\n";
    }
    cout << endl;
    delete[] S;
    delete[] S_CPU;


  return 0;
 }
 
 
 /*
 S :=  [N] -> { [ii,jj] -> [jj-ii,jj] : 0 <= ii <= N/32 and 0 <= jj <= N/32 and jj >= ii};
codegen S;
for (int c0 = 0; c0 <= floord(N, 32); c0 += 1)
  for (int c1 = c0; c1 <= N / 32; c1 += 1)
    (-c0 + c1, c1);

 //scattering
 S1 :=  [N,ii,jj] -> { [i,j,k] -> [i,j,0,k] : i <= k < j and  
0 <= i <= N and 0 <= j <=N and ii*32 <= i < (ii+1)*32 and jj*32 <= j < (jj+1)*32 and ((ii=jj and j>i) or jj>ii)};
S2 :=  [N,ii,jj] -> { [i,j] -> [i,j,1,0] :  
0 <= i <= N and 0 <= j <=N and ii*32 <= i < (ii+1)*32 and jj*32 <= j < (jj+1)*32 and ((ii=jj and j>i) or jj>ii)};
S := S1 + S2; codegen S;
odwroc kolejnosc petli i, s2 na dekrementacje

*/
