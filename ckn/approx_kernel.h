#ifndef APPROX_KERNEL_H
#define APPROX_KERNEL_H
#include "common.h"

#ifdef HAVE_MKL
#define SET_MKL_ST \
   const int num_threads= mkl_get_max_threads(); \
   mkl_set_num_threads(1); \
   mkl_set_dynamic(false);

#define SET_MKL_MT \
   mkl_set_num_threads(num_threads);
#else
#define SET_MKL_ST 
#define SET_MKL_MT 
#endif

template <typename T>
void fast_kmeans(const Matrix<T>& X, Matrix<T>& Z, Vector<int>& idx, const int num_iter = 10, const bool compute_assignments=false) {
   const int n=X.n();
   const int p=Z.n();
   if (num_iter >=0) {
      Vector<int> per;
      per.randperm(n);
      Vector<T> col1, col2;
      for (int ii=0; ii<p; ++ii) {
         X.refCol(per[ii],col1);
         Z.refCol(ii,col2);
         col2.copy(col1);
      };
   } else {
      Z.setAleat();
      Z.normalize();
   }

   // TODO: make the computation of final cluster assignments optional
   for (int ii=0; ii<num_iter+(compute_assignments ? 1 : 0); ++ii) {
      printf("K-means epoch %d\n",ii+1);
      const int size_block=p;
      Vector<T> tmp(n);
      idx.resize(n);
      if (ii==num_iter)
        cout << "Computing final cluster assignments" << endl;
      SET_MKL_ST
#pragma omp parallel for
      for (int jj =0; jj<n; jj+=size_block) {
         const int endblock=MIN(jj+size_block,n);
         const int length_block=endblock-jj;
         Matrix<T> subX, ZtX;
         X.refSubMat(jj,length_block,subX);
         Z.mult(subX,ZtX,true); // compute all pairwise inner products
         Vector<T> col;
         for (int kk=0; kk<length_block; ++kk) {
            ZtX.refCol(kk,col);
            idx[jj+kk]=col.max(); // best matching centroid for this sample
            tmp[jj+kk]=col[idx[jj+kk]]; // tmp holds best matching quality for each sample. used later to revive dead clusters
         }
      }
      SET_MKL_MT
      if (ii==num_iter) //no update after the final cluster assignment step
        return;

      Vector<int> numElem(p);
      numElem.setZeros();
      Z.setZeros();
      Vector<T> col1, col2;
      for (int jj =0; jj<n; ++jj) {
         const int ind=idx[jj];
         numElem[ind]++;
         X.refCol(jj,col1);
         Z.refCol(ind,col2);
         col2.add(col1);
      }
      for (int jj =0; jj<p; ++jj) {
         Z.refCol(jj,col1);
         if (numElem[jj]) {
            col1.normalize();
         } else {
            const int ind=tmp.min();
            tmp[ind]=1;
            X.refCol(ind,col2);
            col1.copy(col2);
         }
      }
   } // end k-means iteration
};

template <typename T>
void compute_kZtZ(const Matrix<T>& Z, Matrix<T>& ZtZ, const T sigma2,const T lambda2, const int type_kernel = 0) {
   Z.XtX(ZtZ);
   switch (type_kernel) {
      case 0: ZtZ.scal(T(1.0)/sigma2); 
              ZtZ.add(-T(1.0)/sigma2);
              ZtZ.exp();
              break;
      case 1: ZtZ.add(T(1.0));
              ZtZ.scal(T(0.5)); 
              ZtZ.pow(T(2.0)); 
              break;
      case 2: ZtZ.add(T(1.0));
              ZtZ.scal(T(0.5)); 
              ZtZ.pow(T(3.0)); 
              break;
      case 4: ZtZ.scal(T(1.0)/sigma2); 
              ZtZ.add(-T(1.0)/sigma2);
              ZtZ.exp();
              ZtZ.add(-exp(-T(1.0)/sigma2));
              break;
      case 5: ZtZ.scal(T(1.0)/sigma2); 
              ZtZ.add(-T(1.0)/sigma2);
              ZtZ.exp();
              ZtZ.add(-exp(-T(2.0)/sigma2));
              break;
   }
   ZtZ.addDiag(lambda2);
   //ZtZ.scal(T(1.0)/(T(1.0)+lambda2));
};

template <typename T>
void compute_kZtX(const Matrix<T>& Z, const Matrix<T>& X, Matrix<T>& ZtX, const T sigma2, const int type_kernel = 0) {
   switch (type_kernel) {
      case 0: Z.mult(X,ZtX,true,false,T(1.0)/sigma2);
              ZtX.add(-T(1.0)/sigma2);
              ZtX.exp();
              break;
      case 1:
              Z.mult(X,ZtX,true,false,T(1.0));
              ZtX.add(T(1.0));
              ZtX.scal(T(0.5)); 
              ZtX.pow(T(2.0));
              break;
      case 2:
              Z.mult(X,ZtX,true,false,T(1.0));
              ZtX.add(T(1.0));
              ZtX.scal(T(0.5)); 
              ZtX.pow(T(3.0));
              break;
      case 4: Z.mult(X,ZtX,true,false,T(1.0)/sigma2);
              ZtX.add(-T(1.0)/sigma2);
              ZtX.exp();
              ZtX.add(-exp(-T(1.0)/sigma2));
              break;
      case 5: Z.mult(X,ZtX,true,false,T(1.0)/sigma2);
              ZtX.add(-T(1.0)/sigma2);
              ZtX.exp();
              ZtX.add(-exp(-T(2.0)/sigma2));
              break;
   }
};

template <typename T>
void compute_kZtXb(const Matrix<T>& ZtX, Matrix<T>& kZtX, const T sigma2, const int type_kernel = 0) {
   kZtX.copy(ZtX);
   switch (type_kernel) {
      case 0: kZtX.scal(T(1.0)/sigma2);
              kZtX.add(-T(1.0)/sigma2);
              kZtX.exp();
              break;
      case 1: kZtX.add(T(1.0));
              kZtX.scal(T(0.5)); 
              kZtX.pow(T(2.0));
              break;
      case 2: kZtX.add(T(1.0));
              kZtX.scal(T(0.5)); 
              kZtX.pow(T(3.0));
              break;
      case 4: kZtX.scal(T(1.0)/sigma2);
              kZtX.add(-T(1.0)/sigma2);
              kZtX.exp();
              kZtX.add(-exp(-T(1.0)/sigma2));
              break;
      case 5: kZtX.scal(T(1.0)/sigma2);
              kZtX.add(-T(1.0)/sigma2);
              kZtX.exp();
              kZtX.add(-exp(-T(2.0)/sigma2));
              break;
   }
};



template <typename T>
T eval_obj(const Matrix<T>& X, const Matrix<T>& W, const T sigma2, const T lambda2) {
   const int n=X.n();
   const int p=W.m();
   Vector<T> obj(n);

   Matrix<T> kZtZ;
   compute_kZtZ(W,kZtZ,sigma2,lambda2);
   Matrix<T> InvkZtZ;
   InvkZtZ.copy(kZtZ);
   InvkZtZ.invSymPos();

   SET_MKL_ST
   const int size_block=p;
#pragma omp parallel for
   for (int jj =0; jj<n; jj+=size_block) {
      const int endblock=MIN(jj+size_block,n);
      const int length_block=endblock-jj;
      Matrix<T> subX, kZtX, alpha;
      X.refSubMat(jj,length_block,subX);
      compute_kZtX(W,subX,kZtX,sigma2);
      InvkZtZ.mult(kZtX,alpha);
      kZtX.mult_elementWise(alpha,kZtX);
      Vector<T> col;
      for (int kk=0; kk<length_block; ++kk) {
         kZtX.refCol(kk,col);
         obj[kk+jj]=1-col.sum();
      }
   }
   SET_MKL_MT
   return obj.sum()/(2*n);
}

template <typename T>
void nystrom_ckn_multiprojection(
    const Matrix<T>& X, // holds the training patches. shape m x n
    Matrix<T>& subspace_centroids, // must be preallocated; shape m x num_subspaces
    const int num_filters,
    // note the * instead of &. these are preallocated arrays of results!
    // all the matrices must be properly sized as for nystrom_ckn()
    Matrix<T>** W, Vector<T>** b, Matrix<T>** W2, Matrix<T>** W3,
    const int type_regul, const int type_kernel, const T sigma, int threads,
    const int iter = 10, const int iter_kmeans = 10, const T lambda2 = 0, const bool compute_cov = false) {
   if (threads == -1) {
      threads=1;
#ifdef _OPENMP
      threads =  MIN(MAX_THREADS,omp_get_num_procs());
#endif
   } 
   threads=init_omp(threads);
#ifdef HAVE_MKL
   mkl_set_num_threads(threads);
   vmlSetMode(0x00000003 | 0x00280000 | 0x00000100);
#endif
   const T lambda=0.1;
   const T sigma2=sigma*sigma;
   const int m=X.m();
   const int n=X.n();
   const int num_subspaces = subspace_centroids.n();
   Timer time;
   Timer time2;
   time.start();
   Vector<int> cluster_assignments;
   cout << "Clustering all training examples" << endl;
   fast_kmeans(X, subspace_centroids, cluster_assignments, iter_kmeans, true);
   time.stop();
   time.printElapsed();
   time.reset();

   // count the number of traing examples assigned to each cluster
   // TODO: make sure they all *have* training data, and what to do if not
   vector<int> cluster_sample_counts(num_subspaces, 0);
   for (int i=0; i<n; ++i)
      cluster_sample_counts[cluster_assignments[i]]++;

   Matrix<T> X_subspace;
   Vector<T> col;
   for (int idx_subspace=0; idx_subspace<num_subspaces; ++idx_subspace) {
     cout << "Training for cluster " << (idx_subspace+1)
          << " (" << cluster_sample_counts[idx_subspace] << " samples)" << endl;

     // for each centroid: make one matrix with the training data just for that
     // TODO: this can be done by making a more clever training function that 
     // receives a list indicating which samples to use instead of copying
     X_subspace.resize(m, cluster_sample_counts[idx_subspace]);
     int count=0;
     for (int i=0; i<n; ++i) {
       if (cluster_assignments[i] == idx_subspace) {
         X_subspace.refCol(count++, col);
         X.copyCol(i, col);
       }
     }
     
    nystrom_ckn(
        X,
        *W[idx_subspace],
        *b[idx_subspace],
        *W2[idx_subspace],
        *W3[idx_subspace],
        type_regul,
        type_kernel,
        sigma,
        threads,
        iter,
        iter_kmeans,
        lambda2,
        compute_cov);

   }
};


template <typename T>
void nystrom_ckn(const Matrix<T>& X, Matrix<T>& W, Vector<T>& b, Matrix<T>& W2,
      Matrix<T>& W3, const int type_regul, const int type_kernel, const T sigma, int threads, const int iter = 10, const int iter_kmeans = 10, const T lambda2 = 0, const bool compute_cov = false) {
   if (threads == -1) {
      threads=1;
#ifdef _OPENMP
      threads =  MIN(MAX_THREADS,omp_get_num_procs());
#endif
   } 
   threads=init_omp(threads);
#ifdef HAVE_MKL
   mkl_set_num_threads(threads);
   vmlSetMode(0x00000003 | 0x00280000 | 0x00000100);
#endif
   const T lambda=0.1;
   const T sigma2=sigma*sigma;
   const int p=W.n();
   const int m=X.m();
   const int n=X.n();
   Timer time;
   Timer time2;
   time.start();
   Vector<int> cluster_assignments_ignored;
   fast_kmeans(X, W, cluster_assignments_ignored, iter_kmeans);
   time.stop();
   time.printElapsed();
   time.reset();
   if (compute_cov) {
      time.start();
      X.XXt(W3);
      W3.scal(T(1.0)/n);
      time.stop();
      time.printElapsed();
      time.reset();
   } else {
      W3.setZeros();
   }
   Matrix<T> kZtZ, invkZtZ, alpha, A, B, AW;
   A.resize(p,p);
   AW.resize(p,p);
   A.setZeros();
   B.resize(m,p);
   B.setZeros();
   time.start();
   for (int ii=0; ii<iter; ++ii) {
      const T obj =eval_obj(X,W,sigma2,lambda2);
      printf("Epoch %d, obj: %g\n",ii+1,obj);
      const int size_block=W.n();
      const int nind=ceil(n/size_block);
      for (int jj =0; jj<n; jj+=size_block) {
         const int endblock=MIN(jj+size_block,n);
         const int length_block=endblock-jj;
         Matrix<T> subX, kZtX, alpha;
         X.refSubMat(jj,length_block,subX);
         compute_kZtZ(W,kZtZ,sigma2,lambda2,type_kernel);
         compute_kZtX(W,subX,kZtX,sigma2,type_kernel);
         //if (type_regul==0) {
            invkZtZ.copy(kZtZ);
            invkZtZ.invSymPos();
            invkZtZ.mult(kZtX,alpha);
         /*} else if (type_regul==1) {
            SET_MKL_ST
            ist_mt(kZtZ,kZtX,alpha,lambda,PENALTY,10);
            SET_MKL_MT
         } else if (type_regul==2) {
            simple_omp(kZtZ,kZtX,alpha,16);
         } else if (type_regul==3) {
            simple_omp(kZtZ,kZtX,alpha,10);
         } else if (type_regul==4) {
            simple_omp(kZtZ,kZtX,alpha,5);
         } else if (type_regul==5) {
            simple_omp(kZtZ,kZtX,alpha,20,true);
         }*/
         const T wt = MIN(MAX(T(0.03)*T(nind)/(ii*nind+jj+1),1/T(nind)),T(0.01));
         alpha.mult(alpha,A,false,true,wt,T(1.0)-wt);
         kZtX.mult_elementWise(alpha,kZtX);
         subX.mult(kZtX,B,false,true,wt,T(1.0)-wt);

         time2.start();
         A.mult_elementWise(kZtZ,AW);
         Vector<T> colW, colB, colA;
         Vector<T> tmp;
         for (int kk=0; kk<p; ++kk) {
            W.refCol(kk,colW);
            B.refCol(kk,colB);
            AW.refCol(kk,colA);
            tmp.copy(colB);
            W.mult(colA,tmp,-T(1.0),T(1.0));
            if (AW(kk,kk)) {
               colW.add(tmp,T(1.0)/AW(kk,kk));
            }
            colW.normalize();
         }
         time2.stop();
      }
   }
   time.stop();
   time.printElapsed();
   time2.printElapsed();
   
   compute_kZtZ(W,kZtZ,sigma2,lambda2,type_kernel);
   kZtZ.InvsqrtMat(W2);
   if (type_kernel==0 || type_kernel==4 || type_kernel==5) {
      W.scal(T(1.0)/sigma2);
      b.set(-T(1.0)/sigma2);
   } else {
      b.set(T(1.0));
   }
};

template <typename T>
inline void reduce_destroy(T** CT,T& C,const int num_threads) {
   C.copy(*(CT[0]));
   for (int ii=1; ii<num_threads; ++ii) 
      C.add(*(CT[ii]));
   for (int ii=0; ii<num_threads; ++ii) 
      delete(CT[ii]);
}

template <typename Tin, typename T>
inline void compute_gradient(const Map<Tin>& maps, Layer<T> layers[], const int
      nlayers, const Vector<int>& Y, const Matrix<T>& Z, const Matrix<T>& W, const T mu,
      Matrix<T>& A, Matrix<T>& B, Matrix<T>& C, Matrix<T>& gradG, 
      Matrix<T>& gradH, Matrix<T>& kZtZ, Vector<T>& out) { 
#ifdef HAVE_MKL
   vmlSetMode(0x00000003 | 0x00280000 | 0x00000100);
#endif
   const int n = maps.z();
   Layer<T>& layer=layers[nlayers-1];
   const T lambda2=T(0.01);
   const T sigma2=layer.sigma*layer.sigma;;
   const bool compute_gradZ = A.m() > 1;
   const bool compute_gradW = gradG.m() > 1;
   
   Timer time;
   time.start();
   /// computes kZtZ
   Matrix<T> invkZtZ;
   compute_kZtZ(Z,kZtZ,sigma2,lambda2);
   /// computes inv(kZtZ)  
   invkZtZ.copy(kZtZ);
   invkZtZ.invSymPos();

   const int nclasses=W.n();
#ifdef _OPENMP
   const int num_threads=omp_get_max_threads();
#else
   const int num_threads=1;
#endif 
   Matrix<T>** AT = new Matrix<T>*[num_threads];
   Matrix<T>** BT = new Matrix<T>*[num_threads];
   Matrix<T>** CT = new Matrix<T>*[num_threads];
   Matrix<T>** gradGT = new Matrix<T>*[num_threads];
   Matrix<T>** gradHT = new Matrix<T>*[num_threads];
   Vector<T>** outT = new Vector<T>*[num_threads];
   for (int ii=0; ii<num_threads; ++ii) {
      if (compute_gradZ) {
         AT[ii]=new Matrix<T>(Z.n(),Z.n());
         AT[ii]->setZeros();
         BT[ii]=new Matrix<T>(Z.m(),Z.n());
         BT[ii]->setZeros();
         gradHT[ii]=new Matrix<T>(Z.m(),Z.n());
         gradHT[ii]->setZeros();
      }
      if (compute_gradW) {
         CT[ii]=new Matrix<T>(W.m(),W.m());
         CT[ii]->setZeros();
         gradGT[ii]=new Matrix<T>(W.m(),W.n());
         gradGT[ii]->setZeros();
      }
      outT[ii]= new Vector<T>(out.n());
      outT[ii]->setZeros();
   }
   /*Vector<double> times1(num_threads);
   times1.setZeros();
   Vector<double> times2(num_threads);
   times2.setZeros();
   Vector<double> times3(num_threads);
   times3.setZeros();
   Vector<double> times4(num_threads);
   times4.setZeros();
   Vector<double> times5(num_threads);
   times5.setZeros();
   */
   
#pragma omp parallel for
   for (int ii=0; ii<n; ++ii) {
#ifdef _OPENMP
      const int numT=omp_get_thread_num();
#else
      const int numT=1;
#endif 
      Vector<T>& outt = *(outT[numT]);
      Map<T> map;
      Map<Tin> mapii;
      maps.refSubMapZ(ii,mapii); 
      // encode bottom layer
      //double time1=omp_get_wtime();
      encode_ckn_map(mapii,layers,nlayers-1,map);
      //times1[numT]+=omp_get_wtime()-time1;
      Matrix<T> X;
      //double time2=omp_get_wtime();
      map.im2col(X,layer.npatch,false,1);
      //times2[numT]+=omp_get_wtime()-time2;


      ///double time3=omp_get_wtime();
      // pre-processing per-example
      if (layer.type_layer == 1 || layer.type_layer == 2 || layer.type_layer == 5)
         centering(X,nlayers==1 ? map.x() : 1);
      if (layer.type_layer == 5) 
         whitening(X);

      // global pre-processing
      if (layer.type_layer == 2) {
         Vector<T> ones(X.n());
         ones.set(T(1.0));
         X.rank1Update(layer.mu,ones,-T(1.0));
         Matrix<T> Z;
         Z.copy(X);
         layer.Wfilt.mult(Z,X);
      }
      if (layer.type_layer == 3) { 
         Vector<T> ones(X.n());
         ones.set(T(1.0));
         X.rank1Update(layer.mu,ones,-T(1.0));
      }
      // records norms
      Vector<T> norms;
      normalize(X,norms);
      norms.scal(T(1.0)/norms.asum());
      //times3[numT]+=omp_get_wtime()-time3;

      // computes beta 
      //double time4=omp_get_wtime();
      Matrix<T> beta;
      compute_kZtX(Z,X,beta,sigma2);
      Matrix<T> alpha;
      invkZtZ.mult(beta,alpha);
      //times4[numT]+=omp_get_wtime()-time4;

      //double time5=omp_get_wtime();
      if (compute_gradZ) {
         Matrix<T>& At = *(AT[numT]);
         Matrix<T>& Bt = *(BT[numT]);
         // computes alpha
         Vector<T> norms_sqrt;
         norms_sqrt.copy(norms);
         norms_sqrt.Sqrt();
         alpha.multDiagRight(norms_sqrt);
         // update A
         alpha.mult(alpha,At,false,true,T(1.0),T(1.0));
         // update B
         alpha.multDiagRight(norms_sqrt);
         alpha.mult_elementWise(beta,alpha);
         X.mult(alpha,Bt,false,true,T(1.0),T(1.0));
      } else {
         alpha.multDiagRight(norms);
         alpha.mult_elementWise(beta,alpha);
      }
      outt[0] += T(0.5)*norms.asum()- alpha.sum();
      // pooling
      Vector<T> betab;
      beta.mult(norms,betab);
      outt[3] += betab.nrm2sq();
      Vector<T> gammai;
      W.multTrans(betab,gammai);
      outt[2] += (Y[ii] == gammai.max());
      
      if (compute_gradW) {
         Matrix<T>& Ct = *(CT[numT]);
         Matrix<T>& gradGt = *(gradGT[numT]);
         // sq loss
         Vector<T> Yones(nclasses);
         for (int jj=0; jj<nclasses; ++jj) {
            if (Y[ii]==jj) {
               Yones[jj]=T(1.0);
            } else {
               Yones[jj]=-T(1.0);
            }
            const T tmp=-Yones[jj]+gammai[jj];
            gammai[jj] = tmp;
            outt[1] += T(0.5)*tmp*tmp;
         }
         Ct.rank1Update(betab,betab);
         gradGt.rank1Update(betab,Yones);
      }
      
      if (compute_gradZ) {
         // update gradH
         Matrix<T>& gradHt = *(gradHT[numT]);
         Vector<T> tmp;
         W.mult(gammai,tmp);
         alpha.setZeros();
         alpha.rank1Update(tmp,norms);
         alpha.mult_elementWise(beta,alpha);
         X.mult(alpha,gradHt,false,true,T(1.0),T(1.0));
      }
      //times5[numT]+=omp_get_wtime()-time5;
   }
   time.stop();
   /*time.printElapsed();
   cout << "Time 1 " << times1.sum() << endl;
   cout << "Time 2 " << times2.sum() << endl;
   cout << "Time 3 " << times3.sum() << endl;
   cout << "Time 4 " << times4.sum() << endl;
   cout << "Time 5 " << times5.sum() << endl;
   stop();*/
   if (compute_gradZ) {
      reduce_destroy(AT,A,num_threads);
      A.scal(T(1.0)/n);
      A.mult_elementWise(kZtZ,A);
      reduce_destroy(BT,B,num_threads);
      B.scal(T(1.0)/n);
      reduce_destroy(gradHT,gradH,num_threads);
      gradH.scal(T(1.0)/(n*nclasses));
   }
   if (compute_gradW) {
      reduce_destroy(gradGT,gradG,num_threads);
      gradG.scal(T(1.0)/(n*nclasses));
      reduce_destroy(CT,C,num_threads);
      C.scal(T(1.0)/(nclasses*n));
   }
   reduce_destroy(outT,out,num_threads);
   out.scal(T(1.0)/n);
   out[0] += T(0.5)*A.sum();
   out[1] /= nclasses;
   delete[](AT);
   delete[](BT);
   delete[](CT);
   delete[](gradGT);
   delete[](gradHT);
   delete[](outT);
};

#ifdef CUDA
template <typename T>
void fast_kmeans_cuda(const Matrix<T>& X, Matrix<T>& Z, const int num_iter = 10) {
   const int n=X.n();
   const int p=Z.n();
   Vector<int> per;
   per.randperm(n);
   Vector<T> col1, col2;
   for (int ii=0; ii<p; ++ii) {
      X.refCol(per[ii],col1);
      Z.refCol(ii,col2);
      col2.copy(col1);
   };
   CudaMatrix<T> CudaZ, CudaSubX, CudaZtX;
   Matrix<T> subX, ZtX;

   for (int ii=0; ii<num_iter; ++ii) {
      printf("K-means epoch %d\n",ii+1);
      const int size_block=10*p;
      Vector<T> tmp(n);
      Vector<int> idx(n);
      CudaZ.setMatrix(Z);
      for (int jj =0; jj<n; jj+=size_block) {
         const int endblock=MIN(jj+size_block,n);
         const int length_block=endblock-jj;
         X.refSubMat(jj,length_block,subX);
         CudaSubX.setMatrix(subX);
         CudaZtX.resize(p,length_block);
         CudaZ.mult(CudaSubX,CudaZtX,true);
         CudaZtX.getMatrix(ZtX);

#pragma omp parallel for
         for (int kk=0; kk<length_block; ++kk) {
            Vector<T> col;
            ZtX.refCol(kk,col);
            idx[jj+kk]=col.max();
            tmp[jj+kk]=col[idx[jj+kk]];
         }
      }
      Vector<int> numElem(p);
      numElem.setZeros();
      Z.setZeros();
      Vector<T> col1, col2;
      for (int jj =0; jj<n; ++jj) {
         const int ind=idx[jj];
         numElem[ind]++;
         X.refCol(jj,col1);
         Z.refCol(ind,col2);
         col2.add(col1);
      }
      for (int jj =0; jj<p; ++jj) {
         Z.refCol(jj,col1);
         if (numElem[jj]) {
            col1.normalize();
         } else {
            const int ind=tmp.min();
            tmp[ind]=1;
            X.refCol(ind,col2);
            col1.copy(col2);
         }
      }
   };
};

template <typename T>
void compute_kZtZ_cuda(const CudaMatrix<T>& Z, CudaMatrix<T>& ZtZ, const CudaVector<T>& ones, const T sigma2,const T lambda2) {
   const int p = Z.n();
   ZtZ.resize(p,p);
   ZtZ.set(-T(1.0)/sigma2);
   Z.mult(Z,ZtZ,true,false,T(1.0)/sigma2,T(1.0));
   ZtZ.exp();
   ZtZ.addDiag(ones,lambda2);
   ZtZ.scal(T(1.0)/(T(1.0)+lambda2));
};

template <typename T>
void compute_kZtX_cuda(const CudaMatrix<T>& Z, const CudaMatrix<T>& X, CudaMatrix<T>& ZtX, const T sigma2, const T lambda2) {
   ZtX.resize(Z.n(),X.n());
   ZtX.set(-T(1.0)/sigma2);
   Z.mult(X,ZtX,true,false,T(1.0)/sigma2,T(1.0));
   ZtX.exp();
//   ZtX.scal(sqrt(T(1.0)/(T(1.0)+lambda2)));
};

template <typename T>
T eval_obj_cuda(const Matrix<T>& X, const CudaMatrix<T>& W, const CudaVector<T>& ones, const T sigma2, const T lambda2) {
   const int n=X.n();
   const int p=W.n();
   T obj=0;

   CudaMatrix<T> kZtZ;
   compute_kZtZ_cuda(W,kZtZ,ones,sigma2,lambda2);

   CudaMatrix<T> InvkZtZ, eye;
   eye.resize(p,p);
   eye.setZeros();
   eye.addDiag(ones);
   kZtZ.solveSymPos(eye,InvkZtZ);
   CudaMatrix<T> CudaSubX;
   CudaMatrix<T> kZtX;
   CudaMatrix<T> alpha;
   
   const int size_block=p >= 1024 ? p : 10*p;
//#pragma omp parallel for
   for (int jj =0; jj<n; jj+=size_block) {
      const int endblock=MIN(jj+size_block,n);
      const int length_block=endblock-jj;
      Matrix<T> subX;
      X.refSubMat(jj,length_block,subX);
      CudaSubX.setMatrix(subX);
      compute_kZtX_cuda(W,CudaSubX,kZtX,sigma2,lambda2);
      alpha.resize(p,length_block);
      InvkZtZ.mult(kZtX,alpha);
      kZtX.mult_elementWise(alpha);
      CudaVector<T> sums;
      sums.resize(length_block);
      kZtX.multTrans(ones,sums);
      CudaVector<T> ones2;
      ones2.resize(length_block);
      ones2.set(T(1.0));
      obj += length_block-ones2.dot(sums);
   }
   return obj/(2*n);
}




template <typename T>
void nystrom_ckn_cuda(const Matrix<T>& X, Matrix<T>& W, Vector<T>& b, Matrix<T>& W2,
      Matrix<T>& W3, const int type_regul, const T sigma, int threads, const int iter = 10, const int iter_kmeans = 10, const T lambda2 = 0) {
   if (threads == -1) {
      threads=1;
#ifdef _OPENMP
      threads =  MIN(MAX_THREADS,omp_get_num_procs());
#endif
   } 
   threads=init_omp(threads);
   mkl_set_num_threads(threads);
#ifdef HAVE_MKL
   vmlSetMode(0x00000003 | 0x00280000 | 0x00000100);
#endif
   const T lambda=0.1;
   const T sigma2=sigma*sigma;
   const int p=W.n();
   const int m=X.m();
   const int n=X.n();

   CudaVector<T> ones; 
   ones.resize(p);
   ones.set(T(1.0));
   CudaVector<T> ones2; 
   ones2.resize(m);
   ones2.set(T(1.0));

   Timer time;
   time.start();
   Vector<int> cluster_assignments_ignored;
   fast_kmeans(X, W, cluster_assignments_ignored, iter_kmeans);
   time.stop();
   time.printElapsed();
   time.reset();
   CudaMatrix<T> CudaW;
   CudaW.setMatrix(W);

   CudaMatrix<T> A, kZtZ, copykZtZ, AW, B, alpha, kZtX;
   A.resize(p,p);
   AW.resize(p,p);
   A.setZeros();
   B.resize(m,p);
   B.setZeros();

   time.start();
   Matrix<T> subX;
   CudaMatrix<T> CudaSubX;
   CudaMatrix<T> grad;
   CudaVector<T> diag;
   for (int ii=0; ii<iter; ++ii) {
      const T obj =eval_obj_cuda(X,CudaW,ones,sigma2,lambda2);
      printf("Epoch %d, obj: %g\n",ii+1,obj);
      const int size_block=W.n();
      const int nind=ceil(n/size_block);
      for (int jj =0; jj<n; jj+=size_block) {
         const int endblock=MIN(jj+size_block,n);
         const int length_block=endblock-jj;
         if (length_block < size_block) {
            break;
         }
         compute_kZtZ_cuda(CudaW,kZtZ,ones,sigma2,lambda2);
         X.refSubMat(jj,length_block,subX);
         CudaSubX.setMatrix(subX);
         compute_kZtX_cuda(CudaW,CudaSubX,kZtX,sigma2,lambda2);
         copykZtZ.copy(kZtZ);
         copykZtZ.solveSymPos(kZtX,alpha);
         const T wt = MIN(MAX(T(0.03)*T(nind)/(ii*nind+jj+1),1/T(nind)),T(0.01));
         alpha.mult(alpha,A,false,true,wt,T(1.0)-wt);

         AW.copy(A);
         AW.mult_elementWise(kZtZ);
         AW.diag(diag);
         kZtX.mult_elementWise(alpha); // kZtX is changed here
         CudaSubX.mult(kZtX,B,false,true,wt,T(1.0)-wt);
         grad.copy(B);
         CudaW.mult(AW,grad,false,false,-T(1.0),T(1.0));
         CudaW.add(grad,T(0.5)*p/diag.dot(ones));
         grad.copy(CudaW);
         grad.sqr();
         grad.multTrans(ones2,diag);
         diag.inv_sqrt();
         CudaW.multDiagRight(diag);
      } 
   }

   time.stop();
   time.printElapsed();

   compute_kZtZ_cuda(CudaW,kZtZ,ones,sigma2,T(0));
   W3.setZeros();
   Matrix<T> cpukZtZ;
   kZtZ.getMatrix(cpukZtZ);
   cpukZtZ.InvsqrtMat(W2,lambda2);
   CudaW.getMatrix(W);
   W.scal(T(1.0)/sigma2);
   b.set(-T(1.0)/sigma2);
};

#endif 

template <typename T>
inline void backprop_map(Map<T>& mapin, Layer<T> layers[], Matrix<T> kZtZ[],const Matrix<T> ZtZm1[], const int nlayers, const int y, const Vector<T>& Y, const Matrix<T>& Wall, const Vector<T>& b, Matrix<T>& gradWall, Vector<T>& gradalpha, Vector<T>& psi, const bool withA=true, const bool only_last_layer=false, const bool normalize_last_layer = false, const int loss = 0, const bool bugA = true, const bool bugB = true, const bool compute_gradalpha=false, const bool compute_gradW=false) {
   Map<T> maps_set1[nlayers]; // (mat1) E.psi^{k-1}
   Map<T> maps_set1b[nlayers]; // (mat1b) Z_k^T E.psi^{k-1} S_k^{-1}
   Map<T> maps_set2[nlayers]; // (mat2) sigma_k'(Z_k^T E.psi^{k-1} S_k^{-1})
   Map<T> maps_set3[nlayers]; // (mat3) A_k sigma_k(Z_k^T E.psi^{k-1} S_k^{-1}) S_k
   Map<T> maps_set4[nlayers]; // A_k sigma_k(Z_k^T E.psi^{k-1} S_k^{-1}) S_k P_k
   Vector<T> norms_set[nlayers]; // S_k
   START_TIMER(0)
   /// forward pass
   for (int ii=0; ii<nlayers; ++ii) {
      Layer<T>& layer=layers[ii];
      Matrix<T>& Zmod = layers[ii].W;
      Matrix<T>& A = layers[ii].W2;
      Vector<T>& norms = norms_set[ii];
      const T sigma2=layer.sigma*layer.sigma;
      const T betascal=layer.new_subsampling ? sqr<T>(T(2.0)) :  T(2.0);
      const T beta = layer.subsampling > 0 ? layer.subsampling/betascal : layer.sub_float/betascal;
      Matrix<T> Z;
      Z.copy(Zmod);
      Z.scal(sigma2);

      /// definition of the maps
      Map<T>& map0 = ii==0 ? mapin :  maps_set4[ii-1];
      Map<T>& map1 = maps_set1[ii];
      Map<T>& map1b = maps_set1b[ii];
      const int yyout = layer.zero_padding ? map0.y() : map0.y() - layer.npatch + 1;
      const int zzout = layer.zero_padding ? map0.z() : map0.z() - layer.npatch + 1;
      const int yout=ceil(yyout/static_cast<double>(layer.stride));
      const int zout=ceil(zzout/static_cast<double>(layer.stride));
      map1.resize(layer.npatch*layer.npatch*map0.x(),yout,zout);
      const int newm = layer.type_kernel==3 ? map1.x()*map1.x() : Z.n();
      map1b.resize(newm,yout,zout);
      Map<T>& map2 = maps_set2[ii];
      map2.resize(newm,yout,zout);
      Map<T>& map3 = maps_set3[ii];
      map3.resize(newm,yout,zout);
      Map<T>& map4 = maps_set4[ii]; // size will be defined later
      /// definition of the matrices of patches
      Matrix<T> mat1, mat1b, mat2, mat3, tmp;
      map1.refMat(mat1);
      map1b.refMat(mat1b);
      map2.refMat(mat2);
      map3.refMat(mat3);

      if (layer.type_layer==4) {
         encode_layer(map0,map4,layer);
      } else {
         /// extract patches (apply operator E)
         map0.im2col(mat1,layer.npatch,layer.zero_padding,layer.stride);
         pre_processing(mat1,layer, layer.num_layer==1 ? mapin.x() : 1);

         /// compute mat2
         if (layer.type_kernel==3) {
            expand_XXt(mat1,mat3);
         } else  {
            tmp.copy(mat1);
            normalize(tmp,norms); 
            Z.mult(tmp,mat1b,true,false);
            compute_kZtXb(mat1b,mat2,sigma2,layer.type_kernel);
            tmp.copy(mat2);
            if (layer.type_kernel==1) {
               mat2.pow(T(0.5));
               //mat2.scal(T(2.0));
            } else if (layer.type_kernel==2) {
               mat2.pow(T(2.0)/T(3.0));
               mat2.scal(T(3.0)/T(2.0));
            } else if (layer.type_kernel==4) {
               mat2.add(exp(-T(1.0)/sigma2));
            } else if (layer.type_kernel==5) {
               mat2.add(exp(-T(2.0)/sigma2));
            }
            tmp.multDiagRight(norms);
            /// compute mat3
            if (A.n() > 1) {
               A.mult(tmp,mat3);
            } else {
               mat3.copy(tmp);
            }
         } 
         if (layer.subsampling == 0) {
            map3.subsampling_new2(map4,layer.sub_float,beta);
         } else if (layer.subsampling == -1) {
            map3.subsampling_new3(map4,layer.sub_float,beta);
         } else if (layer.subsampling == -2) {
            map3.subsampling_new3(map4,T(1.0),beta);
         } else if (layer.subsampling > 1) {
            map3.subsampling_new(map4,layer.subsampling,beta);
         } else {
            map4.copy(map3);
         }
      }
   }
   STOP_TIMER(0)
   START_TIMER(1)

   /// evaluate loss and gradient loss
   const int nclasses=Wall.n();
   Vector<T> psicopy, gamma, psicopy2;
   maps_set4[nlayers-1].refVec(psicopy);
   psi.copy(psicopy);
   psi.add(-psi.mean());
   T norm_psi;
   if (normalize_last_layer) {
      norm_psi=psi.nrm2();
      psi.scal(T(1.0)/norm_psi);
   }

   if (Y.n() > 0) {
      /// psi is a vector
      /// Wall is here a vector
      Vector<T> Wallb;
      Wall.toVect(Wallb);
      Matrix<T> psib(psi.rawX(),Wallb.n(),psi.n()/Wall.m());
      psib.multTrans(Wallb,gamma);
      //Wall.multTrans(psi,gamma,T(1.0));
      gamma.sub(Y);
   } else {
      if (loss==0) {
         Wall.multTrans(psi,gamma,-T(1.0));
         if (b.n() > 0)
            gamma.sub(b);
         // make it loss dependent
         gamma[y]=-gamma[y];
         for (int ii=0; ii<nclasses; ++ii)
            gamma[ii]= 2*MAX(1-gamma[ii],0)/nclasses;
         gamma[y]=-gamma[y];
      } else if (loss==1) {
         Wall.multTrans(psi,gamma);
         if (b.n() > 0)
            gamma.add(b);
         gamma.add(b);
         gamma.add(T(1.0)-gamma[y]);
         gamma.thrsPos();
         gamma[y]=0;
         gamma[y]=-gamma.asum();
      } else if (loss==2) {
         Wall.multTrans(psi,gamma);
         if (b.n() > 0)
            gamma.add(b);
         const T mmax=gamma.maxval();
         gamma.add(-mmax);
         gamma.exp();
         gamma.scal(T(1.0)/gamma.asum());
         for (int ii=0; ii<nclasses; ++ii)
            if (gamma[ii] < T(1e-3))
               gamma[ii]=0;
         gamma[y]-=T(1.0);
      } else if (loss==3) {
         Wall.multTrans(psi,gamma,-T(1.0));
         if (b.n() > 0)
            gamma.sub(b);
         gamma[y]=-gamma[y];
         for (int ii=0; ii<nclasses; ++ii)
            gamma[ii]= 2*MAX(1-gamma[ii],0)/nclasses;
         gamma[y]=-gamma[y];
         gamma.scal(T(1.0)/(1+(nclasses-2)/static_cast<T>(nclasses)));
         gamma[y] *= nclasses-1;
      } else if (loss==4) {
         const T beta=floor(sqrt(nclasses-2));
         Wall.multTrans(psi,gamma,-T(1.0));
         if (b.n() > 0)
            gamma.sub(b);
         gamma[y]=-gamma[y];
         for (int ii=0; ii<nclasses; ++ii)
            gamma[ii]= 2*MAX(1-gamma[ii],0)/nclasses;
         gamma[y]=-gamma[y];
         gamma.scal(T(1.0)/(1+beta/static_cast<T>(nclasses)));
         gamma[y] *= (beta+T(1.0));
      }
   }
   if (gamma.nrm2sq() < T(1e-6))
      return;
   if (compute_gradW) {
#pragma omp critical
      {
         gradWall.rank1Update(psi,gamma);
      }
   }
   //gamma.print("gamma");
   //psi.print("psi");

   /// backward pass initialization
   Matrix<T> grads[nlayers];
   for (int ii=0; ii<nlayers; ++ii) { 
      grads[ii].resize(layers[ii].gradW.m(),layers[ii].gradW.n());
      grads[ii].setZeros();
   }

   STOP_TIMER(1)
   START_TIMER(2)
   /// backward pass
   if (bugB) {
      Matrix<T> W(maps_set4[nlayers-1].x(),maps_set4[nlayers-1].y()*maps_set4[nlayers-1].z());
      Vector<T> Wvec;
      W.toVect(Wvec);
      if (Y.n() > 0) {
         Vector<T> Wallb;
         Wall.toVect(Wallb);
         W.setZeros();
         W.rank1Update(Wallb,gamma);
      } else {
         Wall.mult(gamma,Wvec);
      }
      if (normalize_last_layer) {
         Wvec.scal(T(1.0)/norm_psi);
         Wvec.add(psi,-psi.dot(Wvec));
      }
      for (int ii=nlayers-1; ii>=0; --ii) {
         if (layers[ii].type_layer==4)
            break;

         Matrix<T>& grad = grads[ii];
         Layer<T>& layer=layers[ii];
         Matrix<T>& Zmod = layers[ii].W;
         Matrix<T>& A = layers[ii].W2;
         Matrix<T>& Ah = layers[ii].W3;
         Vector<T>& norms = norms_set[ii];
         const T sigma2=layer.sigma*layer.sigma;
         const T betascal=layer.new_subsampling ? sqr<T>(T(2.0)) :  T(2.0);
         const T beta = layer.subsampling > 0 ? layer.subsampling/betascal : layer.sub_float/betascal;

         /// definition of the maps
         Map<T>& map0 = ii==0 ? mapin :  maps_set4[ii-1];
         Map<T>& map1 = maps_set1[ii];
         Map<T>& map1b = maps_set1b[ii];
         Map<T>& map2 = maps_set2[ii];
         Map<T>& map3 = maps_set3[ii];
         Map<T>& map4 = maps_set4[ii]; 
         /// definition of the matrices of patches
         Matrix<T> mat1, mat1b, mat2, mat3, mat4;
         map1.refMat(mat1);
         map1b.refMat(mat1b);
         map2.refMat(mat2);
         map3.refMat(mat3);
         map4.refMat(mat4);
         Vector<T> tmp_vec, tmp_vec2;
         Matrix<T> tmp, tmp2, tmp3;

         if (layer.type_kernel==3) {
            Map<T> mapW(W.rawX(),map4.x(),map4.y(),map4.z());
            Map<T> mapWPt;
            mapWPt.resize(map3.x(),map3.y(),map3.z());
            if (layer.subsampling == 0) {
               mapW.upscaling_new2(mapWPt,layer.sub_float,beta);
            } else if (layer.subsampling == -1) {
               mapW.upscaling_new3(mapWPt,layer.sub_float,beta);
            } else if (layer.subsampling == -2) {
               mapW.upscaling_new3(mapWPt,T(1.0),beta);
            } else if (layer.subsampling > 1) {
               mapW.upscaling_new(mapWPt,layer.subsampling,beta);
            } else {
               mapWPt.copy(mapW);
            }
            mapWPt.refMat(tmp);
            un_expand_XXt(tmp,mat1,tmp2);
            tmp2.scal(T(2.0));
            W.resize(map0.x(),map0.y()*map0.z());
            Map<T> mapup(W.rawX(),map0.x(),map0.y(),map0.z());
            mapup.col2im(tmp2,layer.npatch,layer.zero_padding,false,layer.stride); 
         } else {
            Matrix<T> Z;
            Z.copy(Zmod);
            Z.scal(sigma2);

            /// compute first term of gradient
            if (withA) {
               A.mult(W,tmp3);
               tmp3.mult(mat4,tmp2,false,true);
               Ah.mult(tmp2,tmp);
               tmp.mult(Ah,tmp2);
               tmp2.transpose(tmp);
               tmp.add(tmp2);
               if (compute_gradalpha && (layer.type_kernel==4 || layer.type_kernel==5)) {
                  const T scal = layer.type_kernel==4 ? T(1.0) : T(2.0);
                  gradalpha[ii]=scal*exp(-scal/sigma2)*tmp.sum();
               } else {
                  gradalpha[ii]=0;
               }
               tmp.mult_elementWise(kZtZ[ii],tmp);
               Z.mult(tmp,grad,false,false,-T(0.5)/sigma2);
               if (compute_gradalpha) {
                  gradalpha[ii]+=-T(0.25)*ZtZm1[ii].dot(tmp);
               }
            } else {
               tmp3.copy(W);
            }
            /// compute second term of gradient
            Map<T> mapAW(tmp3.rawX(),map4.x(),map4.y(),map4.z());
            Map<T> mapAWPt;
            mapAWPt.resize(map3.x(),map3.y(),map3.z());
            if (layer.subsampling == 0) {
               mapAW.upscaling_new2(mapAWPt,layer.sub_float,beta);
            } else if (layer.subsampling == -1) {
               mapAW.upscaling_new3(mapAWPt,layer.sub_float,beta);
            } else if (layer.subsampling == -2) {
               mapAW.upscaling_new3(mapAWPt,T(1.0),beta);
            } else if (layer.subsampling > 1) {
               mapAW.upscaling_new(mapAWPt,layer.subsampling,beta);
            } else {
               mapAWPt.copy(mapAW);
            }
            mapAWPt.refMat(tmp);
            if (compute_gradalpha && (layer.type_kernel==4 || layer.type_kernel==5)) {
               tmp.mult(norms,tmp_vec);
               const T scal = layer.type_kernel==4 ? T(1.0) : T(2.0);
               gradalpha[ii]+=tmp_vec.sum()* scal*exp(-scal/sigma2); // first term of gradient
            }
            tmp.mult_elementWise(mat2,tmp);
            mat1.mult(tmp,grad,false,true,T(1.0)/sigma2,T(1.0));

            if (compute_gradalpha) {
               Matrix<T> tmp_a;
               tmp_a.copy(mat1b);
               tmp_a.add(-T(1.0));
               tmp_a.multDiagRight(norms);
               gradalpha[ii]+=tmp_a.dot(tmp); // first term of gradient
            }
            if (ii==0 || only_last_layer) break;

            /// prepare W for the next iteration
            Z.mult(tmp,tmp2,false,false,T(1.0)/sigma2); // tmp2 contains Z*B
            Map<T> mapW(W.rawX(),map4.x(),map4.y(),map4.z());
            Map<T> mapWPt;
            mapWPt.resize(map3.x(),map3.y(),map3.z());
            if (layer.subsampling == 0) {
               mapW.upscaling_new2(mapWPt,layer.sub_float,beta);
            } else if (layer.subsampling == -1) {
               mapW.upscaling_new3(mapWPt,layer.sub_float,beta);
            } else if (layer.subsampling == -2) {
               mapW.upscaling_new3(mapWPt,T(1.0),beta);
            } else if (layer.subsampling > 1) {
               mapW.upscaling_new(mapWPt,layer.subsampling,beta);
            } else {
               mapWPt.copy(mapW);
            }
            Matrix<T> WPt;
            mapWPt.refMat(WPt);

            WPt.dot_col(mat3,tmp_vec);   // 
            tmp2.dot_col(mat1,tmp_vec2); // Z*B .* E[I_{j-1}]
            tmp_vec.sub(tmp_vec2);
            tmp_vec2.copy(norms);
            tmp_vec2.inv();
            tmp_vec2.sqr();
            for (int kk=0; kk<norms.n(); ++kk) 
               if (norms[kk] < EPS_NORM) tmp_vec2[kk]=0;
            tmp_vec.mult(tmp_vec,tmp_vec2);
            tmp3.copy(mat1);
            tmp3.multDiagRight(tmp_vec); 

            W.resize(map0.x(),map0.y()*map0.z());
            Map<T> mapup(W.rawX(),map0.x(),map0.y(),map0.z());
            tmp2.add(tmp3);  
            mapup.col2im(tmp2,layer.npatch,layer.zero_padding,false,layer.stride); 
         }
      }

   } else {
      /// here starts the slow version
   /*   for (int jj=0; jj<nclasses; ++jj) {
         if (gamma[jj]) {
            Matrix<T> Winit(Wall.rawX()+jj*Wall.m(),maps_set4[nlayers-1].x(),maps_set4[nlayers-1].y()*maps_set4[nlayers-1].z());
            Matrix<T> W;
            W.copy(Winit);
            if (normalize_last_layer) {
               Vector<T> Wvec;
               W.toVect(Wvec);
               Wvec.scal(T(1.0)/norm_psi);
               Wvec.add(psi,-psi.dot(Wvec));
            }
            for (int ii=nlayers-1; ii>=0; --ii) {
               if (layers[ii].type_layer==4)
                  break;

               Matrix<T>& grad = grads[ii];
               Layer<T>& layer=layers[ii];
               Matrix<T>& Zmod = layers[ii].W;
               Matrix<T>& A = layers[ii].W2;
               Matrix<T>& Ah = layers[ii].W3;
               Vector<T>& norms = norms_set[ii];
               const T sigma2=layer.sigma*layer.sigma;
               const T betascal=layer.new_subsampling ? sqr<T>(T(2.0)) :  T(2.0);
               const T beta = layer.subsampling ? layer.subsampling/betascal : layer.sub_float/betascal;

               /// definition of the maps
               Map<T>& map0 = ii==0 ? mapin :  maps_set4[ii-1];
               Map<T>& map1 = maps_set1[ii];
               Map<T>& map2 = maps_set2[ii];
               Map<T>& map3 = maps_set3[ii];
               Map<T>& map4 = maps_set4[ii]; 
               /// definition of the matrices of patches
               Matrix<T> mat1, mat2, mat3, mat4;
               map1.refMat(mat1);
               map2.refMat(mat2);
               map3.refMat(mat3);
               map4.refMat(mat4);
               Vector<T> tmp_vec, tmp_vec2;
               Matrix<T> tmp, tmp2, tmp3;

               if (layer.type_kernel==3) {
                  Map<T> mapW(W.rawX(),map4.x(),map4.y(),map4.z());
                  Map<T> mapWPt;
                  mapWPt.resize(map3.x(),map3.y(),map3.z());
                  if (layer.subsampling == 0) {
                     mapW.upscaling_new2(mapWPt,layer.sub_float,beta);
                  } else if (layer.subsampling > 1) {
                     mapW.upscaling_new(mapWPt,layer.subsampling,beta);
                  } else {
                     mapWPt.copy(mapW);
                  }
                  mapWPt.refMat(tmp);
                  un_expand_XXt(tmp,mat1,tmp2);
                  tmp2.scal(T(2.0));
                  W.resize(map0.x(),map0.y()*map0.z());
                  Map<T> mapup(W.rawX(),map0.x(),map0.y(),map0.z());
                  mapup.col2im(tmp2,layer.npatch,layer.zero_padding,false,layer.stride); 
               } else {
                  Matrix<T> Z;
                  Z.copy(Zmod);
                  Z.scal(sigma2);

                  /// compute first term of gradient
                  if (withA) {
                     if (bugA) {
                        A.mult(W,tmp3);
                        tmp3.mult(mat4,tmp2,false,true);
                        Ah.mult(tmp2,tmp);
                        tmp.mult(Ah,tmp2);
                        tmp2.transpose(tmp);
                        tmp.add(tmp2);
                        tmp.mult_elementWise(kZtZ[ii],tmp);
                        Z.mult(tmp,grad,false,false,-T(0.5)*gamma[jj]/sigma2,T(1.0));
                     } else {
                        A.mult(W,tmp3);
                        A.mult(tmp3,tmp2);
                        tmp2.mult(mat4,tmp,false,true);
                        tmp.transpose(tmp2);
                        tmp.add(tmp2);
                        tmp.mult_elementWise(kZtZ[ii],tmp);
                        Z.mult(tmp,grad,false,false,-T(0.5)*gamma[jj]/sigma2,T(1.0));
                     }
                  } else {
                     tmp3.copy(W);
                  }
                  /// compute second term of gradient
                  Map<T> mapAW(tmp3.rawX(),map4.x(),map4.y(),map4.z());
                  Map<T> mapAWPt;
                  mapAWPt.resize(map3.x(),map3.y(),map3.z());
                  if (layer.subsampling == 0) {
                     mapAW.upscaling_new2(mapAWPt,layer.sub_float,beta);
                  } else if (layer.subsampling > 1) {
                     mapAW.upscaling_new(mapAWPt,layer.subsampling,beta);
                  } else {
                     mapAWPt.copy(mapAW);
                  }
                  mapAWPt.refMat(tmp);
                  tmp.mult_elementWise(mat2,tmp);
                  mat1.mult(tmp,grad,false,true,gamma[jj]/sigma2,T(1.0));
                  if (ii==0 || only_last_layer) break;

                  /// prepare W for the next iteration
                  Z.mult(tmp,tmp2,false,false,T(1.0)/sigma2); // tmp2 contains B
                  Map<T> mapW(W.rawX(),map4.x(),map4.y(),map4.z());
                  Map<T> mapWPt;
                  mapWPt.resize(map3.x(),map3.y(),map3.z());
                  if (layer.subsampling == 0) {
                     mapW.upscaling_new2(mapWPt,layer.sub_float,beta);
                  } else if (layer.subsampling > 1) {
                     mapW.upscaling_new(mapWPt,layer.subsampling,beta);
                  } else {
                     mapWPt.copy(mapW);
                  }
                  Matrix<T> WPt;
                  mapWPt.refMat(WPt);

                  WPt.dot_col(mat3,tmp_vec);
                  tmp2.dot_col(mat1,tmp_vec2);
                  tmp_vec.sub(tmp_vec2);
                  tmp_vec2.copy(norms);
                  tmp_vec2.inv();
                  tmp_vec2.sqr();
                  for (int kk=0; kk<norms.n(); ++kk) 
                     if (norms[kk] < EPS_NORM) tmp_vec2[kk]=0;
                  tmp_vec.mult(tmp_vec,tmp_vec2);
                  tmp3.copy(mat1);
                  tmp3.multDiagRight(tmp_vec); 
                  tmp2.add(tmp3);  
                  W.resize(map0.x(),map0.y()*map0.z());
                  Map<T> mapup(W.rawX(),map0.x(),map0.y(),map0.z());
                  mapup.col2im(tmp2,layer.npatch,layer.zero_padding,false,layer.stride); 
               }
            }
         }
      } */
   }
   STOP_TIMER(2)

   /// update gradient with omp critical
#pragma omp critical
   {
      for (int ii=0; ii<nlayers; ++ii) { 
         if (layers[ii].type_kernel!=3 && layers[ii].type_layer != 4)
            layers[ii].gradW.add(grads[ii]);
      }
   }
}

template <typename T>
inline void backprop_inputmap(Map<T>& mapin, Layer<T> layers[], const int nlayers, Vector<T>& grad, Vector<T>& psi, const bool withA=true, const bool normalize = false) {
   Map<T> maps_set1[nlayers]; // (mat1) E.psi^{k-1}
   Map<T> maps_set2[nlayers]; // (mat2) sigma_k(Z_k^T E.psi^{k-1} S_k^{-1})
   Map<T> maps_set3[nlayers]; // (mat3) A_k sigma_k(Z_k^T E.psi^{k-1} S_k^{-1}) S_k
   Map<T> maps_set4[nlayers]; // A_k sigma_k(Z_k^T E.psi^{k-1} S_k^{-1}) S_k P_k
   Vector<T> norms_set[nlayers]; // S_k
   /// forward pass
   for (int ii=0; ii<nlayers; ++ii) {
      Layer<T>& layer=layers[ii];
      Matrix<T>& Zmod = layers[ii].W;
      Matrix<T>& A = layers[ii].W2;
      Vector<T>& norms = norms_set[ii];
      const T sigma2=layer.sigma*layer.sigma;
      const T betascal=layer.new_subsampling ? sqr<T>(T(2.0)) :  T(2.0);
      const T beta = layer.subsampling > 0 ? layer.subsampling/betascal : layer.sub_float/betascal;
      Matrix<T> Z;
      Z.copy(Zmod);
      Z.scal(sigma2);

      /// definition of the maps
      Map<T>& map0 = ii==0 ? mapin :  maps_set4[ii-1];
      Map<T>& map1 = maps_set1[ii];
      const int yyout = layer.zero_padding ? map0.y() : map0.y() - layer.npatch + 1;
      const int zzout = layer.zero_padding ? map0.z() : map0.z() - layer.npatch + 1;
      const int yout=ceil(yyout/static_cast<double>(layer.stride));
      const int zout=ceil(zzout/static_cast<double>(layer.stride));
      map1.resize(layer.npatch*layer.npatch*map0.x(),yout,zout);
      const int newm = layer.type_kernel==3 ? map1.x()*map1.x() : Z.n();
      Map<T>& map2 = maps_set2[ii];
      map2.resize(newm,yout,zout);
      Map<T>& map3 = maps_set3[ii];
      map3.resize(newm,yout,zout);
      Map<T>& map4 = maps_set4[ii]; // size will be defined later
      /// definition of the matrices of patches
      Matrix<T> mat1, mat2, mat3, tmp;
      map1.refMat(mat1);
      map2.refMat(mat2);
      map3.refMat(mat3);

      if (layer.type_layer==4) {
         encode_layer(map0,map4,layer);
      } else {

         /// extract patches (apply operator E)
         map0.im2col(mat1,layer.npatch,layer.zero_padding,layer.stride);
         pre_processing(mat1,layer, layer.num_layer==1 ? mapin.x() : 1);

         /// compute mat2
         if (layer.type_kernel==0) {
            tmp.copy(mat1);
            normalize(tmp,norms); 
            compute_kZtX(Z,tmp,mat2,sigma2);
            tmp.copy(mat2);
            tmp.multDiagRight(norms);
            /// compute mat3
            if (A.n() > 1) {
               A.mult(tmp,mat3);
            } else {
               mat3.copy(tmp);
            }
         } else if (layer.type_kernel==3) {
            expand_XXt(mat1,mat3);
         }
         if (layer.subsampling == 0) {
            map3.subsampling_new2(map4,layer.sub_float,beta);
         } else if (layer.subsampling == -1) {
            map3.subsampling_new3(map4,layer.sub_float,beta);
         } else if (layer.subsampling == -2) {
            map3.subsampling_new3(map4,T(1.0),beta);
         } else if (layer.subsampling > 1) {
            map3.subsampling_new(map4,layer.subsampling,beta);
         } else {
            map4.copy(map3);
         }
      }
   }

   /// evaluate loss and gradient loss
   Vector<T> psicopy;
   maps_set4[nlayers-1].refVec(psicopy);
   psi.copy(psicopy);
   T norm_psi;

   /// backward pass
   Matrix<T> Winit(psi.rawX(),maps_set4[nlayers-1].x(),maps_set4[nlayers-1].y()*maps_set4[nlayers-1].z());
   Matrix<T> W;
   W.copy(Winit);
   for (int ii=nlayers-1; ii>=0; --ii) {
      if (layers[ii].type_layer==4)
         break;

      Layer<T>& layer=layers[ii];
      Matrix<T>& Zmod = layers[ii].W;
      Matrix<T>& A = layers[ii].W2;
      Matrix<T>& Ah = layers[ii].W3;
      Vector<T>& norms = norms_set[ii];
      const T sigma2=layer.sigma*layer.sigma;
      const T betascal=layer.new_subsampling ? sqr<T>(T(2.0)) :  T(2.0);
      const T beta = layer.subsampling > 0 ? layer.subsampling/betascal : layer.sub_float/betascal;

      /// definition of the maps
      Map<T>& map0 = ii==0 ? mapin :  maps_set4[ii-1];
      Map<T>& map1 = maps_set1[ii];
      Map<T>& map2 = maps_set2[ii];
      Map<T>& map3 = maps_set3[ii];
      Map<T>& map4 = maps_set4[ii]; 
      /// definition of the matrices of patches
      Matrix<T> mat1, mat2, mat3, mat4;
      map1.refMat(mat1);
      map2.refMat(mat2);
      map3.refMat(mat3);
      map4.refMat(mat4);
      Vector<T> tmp_vec, tmp_vec2;
      Matrix<T> tmp, tmp2, tmp3;

      Matrix<T> Z;
      Z.copy(Zmod);
      Z.scal(sigma2);

      /// compute first term of gradient
      if (withA) {
         A.mult(W,tmp3);
      } else {
         tmp3.copy(W);
      }
      /// compute second term of gradient
      Map<T> mapAW(tmp3.rawX(),map4.x(),map4.y(),map4.z());
      Map<T> mapAWPt;
      mapAWPt.resize(map3.x(),map3.y(),map3.z());
      if (layer.subsampling == 0) {
         mapAW.upscaling_new2(mapAWPt,layer.sub_float,beta);
      } else if (layer.subsampling == -1) {
         mapAW.upscaling_new3(mapAWPt,layer.sub_float,beta);
      } else if (layer.subsampling == -2) {
         mapAW.upscaling_new3(mapAWPt,T(1.0),beta);
      } else if (layer.subsampling > 1) {
         mapAW.upscaling_new(mapAWPt,layer.subsampling,beta);
      } else {
         mapAWPt.copy(mapAW);
      }
      mapAWPt.refMat(tmp);
      tmp.mult_elementWise(mat2,tmp);

      /// prepare W for the next iteration
      Z.mult(tmp,tmp2,false,false,T(1.0)/sigma2); // tmp2 contains B
      Map<T> mapW(W.rawX(),map4.x(),map4.y(),map4.z());
      Map<T> mapWPt;
      mapWPt.resize(map3.x(),map3.y(),map3.z());
      if (layer.subsampling == 0) {
         mapW.upscaling_new2(mapWPt,layer.sub_float,beta);
      } else if (layer.subsampling == -1) {
         mapW.upscaling_new3(mapWPt,layer.sub_float,beta);
      } else if (layer.subsampling == -2) {
         mapW.upscaling_new3(mapWPt,T(1.0),beta);
      } else if (layer.subsampling > 1) {
         mapW.upscaling_new(mapWPt,layer.subsampling,beta);
      } else {
         mapWPt.copy(mapW);
      }
      Matrix<T> WPt;
      mapWPt.refMat(WPt);

      WPt.dot_col(mat3,tmp_vec);
      tmp2.dot_col(mat1,tmp_vec2);
      tmp_vec.sub(tmp_vec2);
      tmp_vec2.copy(norms);
      tmp_vec2.inv();
      tmp_vec2.sqr();
      for (int kk=0; kk<norms.n(); ++kk) 
         if (norms[kk] < EPS_NORM) tmp_vec2[kk]=0;
      tmp_vec.mult(tmp_vec,tmp_vec2);
      tmp3.copy(mat1);
      tmp3.multDiagRight(tmp_vec); 
      tmp2.add(tmp3);  
      W.resize(map0.x(),map0.y()*map0.z());
      Map<T> mapup(W.rawX(),map0.x(),map0.y(),map0.z());
      mapup.col2im(tmp2,layer.npatch,layer.zero_padding,false,layer.stride); 
   }
   Vector<T> Wvec;
   W.toVect(Wvec);
   grad.copy(Wvec);
}

template <typename Tin, typename T>
inline void ckn_backprop(const Map<Tin>& maps, Layer<T> layers[],const int nlayers, const Vector<int>& y, const Matrix<T>& Y, const Matrix<T>& W, const Vector<T>& b, Matrix<T>& gradW, Vector<T>& gradalpha, Matrix<T>& psi, const bool withA=true, const bool only_last_layer=false, const bool normalize_last_layer = false, const int loss = 0, const bool bugA = true, const bool bugB = true, const bool compute_gradalpha = false, const bool compute_gradW = false) {
   const int n = maps.z();
   Matrix<T> kZtZ[nlayers]; // S_k
   Matrix<T> ZtZm1[nlayers]; // S_k
#pragma omp parallel for
   for (int ii=0; ii<nlayers; ++ii) {
      layers[ii].gradW.setZeros();
      const T sigma2=layers[ii].sigma*layers[ii].sigma;
      if (withA && layers[ii].type_kernel!=3) {
         Matrix<T>& Zmod = layers[ii].W;
         Matrix<T> Z;
         Z.copy(Zmod);
         Z.scal(sigma2);
         Z.mult(Z,ZtZm1[ii],true,false);
         ZtZm1[ii].add(-T(1.0));
         compute_kZtZ(Z,kZtZ[ii],sigma2,T(0),layers[ii].type_kernel);
         if (layers[ii].type_kernel==1) {
            kZtZ[ii].pow(T(0.5));
         } else if (layers[ii].type_kernel==2) {
            kZtZ[ii].pow(T(2.0)/T(3.0));
            kZtZ[ii].scal(T(3.0/2.0));
         } else if (layers[ii].type_kernel==4) {
            kZtZ[ii].add(exp(-T(1.0)/sigma2));
         } else if (layers[ii].type_kernel==5) {
            kZtZ[ii].add(exp(-T(2.0)/sigma2));
         }
      }
   }
   if (compute_gradW)
      gradW.setZeros();
   if (compute_gradalpha)
      gradalpha.setZeros();
#pragma omp parallel for
   for (int ii=0; ii<n; ++ii) {
      /// forward pass
      Map<Tin> mapii;
      maps.refSubMapZ(ii,mapii); 
      Map<T> map;
      get_zeromap(mapii,map,layers[0].type_layer);
      Vector<T> psivec;
      psi.refCol(ii,psivec);
      Vector<T> gradalpha2(nlayers);
      gradalpha2.setZeros();
      Vector<T> Ycol;
      if (Y.m() > 0) {
         Y.refCol(ii,Ycol);
         backprop_map(map,layers,kZtZ,ZtZm1,nlayers,0,Ycol,W,b,gradW,gradalpha2,psivec,withA,only_last_layer,normalize_last_layer,loss,bugA,bugB,compute_gradalpha,compute_gradW);
      } else {
         backprop_map(map,layers,kZtZ,ZtZm1,nlayers,y[ii],Ycol,W,b,gradW,gradalpha2,psivec,withA,only_last_layer,normalize_last_layer,loss,bugA,bugB,compute_gradalpha,compute_gradW);
      }
      if (compute_gradalpha) {
#pragma omp critical 
         {
            gradalpha.add(gradalpha2);
         }
      }

   }
   for (int ii=0; ii<nlayers; ++ii) {
      layers[ii].gradW.scal(T(1.0)/n);
   }
   if (compute_gradW)
      gradW.scal(T(1.0)/n);
   if (compute_gradalpha)
      gradalpha.scal(T(1.0)/n);
};

template <typename Tin, typename T>
inline void ckn_backprop_input(const Map<Tin>& mapin, Layer<T> layers[],const int nlayers, Map<T>& grad, Vector<T>& psi, const bool withA=true, const bool normalize_last_layer = false) {
   /// forward pass
   Map<T> map;
   get_zeromap(mapin,map,layers[0].type_layer);
   Vector<T> gradI;
   grad.refVec(gradI);
   backprop_inputmap(map,layers,nlayers,gradI,psi,withA,normalize_last_layer);
};






#endif // APPROX_KERNEL_H
