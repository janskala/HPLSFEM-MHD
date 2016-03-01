#include "equations.h"

namespace mhd
{
  
  template <int dim>
  void MHDequations<dim>::constantBC(std::vector<Vector<double> > &/* lvq */,  // lin values
                        std::vector<std::vector<Tensor<1,dim> > > &/*lgq*/ ,  // lin gradients
                        std::vector<Vector<double> > &/* ovq*/ ,               // old values
                        std::vector<std::vector<Tensor<1,dim> > > &/*ogq*/,  // old gradients
                        std::vector<Vector<double> > &viq,               // initial values
                        std::vector<double > &eta,  // eta values
                        std::vector<Tensor<1,dim> > &etag, // eta gradients
                        const Tensor<1,dim> &n ,                          // normal
                        const unsigned int qp)                           // qp
  {
    // set state vector values
    for (unsigned int i = 0; i < Nv; i++){
      vl[i] = viq[qp](i);
      vo[i] = viq[qp](i);
    }
    // ... and for gradients
    for(unsigned int j = 0; j < dim; j++)
      for(unsigned int i = 0; i < Nv; i++){
          dvx[j][i]=0.0;
          dox[j][i]=0.0;
      }
      
    ETA=eta[qp];
    for(unsigned int j = 0; j < dim; j++)
      if (std::fabs(n[j])>0.5) ETAg[j]=0.0;
      else ETAg[j]=etag[qp][j];
    
  }
  
  template <int dim>
  void MHDequations<dim>::freeBC(std::vector<Vector<double> > &lvq,      // lin values
                        std::vector<std::vector<Tensor<1,dim> > > &lgq,  // lin gradients
                        std::vector<Vector<double> > &ovq,               // old values
                        std::vector<std::vector<Tensor<1,dim> > > &ogq,  // old gradients
                        std::vector<Vector<double> > &/* viq */,               // initial values
                        std::vector<double > &eta,  // eta values
                        std::vector<Tensor<1,dim> > &etag, // eta gradients
                        const Tensor<1,dim> &n,                          // normal
                        const unsigned int qp)                           // qp
  {
    // set state vector values
    for (unsigned int i = 0; i < Nv; i++){
      vl[i] = lvq[qp](i);
      vo[i] = ovq[qp](i);
    }
    double cf=0.9;
    vl[1]*=cf;
    vl[2]*=cf;
    vl[3]*=cf;
    vo[1]*=cf;
    vo[2]*=cf;
    vo[3]*=cf;
    // ... and for gradients
    for(unsigned int j = 0; j < dim; j++)
      /*if (std::fabs(n[j])>0.5){
        for(unsigned int i = 0; i < Nv; i++)
          dvx[j][i]=dox[j][i]=0.0;
        /*int k=(j+1)%dim; // normal component of B is set as div B = 0
        int l=(k+1)%dim;
        std::cout<<k<<" "<<l<<" here3\n";
        dox[j][5+j]=-ogq[qp][5+k][k]-ogq[qp][5+l][l];
        dvx[j][5+j]=-lgq[qp][5+k][k]-lgq[qp][5+l][l];*/
      //}else
        for(unsigned int i = 0; i < Nv; i++){
          dvx[j][i]=lgq[qp][i][j];
          dox[j][i]=ogq[qp][i][j];
        }
    
    ETA=eta[qp];
    for(unsigned int j = 0; j < dim; j++)
      if (std::fabs(n[j])>0.5) ETAg[j]=0.0;
      else ETAg[j]=etag[qp][j];
  }
  
  template <int dim>
  void MHDequations<dim>::noFlowBC(std::vector<Vector<double> > &lvq,    // lin values
                        std::vector<std::vector<Tensor<1,dim> > > &lgq,  // lin gradients
                        std::vector<Vector<double> > &ovq,               // old values
                        std::vector<std::vector<Tensor<1,dim> > > &ogq,  // old gradients
                        std::vector<Vector<double> > &/* viq */,               // initial values
                        std::vector<double > &eta,  // eta values
                        std::vector<Tensor<1,dim> > &etag, // eta gradients
                        const Tensor<1,dim> &n,                          // normal
                        const unsigned int qp)                           // qp
  {
    unsigned int n0=0;
    
    for(unsigned int j = 0; j < dim; j++) // find 
      if (std::fabs(n[j])>0.5) n0=j;
    
    // set state vector values
    for (unsigned int i = 0; i < Nv; i++){
      vl[i] = lvq[qp](i);
      vo[i] = ovq[qp](i);
    }
    double cf=0.1;
    vo[1]*=cf;
    vo[2]*=cf;
    vo[3]*=cf;
    
    // ... and for gradients
    for(unsigned int j = 0; j < dim; j++)
      for(unsigned int i = 0; i < Nv; i++){
        dvx[j][i]=lgq[qp][i][j];
        dox[j][i]=ogq[qp][i][j];
      }

    dvx[n0][1+n0]=dox[n0][1+n0]=0.0;  // velocity
    
    ETA=eta[qp];
    for(unsigned int j = 0; j < dim; j++)
      if (std::fabs(n[j])>0.5) ETAg[j]=0.0;
      else ETAg[j]=etag[qp][j];
}
  
  template <int dim>
  void MHDequations<dim>::mirrorBC(std::vector<Vector<double> > &lvq,    // lin values
                        std::vector<std::vector<Tensor<1,dim> > > &lgq,  // lin gradients
                        std::vector<Vector<double> > &ovq,               // old values
                        std::vector<std::vector<Tensor<1,dim> > > &ogq,  // old gradients
                        std::vector<Vector<double> > &/* viq */,               // initial values
                        std::vector<double > &eta,  // eta values
                        std::vector<Tensor<1,dim> > &etag, // eta gradients
                        const Tensor<1,dim> &n,                          // normal
                        const unsigned int qp)                           // qp
  {
    unsigned int c=0,d;
    // set state vector values
    for (unsigned int i = 0; i < Nv; i++){
      vl[i] = lvq[qp](i);
      vo[i] = ovq[qp](i);
    }
    // ... and for gradients
    for(unsigned int j = 0; j < dim; j++){
      if (std::fabs(n[j])>0.5)  c=j;
      for(unsigned int i = 0; i < Nv; i++){
        dvx[j][i]=lgq[qp][i][j];
        dox[j][i]=ogq[qp][i][j];
      }
    }
    d=(c+1)%dim;

    double cf=0.1;
    vl[1+c]*=cf;  // mirror velocity
    vo[1+c]*=cf;
    vl[4+d]*=cf;  // mirror B
    vo[4+d]*=cf;
    
    ETA=eta[qp];
    for(unsigned int j = 0; j < dim; j++)
      ETAg[j]=etag[qp][j];
    ETAg[c]=0.0;
  }
  
} // end of namespace mhd