#include "equations.h"

namespace mhd
{
  
  template <int dim>
  void MHDequations<dim>::constantBC(std::vector<Vector<double> > &/* lvq */,  // lin values
                        std::vector<std::vector<Tensor<1,dim> > > &/* lgq */,  // lin gradients
                        std::vector<Vector<double> > &/* ovq*/ ,               // old values
                        std::vector<std::vector<Tensor<1,dim> > > &/*ogq */,  // old gradients
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
    // ... and for gradients
    for(unsigned int j = 0; j < dim; j++)
      if (std::fabs(n[j])>0.5)
        for(unsigned int i = 0; i < Nv; i++)
          dvx[j][i]=dox[j][i]=0.0;
      else
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
    // set state vector values
    vl[0] = lvq[qp](0);
    vo[0] = ovq[qp](0);
    for (unsigned int i = 1; i < 3; i++){
      vl[i] = 0.0;
      vo[i] = 0.0;
    }
    for (unsigned int i = 3; i < Nv; i++){
      vl[i] = lvq[qp](i);
      vo[i] = ovq[qp](i);
    }
    // ... and for gradients
    for(unsigned int j = 0; j < dim; j++)
      if (std::fabs(n[j])>0.5)
        for(unsigned int i = 0; i < Nv; i++){
          dvx[j][i]=0.0;
          dox[j][i]=0.0;
        }
      else
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
    unsigned int c,d,e;
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
    d=(c+1)%3;
    e=(d+1)%3;
      
    vl[1+c]=0.0;  // mirror velocity
    vo[1+c]=0.0;
    vl[4+d]=0.0;  // mirror B
    vo[4+d]=0.0;
    
    dvx[c][0]=0.0; // density normal derivative is zero
    dox[c][0]=0.0;
    dvx[c][7]=0.0; // energy normal derivative is zero
    dox[c][7]=0.0;
    dvx[c][8+e]=0.0;  // mirror gradient of J
    dox[c][8+e]=0.0;
    
    ETA=eta[qp];
    for(unsigned int j = 0; j < dim; j++)
      ETAg[j]=etag[qp][j];
    ETAg[c]=0.0;
  }
  
} // end of namespace mhd