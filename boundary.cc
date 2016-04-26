#include "equations.h"

namespace mhd
{
  
  template <int dim>
  void MHDequations<dim>::constantBC(std::vector<Vector<double> >* Vqp[],
                        std::vector<std::vector<Tensor<1,dim> > >* /*Gqp*/[],
                        std::vector<Vector<double> > &/*viq*/,               // initial values
                        const Tensor<1,dim> &/*n*/,                         // normal
                        const unsigned int qp)                           // qp
  {
    // set state vector values
    for(int k = 0; k <= 2+DIRK.stage; k++)
      for (unsigned int i = 0; i < Nv; i++)
        V[k][i] = (*Vqp[k])[qp](i);
      
    // ... and for gradients
    for(int k = 0; k <= 2+DIRK.stage; k++)
      for(unsigned int j = 0; j < dim; j++)
        for(unsigned int i = 0; i < Nv; i++)
          G[k][j][i]=0.0;
    
  }
  
  template <int dim>
  void MHDequations<dim>::freeBC(std::vector<Vector<double> >* Vqp[],
                        std::vector<std::vector<Tensor<1,dim> > >* Gqp[],
                        std::vector<Vector<double> > &/* viq */,         // initial values
                        const Tensor<1,dim> &n,                          // normal
                        const unsigned int qp)                           // qp
  {
    unsigned int n0=0;
    
    for(unsigned int j = 0; j < dim; j++) // find 
      if (std::fabs(n[j])>0.5) n0=j;
      
    // set state vector values
    for(int k = 0; k <= 2+DIRK.stage; k++)
      for (unsigned int i = 0; i < Nv; i++)
        V[k][i] = (*Vqp[k])[qp](i);
    
    double cf=0.9;
    V[0][1]*=cf;
    V[0][2]*=cf;
    V[0][3]*=cf;
    cf=0.8;
    V[1][1]*=cf;
    V[1][2]*=cf;
    V[1][3]*=cf;
    for(int k = 2; k <= 2+DIRK.stage; k++){
      V[1][1]*=cf;
      V[1][2]*=cf;
      V[1][3]*=cf;
    }
    // ... and for gradients
    for(int k = 0; k <= 2+DIRK.stage; k++)
      for(unsigned int j = 0; j < dim; j++)
        for(unsigned int i = 0; i < Nv; i++)
          G[k][j][i]=(*Gqp[k])[qp][i][j];
        
    for(int k = 0; k <= 2+DIRK.stage; k++){
     G[k][n0][1]=0.0;  // velocity
     G[k][n0][2]=0.0;
     G[k][n0][3]=0.0;
    }
  }
  
  template <int dim>
  void MHDequations<dim>::noFlowBC(std::vector<Vector<double> >* Vqp[],
                        std::vector<std::vector<Tensor<1,dim> > >* Gqp[],
                        std::vector<Vector<double> > &/* viq */,         // initial values
                        const Tensor<1,dim> &n,                          // normal
                        const unsigned int qp)                           // qp
  {
    unsigned int n0=0;
    
    for(unsigned int j = 0; j < dim; j++) // find 
      if (std::fabs(n[j])>0.5){
        n0=j;
        break;
      }
    
    // set state vector values
    for(int k = 0; k <= 2+DIRK.stage; k++)
      for (unsigned int i = 0; i < Nv; i++)
        V[k][i] = (*Vqp[k])[qp](i);
    
    double cf=0.1;
    V[0][1]*=cf;
    V[0][2]*=cf;
    V[0][3]*=cf;
    
    // ... and for gradients
    for(int k = 0; k <= 2+DIRK.stage; k++)
      for(unsigned int j = 0; j < dim; j++)
        for(unsigned int i = 0; i < Nv; i++)
          G[k][j][i]=(*Gqp[k])[qp][i][j];

    for(int k = 0; k <= 2+DIRK.stage; k++){
     G[k][n0][1]=0.0;  // velocity
     G[k][n0][2]=0.0;
     G[k][n0][3]=0.0;
    }
}
  
  template <int dim>
  void MHDequations<dim>::mirrorBC(std::vector<Vector<double> >* Vqp[],
                        std::vector<std::vector<Tensor<1,dim> > >* Gqp[],
                        std::vector<Vector<double> > &/* viq */,               // initial values
                        const Tensor<1,dim> &n,                          // normal
                        const unsigned int qp)                           // qp
  {
    unsigned int c=0,d;
    // set state vector values
    for(int k = 0; k <= 2+DIRK.stage; k++)
      for (unsigned int i = 0; i < Nv; i++)
        V[k][i] = (*Vqp[k])[qp](i);
    
    // ... and for gradients
    for(int k = 0; k <= 2+DIRK.stage; k++)
      for(unsigned int j = 0; j < dim; j++)
        for(unsigned int i = 0; i < Nv; i++)
          G[k][j][i]=(*Gqp[k])[qp][i][j];
    
    for(unsigned int j = 0; j < dim; j++)
      if (std::fabs(n[j])>0.5){  
        c=j;
        break;
      }
      
    d=(c+1)%dim;

    double cf=0.1;
    for(int k = 0; k <= 2+DIRK.stage; k++){
      V[k][1+c]*=cf;  // mirror velocity
      V[k][4+d]*=cf;  // mirror B
    }
  }
  
} // end of namespace mhd