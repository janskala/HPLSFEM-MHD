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
    double cf=0.99;
    vo[1]*=cf;
    vo[2]*=cf;
    vo[3]*=cf;
    // ... and for gradients
    for(unsigned int j = 0; j < dim; j++)
      if (std::fabs(n[j])>0.5){
        for(unsigned int i = 0; i < Nv; i++)
          dvx[j][i]=dox[j][i]=0.0;
        int k=(j+1)%3; // normal component of B is set as div B = 0
        int l=(k+1)%3;
        dox[j][5+j]=-ogq[qp][k][5+k]-ogq[qp][l][5+l];
        dvx[j][5+j]=-lgq[qp][k][5+k]-lgq[qp][l][5+l];
      }else
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
    unsigned int n0=0,dir,t1;//,t2;
    
    for(unsigned int j = 0; j < dim; j++) // find 
      if (std::fabs(n[j])>0.5) n0=j;
    dir=-1;//(n[n0]>.0?1:-1);
    t1=(3+n0+dir)%3;
    //t2=(3+t1+dir)%3;
    
    // set state vector values
    for (unsigned int i = 0; i < Nv; i++){
      vl[i] = lvq[qp](i);
      vo[i] = ovq[qp](i);
    }
    double cf=0.1;
    vo[1]*=cf;
    vo[2]*=cf;
    vo[3]*=cf;
    //vo[4+t1]*=cf;
    
    // ... and for gradients
    for(unsigned int j = 0; j < dim; j++)
      for(unsigned int i = 0; i < Nv; i++){
        dvx[j][i]=lgq[qp][i][j];
        dox[j][i]=ogq[qp][i][j];
      }

    dvx[n0][1+n0]=dox[n0][1+n0]=0.0;  // velocity
    //dvx[n0][1+t1]=dox[n0][1+t1]=0.0;
    //dvx[n0][1+t2]=dox[n0][1+t2]=0.0;
    //dvx[n0][4+t1]=dox[n0][4+t1]=0.0;  // tangent B
    //dvx[n0][4+t2]=dox[n0][4+t2]=0.0;
    
    // derivative of momentum
    /*for(unsigned int j = 0; j < dim; j++){
      dox[j][1+n0]=dvx[j][1+n0]=0.0;
      dox[j][1+t1]=dvx[j][1+t1]=0.0;
      dox[j][1+t2]=dvx[j][1+t2]=0.0;
    }*/
    // normal component of B is set as div B = 0
    //dox[n0][4+n0]=-ogq[qp][t1][4+t1]-ogq[qp][t2][4+t2];
    //dvx[n0][4+n0]=-lgq[qp][t1][4+t1]-lgq[qp][t2][4+t2];
    // normal derivative of tangent component of B is set to 0
    //dox[n0][4+t1]=dvx[n0][4+t1]=0.0;
    
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
    unsigned int c=0,d,e;
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
    dvx[c][1+c]=0.0; // momentum normal derivative is zero
    dox[c][1+c]=0.0;
    dvx[c][1+d]=0.0; // momentum normal derivative is zero
    dox[c][1+d]=0.0;
    dvx[c][1+e]=0.0; // momentum normal derivative is zero
    dox[c][1+e]=0.0;
    dvx[c][4+d]=0.0; // B tangent derivative is zero
    dox[c][4+d]=0.0;
    dvx[c][4+e]=0.0; // B tangent derivative is zero
    dox[c][4+e]=0.0;
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