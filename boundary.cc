#include "equations.h"

namespace mhd
{
  
  template <int dim>
  void MHDequations<dim>::constantBC(std::vector<Vector<double> >* Vqp[],
                        std::vector<std::vector<Tensor<1,dim> > >* /*Gqp*/[],
                        std::vector<Vector<double> > &/*viq*/,               // initial values
                        const Tensor<1,dim> &/*n*/,                         // normal
                        const Point<dim>&/* pt */,                          // point
                        const unsigned int qp)                              // qp
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
                        const Point<dim>&/* pt */,                       // point
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
                        const Point<dim>&/* pt */,                       // point
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
                        const Point<dim>&/* pt */,                       // point
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
  
    template <int dim>
  void MHDequations<dim>::vortexBC(std::vector<Vector<double> >* Vqp[],
                        std::vector<std::vector<Tensor<1,dim> > >* Gqp[],
                        std::vector<Vector<double> > &/* viq */,         // initial values
                        const Tensor<1,dim> &n,                          // normal
                        const Point<dim> &pt,                             // point
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
    
  //----------- Geometry

  //--- Torus: the following parameters MUST be the same as in init.cc!

  // Torus major radius
  static const double R_t=4.0;

  // Submerging of torus main axis in units of R_t
  static const double d2R_t=0.5;
  
  double dx=0.05;

  //---- Foot-point motions parameters

  // magnitude of driving-flow angular speed
  static const double omega_0=0.05;

  // Projection factor [ epsilon=cos(alpha) where alpha is declination 
  // of the torus minor axis from vertical direction]
  double epsilon=sqrt(1.0-d2R_t*d2R_t);

  // Pre-calculation of derived factors for computation speed-up
  double epsilon2=epsilon*epsilon;
  double invEpsilon=1.0/epsilon;

  // Foot-Point Center(s) [0, +/-y_fpc, 0]
  double y_fpc=R_t*epsilon;
  
  // Driver-decay scale and gradient
  double ddScale=2.0*dx;
  double ddGrad=1.0/ddScale;

  // dynamics of driving vortex flow
  double omega=omega_0*0.5*(1.0+tanh(0.2*(*time-20.0)));

  double xx,yy,rfp1,rfp2,vfp1,vfp2;
  
  xx=pt[0]-((*boxP2)[0]+(*boxP1)[0])*0.5;
  yy=pt[1]-((*boxP2)[1]+(*boxP1)[1])*0.5;


  rfp1=sqrt(epsilon2*(yy-y_fpc)*(yy-y_fpc)+xx*xx);
  rfp2=sqrt(epsilon2*(yy+y_fpc)*(yy+y_fpc)+xx*xx);

  vfp1=0.5*(1.0+tanh(ddGrad*(1.0-rfp1)));
  vfp2=0.5*(1.0+tanh(ddGrad*(1.0-rfp2)));

  /*

    barta@asu.cas.cz
    09/02/2012
    
    The driving is intended to be by specific velocity profile, therefore
    it has to be scaled by the density before the momentum is set.
    This change has been implemented now.


    N.B.: Only internal cells are set here, the extension is given by
    the standard, non-specific BC (e.g. von Neumann)

  */

  V[0][1]=-omega*epsilon*((yy-y_fpc)*vfp1+(yy+y_fpc)*vfp2)*V[0][0];
  V[0][2]=omega*invEpsilon*(xx*vfp1+xx*vfp2)*V[0][0];
  V[0][3]=0.0;

    
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
  
} // end of namespace mhd