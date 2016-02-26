#include "equations.h"
#include <ostream>

namespace mhd
{
  // RHS of MHD qeuations
  template <int dim>
  RightHandSide<dim>::RightHandSide() : Function<dim>(dim)
  {}

  template <int dim>
  inline
  void RightHandSide<dim>::vector_value(const Point<dim> &/*p*/,
                                         Vector<double>   &values) const
  {
    values(0) = 0.0;  // rho
    values(1) = 0.0;  // pi
    values(2) = 0.0;
    values(3) = 0.0;
    values(4) = 0.0;  // B
    values(5) = 0.0;  
    values(6) = 0.0;
    values(7) = 0.0;
    values(8) = 0.0;  // J
    values(9) = 0.0;
    values(10) = 0.0;
    values(11) = 0.0; // div B
  }


  template <int dim>
  void RightHandSide<dim>::vector_value_list(const std::vector<Point<dim> > &points,
                                              std::vector<Vector<double> >   &value_list) const
  {
    //Assert(value_list.size() == points.size(),
     //       ExcDimensionMismatch(value_list.size(), points.size()));

    const unsigned int n_points = points.size();
    for(unsigned int p=0; p<n_points; ++p)
      RightHandSide<dim>::vector_value(points[p], value_list[p]);
  }  
  
  template <int dim>
  MHDequations<dim>::MHDequations(Params &pars, MPI_Comm comm) : 
                                  mpi_communicator(comm)
  {
    dt=1e-6;
    vmax=0.0;
    
    for(unsigned int i=0;i<Ne;i++){
      Flx[0][i]=Flx[1][i]=Flx[2][i]=0.0;
      for(unsigned int j=0;j<Nv;j++){
        A[0][i][j]=A[1][i][j]=A[2][i][j]=0.0;
        B[i][j]=0.0;
      }
    }
    
    pars.setGravity(&gravity[0]);
    pars.setWeights(&weights[0]);
    pars.prm.enter_subsection("Simulation");
    {
      GAMMA=pars.prm.get_double("gamma");
      ETAmet=pars.prm.get_double("eta method");
      ETApar1=pars.prm.get_double("eta param 1");
      ETApar2=pars.prm.get_double("eta param 2");
    }
    pars.prm.leave_subsection();
    pars.prm.enter_subsection("Numerics");
    {
      theta=pars.prm.get_double("theta");
      CFL=pars.prm.get_double("CFL");
    }
    pars.prm.leave_subsection();
    
    NRLin=false;
    
    switch(ETAmet){
      case 0:
        setEta=&MHDequations::setEtaConst;
        break;
      case 1:
        setEta=&MHDequations::setEtaJ;
        break;
      case 2:
        setEta=&MHDequations::setEtaVD;
        break;
    }
    
    
    ETAg[0]=ETAg[1]=ETAg[2]=0.0;
    
    BCp[0]=&MHDequations::constantBC;
    BCp[1]=&MHDequations::freeBC;
    BCp[2]=&MHDequations::noFlowBC;
    BCp[3]=&MHDequations::mirrorBC;
  }
  
  template <int dim>
  MHDequations<dim>::~MHDequations()
  {
  }
  
  template <int dim>
  double* MHDequations<dim>::getWeights()
  {
    return &weights[0];
  }
  
  template <int dim>
  double MHDequations<dim>::getDt()
  {
    return dt;
  }
  
  template <int dim>
  void MHDequations<dim>::useNRLinearization(bool bl)
  {
    NRLin=bl;
  }
  
  template <int dim>
  void MHDequations<dim>::setFEvals(const FEFaceValues<dim> &fv,
                                    const unsigned int dofs,
                                    const unsigned int qp)
  {
    const FEValuesExtractors::Scalar u(0);
    
    for(unsigned int i=0;i<dofs/Nv;i++){
      unsigned int j=i*Nv;
      fev[i] = fv[u].value(j,qp);
      for(unsigned int d=0;d<dim;d++)
        feg[d][i] = fv[u].gradient(j,qp)[d];
    }
  }
  
  template <int dim>
  void MHDequations<dim>::setFEvals(const FEValues<dim> &fv,
                                    const unsigned int dofs,
                                    const unsigned int qp)
  {
    const FEValuesExtractors::Scalar u(0);
    
    for(unsigned int i=0;i<dofs/Nv;i++){
      unsigned int j=i*Nv;
      fev[i] = fv[u].value(j,qp);
      for(unsigned int d=0;d<dim;d++)
        feg[d][i] = fv[u].gradient(j,qp)[d];
    }
  }
  
  template <int dim>
  void MHDequations<dim>::set_state_vector_for_qp(std::vector<Vector<double> > &lvq,
                        std::vector<std::vector<Tensor<1,dim> > > &lgq,
                        std::vector<Vector<double> > &ovq,
                        std::vector<std::vector<Tensor<1,dim> > > &ogq,
                        std::vector<double > &eta,  // eta values
                        std::vector<Tensor<1,dim> > &etag, // eta gradients
                        const unsigned int qp)
  {
    // set state vector values
    for (unsigned int i = 0; i < Nv; i++){
      vl[i] = lvq[qp](i);
      vo[i] = ovq[qp](i);
    }
    // ... and for gradients
    for(unsigned int j = 0; j < dim; j++)
      for(unsigned int i = 0; i < Nv; i++){
        dvx[j][i]=lgq[qp][i][j];
        dox[j][i]=ogq[qp][i][j];
      }
    ETA=eta[qp];
    for(unsigned int j = 0; j < dim; j++)
      ETAg[j]=etag[qp][j];
  }
  
  template <int dim>
  void MHDequations<dim>::set_operator_matrixes(FullMatrix<double> *O,
                                      const unsigned int dofs)
  {
    double dtval;
    
    JacobiM(vl);
    if (NRLin) dxA(vl,dvx);
    
    for(unsigned int i=0;i<dofs/Nv;i++){
      double val = fev[i];
      dtval=theta*dt*val;
          
      // operator part: sum_i dA_i/dx_i
      if (NRLin){  // when cell is small enough then add: sum_i dA_i/dx_i
        for(unsigned int k=0;k<Nt;k++)
          for(unsigned int l=0;l<Nv;l++)
            O[i](k,l)=B[k][l]*dtval; 
      }else{  // the cell is big, clear it only
        for(unsigned int k=0;k<Nt;k++)
          for(unsigned int l=0;l<Nv;l++)
            O[i](k,l)=0.0;
      }
      for(unsigned int k=Nt;k<Ne;k++)
        for(unsigned int l=0;l<Nv;l++)
          O[i](k,l)=0.0;
      // diagonal part 1
      for(unsigned int l=Nt;l<Nv;l++) O[i](l,l)+=dt*val;
      for(unsigned int l=0;l<Nt;l++) O[i](l,l)+=val;
        
      // add gravity terms
      O[i](1,0)-=gravity[0]*dtval;
      O[i](2,0)-=gravity[1]*dtval;
      O[i](3,0)-=gravity[2]*dtval;
      O[i](7,1)-=gravity[0]*dtval;
      O[i](7,2)-=gravity[1]*dtval;
      O[i](7,3)-=gravity[2]*dtval;
      // operator part: sum_i A_i \phi_i/dx_i
      for(unsigned int d=0;d<dim;d++){
        double dtgrad = dt*feg[d][i];
        for(unsigned int k=Nt;k<Ne;k++)  // time independent part
          for(unsigned int l=0;l<Nv;l++)
            O[i](k,l)+=A[d][k][l]*dtgrad;
        dtgrad*=theta;
        for(unsigned int k=0;k<Nt;k++)   // time dependent part
          for(unsigned int l=0;l<Nv;l++)
            O[i](k,l)+=A[d][k][l]*dtgrad;
        
      }
    }
  }

  template <int dim>
  void MHDequations<dim>::set_rhs(Vector<double> &F)
  {
    double sum[Nt],sum2[Nt],dtth,dtoth;
    
    JacobiM(vo);
    
    for(unsigned int k=0;k<Nt;k++){
      F[k]=vo[k];
      sum[k]=sum2[k]=0.0;
    }
    for(unsigned int k=Nt;k<Ne;k++) F[k]=0.0;
      
    if (NRLin)
      for(unsigned int k=0;k<Nt;k++)
        for(unsigned int l=0;l<Nv;l++)
          sum2[k]+=B[k][l]*vl[l];
    
    for(unsigned int d=0;d<dim;d++)
      for(unsigned int k=0;k<Nt;k++)
        for(unsigned int l=0;l<Nv;l++)
          sum[k]+=A[d][k][l]*dox[d][l];
      
    dtth=-dt*theta;
    dtoth=dt*(1.0-theta);
    for(unsigned int k=0;k<Nt;k++)
      F[k]-=dtoth*sum[k]-dtth*sum2[k];
    
    // add gravity terms
    F[1]+=gravity[0]*vo[0]*dtoth;
    F[2]+=gravity[1]*vo[0]*dtoth;
    F[3]+=gravity[2]*vo[0]*dtoth;
    F[7]+=dtoth*(gravity[0]*vo[1]+gravity[1]*vo[2]+gravity[2]*vo[3]);
    
    // terms with eta derivative
    F[4]+=dtth*(vl[10]*ETAg[1]-vl[9]*ETAg[2])-dtoth*(vo[10]*ETAg[1]-vo[9]*ETAg[2]);
    F[5]+=dtth*(-vl[10]*ETAg[0]+vl[8]*ETAg[2])-dtoth*(-vo[10]*ETAg[0]+vo[8]*ETAg[2]);
    F[6]+=dtth*(vl[9]*ETAg[0]-vl[8]*ETAg[1])-dtoth*(vo[9]*ETAg[0]-vo[8]*ETAg[1]);
    F[7]+=2*dtth*(ETAg[0]*(vl[6]*vl[9]-vl[5]*vl[10])+ETAg[1]*(vl[4]*vl[10]-vl[6]*vl[8])+
                  ETAg[2]*(vl[5]*vl[8]-vl[4]*vl[9]))-2*dtoth*(
                  ETAg[0]*(vo[6]*vo[9]-vo[5]*vo[10])+ETAg[1]*(vo[4]*vo[10]-vo[6]*vo[8])+
                  ETAg[2]*(vo[5]*vo[8]-vo[4]*vo[9]));
  }
  
  template <int dim>
  bool MHDequations<dim>::checkOverflow(LA::MPI::Vector &v, LA::MPI::Vector &o)
  {
    double Uk,Um,buf,pc=1e-4/(GAMMA-1.0),rhs,rhc=1e-1;
    int overflow=0;
    std::pair<types::global_dof_index, types::global_dof_index> range=v.local_range();
    for(unsigned int i=range.first;i<range.second;i+=Nv){
      // check density   rh1 v.v = pi.pi/rh = rh1^2(v1^2+...)/rh2
      if (v[i+0]<rhc){
        //std::cout<<Uk<<" "<<v[i+0]<<" "<<v[i+1]<<" "<<v[i+2]<<"\n";
        //rhs=std::exp( std::log( std::sqrt(std::fabs(v[i+0])*rhc) ) ); // order average
        //if (rhs>rhc) rhs=rhc;
        //buf=0.5*std::sqrt(rhs/std::fabs(v[i+0]));
        //if (v[i+0]<0.0) buf*=-1.0;
        // change velocity in order to conserve total energy
        rhs=rhc+0.5*(o[i+0]-rhc);
        buf=rhs/o[i+0];
        v[i+0]=rhs;
        /*v[i+1]=o[i+1]*buf;
        v[i+2]=o[i+2]*buf;
        v[i+3]=o[i+3]*buf;*/
        
        //v[i+7]-=0.75*Uk;
        overflow=1;
      }
      rhs=1.0/v[i+0];
      if (v[i+1]*rhs>5.0) v[i+1]=5.0*v[i+0];
      if (v[i+1]*rhs<-5.0) v[i+1]=-5.0*v[i+0];
      if (v[i+2]*rhs>5.0) v[i+2]=5.0*v[i+0];
      if (v[i+2]*rhs<-5.0) v[i+2]=-5.0*v[i+0];
      if (v[i+3]*rhs>5.0) v[i+3]=5.0*v[i+0];
      if (v[i+3]*rhs<-5.0) v[i+3]=-5.0*v[i+0];
      
      // check pressure
      Uk=(v[i+1]*v[i+1]+v[i+2]*v[i+2]+v[i+3]*v[i+3])/(v[i+0]);
      Um=v[i+4]*v[i+4]+v[i+5]*v[i+5]+v[i+6]*v[i+6];
      buf=Um+Uk;
      if ((v[i+7]-buf)<pc){
        v[i+7]=buf+pc;
        overflow=1;
      }
    }
    
    overflow=Utilities::MPI::max(overflow,mpi_communicator);
    v.compress(VectorOperation::insert);  // write changes in parallel vector
    return overflow!=0;
  }
  
  template <int dim>
  void MHDequations<dim>::checkDt(LA::MPI::Vector &v)
  {
    double buf,p,a,vlc,cf,B2,iRh,iRh2;
    
    newdt=1e99;
    vmax=0.0;
    std::pair<types::global_dof_index, types::global_dof_index> range=v.local_range();
    for(unsigned int i=range.first;i<range.second;i+=Nv){
      iRh=1.0/v[i+0];
      iRh2=iRh*iRh;
      B2=v[i+4]*v[i+4]+v[i+5]*v[i+5]+v[i+6]*v[i+6];
      buf=(v[i+1]*v[i+1]+v[i+2]*v[i+2]+v[i+3]*v[i+3])*iRh+B2;
      
      p=GAMMA*(GAMMA-1.0)*(v[i+7]-buf); // gamma*p
      a=(p+B2)*iRh;
      cf=sqrt(0.5*(a+sqrt(a*a-4.0*p*v[i+4]*v[i+4]*iRh2)));
      if (v[i+1]>0.0) vlc=v[i+1]*iRh+cf;
      else vlc=v[i+1]*iRh-cf;
      buf=vlc*vlc;
      cf=sqrt(0.5*(a+sqrt(a*a-4.0*p*v[i+5]*v[i+5]*iRh2)));
      if (v[i+2]>0.0) vlc=v[i+2]*iRh+cf;
      else vlc=v[i+2]*iRh-cf;
      buf+=vlc*vlc;
      cf=sqrt(0.5*(a+sqrt(a*a-4.0*p*v[i+6]*v[i+6]*iRh2)));
      if (v[i+3]>0.0) vlc=v[i+3]*iRh+cf;
      else vlc=v[i+3]*iRh-cf;
      buf+=vlc*vlc;
      
      vlc=sqrt(buf); // fast magnetoacustic wave speed
      buf = CFL*hmin/vlc;
      if (buf<newdt) newdt = buf;  // use minimum dt
      buf = CFL*hmin*hmin/(2.0*ETAmax);
      if (buf<newdt) newdt = buf;  // use minimum dt for resistivity
      if (vlc>vmax) vmax=vlc;
    }

    newdt=Utilities::MPI::min(newdt,mpi_communicator);
    vmax=Utilities::MPI::max(vmax,mpi_communicator);

  }
  
  template <int dim>
  void MHDequations<dim>::setNewDt()
  {
    dt=newdt;
  }
  
  template <int dim>
  void MHDequations<dim>::setMinh(double minh)
  {
    //hmin=minh;
    hmin=Utilities::MPI::min(minh,mpi_communicator);
  }
  
  template <int dim>
  void MHDequations<dim>::setDt(double ndt)
  {
    dt=ndt;
  }

  template <int dim>
  double MHDequations<dim>::getVmax()
  {
    return vmax;
  }
  
  template <int dim>
  void MHDequations<dim>::fluxes(double v[])
  {
    double iRh, Uk,Um, p, E1, E2, E3;

    iRh = 1.0 / v[0];
    Uk = iRh*(v[1] * v[1] + v[2] * v[2] + v[3] * v[3]);
    Um = v[4] * v[4] + v[5] * v[5] + v[6] * v[6];
    p = (GAMMA-1.0)*(v[7] - Um - Uk);
    E1 = (v[3] * v[5] - v[2] * v[6])*iRh + ETA*v[8];
    E2 = (v[1] * v[6] - v[3] * v[4])*iRh + ETA*v[9];
    E3 = (v[2] * v[4] - v[1] * v[5])*iRh + ETA*v[10];

    switch(dim){
      case 3:
        Flx[2][0] = v[3];
        Flx[2][1] = v[3]*v[1]*iRh-v[6]*v[4];
        Flx[2][2] = v[3]*v[2]*iRh-v[6]*v[5];
        Flx[2][3] = v[3]*v[3]*iRh-v[6]*v[6]+0.5*(Um+p);
        Flx[2][4] = -E2;
        Flx[2][5] =  E1;
        //Flx[2][6] = 0;
        Flx[2][7] = (p*GAMMA/(GAMMA-1.0)+Uk)*v[3]*iRh + 2.0*(E1*v[5]-E2*v[4]);
      case 2:
        Flx[1][0] = v[2];
        Flx[1][1] = v[2]*v[1]*iRh-v[5]*v[4];
        Flx[1][2] = v[2]*v[2]*iRh-v[5]*v[5]+0.5*(Um+p);
        Flx[1][3] = v[2]*v[3]*iRh-v[5]*v[6];
        //Flx[1][4] = 0;
        Flx[1][5] =  E3;
        Flx[1][6] = -E1;
        Flx[1][7] = (p*GAMMA/(GAMMA-1.0)+Uk)*v[2]*iRh + 2.0*(E3*v[4]-E1*v[6]);
      case 1:
        Flx[0][0] = v[1];
        Flx[0][1] = v[1]*v[1]*iRh-v[4]*v[4]+0.5*(Um+p);
        Flx[0][2] = v[1]*v[2]*iRh-v[4]*v[5];
        Flx[0][3] = v[1]*v[3]*iRh-v[4]*v[6];
        //Flx[0][4] = 0;
        Flx[0][5] = -E3;
        Flx[0][6] =  E2;
        Flx[0][7] = (p*GAMMA/(GAMMA-1.0)+Uk)*v[1]*iRh + 2.0*(E2*v[6]-E3*v[5]);
    }
  }

} //end of namespace mhd