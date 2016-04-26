#include "equations.h"
#include <ostream>

namespace mhd
{
  // RHS of MHD qeuations
  template <int dim>
  MHDequations<dim>::RightHandSide::RightHandSide() : Function<dim>(dim)
  {}

  template <int dim>
  inline
  void MHDequations<dim>::RightHandSide::vector_value(const Point<dim> &/*p*/,
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
  void MHDequations<dim>::RightHandSide::vector_value_list(const std::vector<Point<dim> > &points,
                                              std::vector<Vector<double> >   &value_list) const
  {
    const unsigned int n_points = points.size();
    for(unsigned int p=0; p<n_points; ++p)
      this->vector_value(points[p], value_list[p]);
  }
  
  // Implementation of the body of MHDequations
  template <int dim>
  MHDequations<dim>::MHDequations(Params &pars, mapDoFs &stvdof, MPI_Comm comm) : 
                                  mpi_communicator(comm), stv2dof(stvdof)
  {
    newdt=dt=1e-6;
    vmax=0.0;
    
    for(unsigned int i=0;i<Ne;i++){
//       Flx[0][i]=Flx[1][i]=Flx[2][i]=0.0;
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
    
    for(unsigned int i=0;i<Nv;i++){
      fev[i] = new double[0];
      for(unsigned int d=0;d<3;d++)
        feg[d][i] = new double[0];
    }
    
    
    for(unsigned int i=0;i<5;i++)
      for(unsigned int j=0;j<DIRK.maxStageAll;j++){
        DIRK.tau[i][j]=0.0;
        for(unsigned int k=0;k<DIRK.maxStageAll+1;k++)
          DIRK.ab[i][k][j]=0.0;
    }
    // DIRK(1,2)
    DIRK.ab[0][0][0]=0.5;
    DIRK.ab[0][1][0]=1.0;
    DIRK.tau[0][0]=0.5;
    
    // DIRK(2,2)
    double alp=1.0-0.5*std::sqrt(2.0);
    DIRK.ab[1][0][0]=alp;
    DIRK.ab[1][1][0]=1.0-alp;
    DIRK.ab[1][1][1]=alp;
    DIRK.ab[1][2][0]=1.0-alp;
    DIRK.ab[1][2][1]=alp;
    DIRK.tau[1][0]=alp;
    DIRK.tau[1][1]=1.0;
    
    // DIRK(2,3)
    alp=1.0/(2.0*std::sqrt(3.0));
    DIRK.ab[2][0][0]=0.5+alp;
    DIRK.ab[2][1][0]=-2.0*alp;
    DIRK.ab[2][1][1]=0.5+alp;
    DIRK.ab[2][2][0]=0.5;
    DIRK.ab[2][2][1]=0.5;
    DIRK.tau[2][0]=0.5+alp;
    DIRK.tau[2][1]=0.5-alp;
    
    // DIRK(3,3)
    alp=0.5;  // or 1/6
    double t2=(1.0+alp)*0.5;
    double b1=-(6.0*alp*alp-16.0*alp+1.0)*0.25;
    double b2=(6.0*alp*alp-20.0*alp+5.0)*0.25;
    DIRK.ab[3][0][0]=alp;
    DIRK.ab[3][1][0]=t2-alp;
    DIRK.ab[3][1][1]=alp;
    DIRK.ab[3][2][0]=b1;
    DIRK.ab[3][2][1]=b2;
    DIRK.ab[3][2][2]=alp;
    DIRK.ab[3][3][0]=b1;
    DIRK.ab[3][3][1]=b2;
    DIRK.ab[3][3][2]=alp;
    DIRK.tau[3][0]=alp;
    DIRK.tau[3][1]=t2;
    DIRK.tau[3][2]=1.0;
    
    // DIRK(3,4)
    alp=2.0*std::cos(2.0*std::asin(1.0)/18.0);
    t2=(1.0+alp)*0.5;
    b1=(1.0+alp);
    DIRK.ab[4][0][0]=t2;
    DIRK.ab[4][1][0]=-alp*0.5;
    DIRK.ab[4][1][1]=t2;
    DIRK.ab[4][2][0]=b1;
    DIRK.ab[4][2][1]=-(1.0+2.0*alp);
    DIRK.ab[4][2][2]=t2;
    DIRK.ab[4][3][0]=1.0/(6.0*alp*alp);
    DIRK.ab[4][3][1]=1.0-1.0/(3.0*alp*alp);
    DIRK.ab[4][3][2]=1.0/(6.0*alp*alp);
    DIRK.tau[4][0]=t2;
    DIRK.tau[4][1]=0.5;
    DIRK.tau[4][2]=0.5*(1.0-alp);
  }
  
  template <int dim>
  MHDequations<dim>::~MHDequations()
  {
    for(unsigned int i=0;i<Nv;i++){
      delete [] fev[i];
      for(unsigned int d=0;d<3;d++)
        delete [] feg[d][i];;
    }
  }
  
  template <int dim>
  void MHDequations<dim>::reinitFEval()
  {
    for(unsigned int i=0;i<Nv;i++){
      delete [] fev[i];
      for(unsigned int d=0;d<dim;d++)
        delete [] feg[d][i];
    }
    
    for(unsigned int i=0;i<Nv;i++){
      fev[i] = new double[stv2dof.Nstv];
      for(unsigned int d=0;d<dim;d++)
        feg[d][i] = new double[stv2dof.Nstv];
    }
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
  void MHDequations<dim>::setCNMethod()
  {
    clcMat=&MHDequations::set_operators_full;
    clcRhs=&MHDequations::set_rhs_CN;
  }
  
  template <int dim>
  void MHDequations<dim>::setDIRKMethod(unsigned int m)
  {
    DIRK.method=m;
    DIRK.maxStage=DIRK.stages[m];
    DIRK.stage=0;
    clcMat=&MHDequations::set_operators_full;
    clcRhs=&MHDequations::set_rhs_DIRK;
  }
  
  template <int dim>
  void MHDequations<dim>::setDIRKStage(unsigned int s)
  {
    if (s>=DIRK.stages[DIRK.method]){
      DIRK.stage=DIRK.stages[DIRK.method]-1;
      theta=1.0;
      NRLinBck=NRLin;
      NRLin=false;    // In the last step we have pure explicit evaluation
      // set evaluation functions
      clcMat=&MHDequations::set_operators_diag;
    }else{
      DIRK.stage=s;
      theta=DIRK.ab[DIRK.method][s][s]*DIRK.tau[DIRK.method][s];
      if (s==0){  // set evaluation functions
        clcMat=&MHDequations::set_operators_full;
        NRLin=NRLinBck;
      }
    }
  }
  
  template <int dim>
  void MHDequations<dim>::setFEvals(const FEFaceValues<dim> &fv,
                                    const unsigned int qp)
  {
    for(unsigned int i=0;i<stv2dof.Nstv;i++)   // over the state vectors
      for(unsigned int j=0;j<Nv;j++){          // over the variables
        int dof=stv2dof.stateD[i][j];
        if (dof>=0){
          fev[j][i]=fv.shape_value_component(dof,qp,j);
          for(unsigned int d=0;d<dim;d++)
            if (std::fabs(fv.normal_vector(qp)[d])<0.5)
              feg[d][j][i] = fv.shape_grad_component(dof,qp,j)[d];
            else feg[d][j][i] = 0.0;
        }else{
          fev[j][i]=0.0;
          for(unsigned int d=0;d<dim;d++)
            feg[d][j][i] = 0.0;
        }
      }
  }
  
  template <int dim>
  void MHDequations<dim>::setFEvals(const FEValues<dim> &fv,
                                    const unsigned int qp)
  {
    for(unsigned int i=0;i<stv2dof.Nstv;i++)   // over the state vectors
      for(unsigned int j=0;j<Nv;j++){          // over the variables
        int dof=stv2dof.stateD[i][j];
        if (dof>=0){
          fev[j][i]=fv.shape_value_component(dof,qp,j);
          for(unsigned int d=0;d<dim;d++)
            feg[d][j][i] = fv.shape_grad_component(dof,qp,j)[d];
        }else{
          fev[j][i]=0.0;
          for(unsigned int d=0;d<dim;d++)
            feg[d][j][i] = 0.0;
        }
      }
  }
  
  template <int dim>
  void MHDequations<dim>::set_state_vector_for_qp(std::vector<Vector<double> >* Vqp[],
                        std::vector<std::vector<Tensor<1,dim> > >* Gqp[],
                        const unsigned int qp)
  {
    // set state vector values
    for(unsigned int k = 0; k < 2+DIRK.stage; k++)
      for (unsigned int i = 0; i < Nv; i++)
        V[k][i] = (*Vqp[k])[qp](i);
    
    // ... and for gradients
    for(unsigned int k = 0; k < 2+DIRK.stage; k++)
      for(unsigned int j = 0; j < dim; j++)
        for(unsigned int i = 0; i < Nv; i++)
          G[k][j][i]=(*Gqp[k])[qp][i][j];

    // checking underflow of density and pressure
    const double pc=1e-4/(GAMMA-1.0),rhc=1e-1;
    if (V[0][0]<rhc){  // check density
      V[0][0]=rhc;
    }
    if (V[1][0]<rhc){
      V[1][0]=rhc;
    }
    
    // check pressure
    double Uk,Um,buf;
    Uk=(V[0][1]*V[0][1]+V[0][2]*V[0][2]+V[0][3]*V[0][3])/(V[0][0]);
    Um=V[0][4]*V[0][4]+V[0][5]*V[0][5]+V[0][6]*V[0][6];
    buf=Um+Uk;
    if ((V[0][7]-buf)<pc){
      V[0][7]=buf+pc;
    }
    Uk=(V[1][1]*V[1][1]+V[1][2]*V[1][2]+V[1][3]*V[1][3])/(V[1][0]);
    Um=V[1][4]*V[1][4]+V[1][5]*V[1][5]+V[1][6]*V[1][6];
    buf=Um+Uk;
    if ((V[1][7]-buf)<pc){
      V[1][7]=buf+pc;
    }
    
    checkDt();
  }
  
  template <int dim>
  void MHDequations<dim>::calucate_matrix_rhs(FullMatrix<double> *O, Vector<double> &F)
  {
    (this->*clcMat)(O);
    (this->*clcRhs)(F);
  }
  
  template <int dim>
  void MHDequations<dim>::set_operators_full(FullMatrix<double> *O)
  {
    JacobiM(V[0]);
    if (NRLin) dxA(V[0],G[0]);
    
    for(unsigned int i=0;i<stv2dof.Nstv;i++){
      double thdt=theta*dt;
          
      // operator part: sum_i dA_i/dx_i
      if (NRLin){  // when cell is small enough then add: sum_i dA_i/dx_i
        for(unsigned int k=0;k<Nt;k++)
          for(unsigned int l=0;l<Nv;l++)
            O[i](k,l)=B[k][l]*thdt*fev[l][i];
      }else{  // the cell is big, clear it only
        for(unsigned int k=0;k<Nt;k++)
          for(unsigned int l=0;l<Nv;l++)
            O[i](k,l)=0.0;
      }
      for(unsigned int k=Nt;k<Ne;k++)
        for(unsigned int l=0;l<Nv;l++)
          O[i](k,l)=0.0;
      // diagonal part 1
      for(unsigned int l=Nt;l<Nv;l++) O[i](l,l)+=dt*fev[l][i];
      for(unsigned int l=0;l<Nt;l++) O[i](l,l)+=fev[l][i];
        
      // add gravity terms
      O[i](1,0)-=gravity[0]*thdt*fev[0][i];
      O[i](2,0)-=gravity[1]*thdt*fev[0][i];
      O[i](3,0)-=gravity[2]*thdt*fev[0][i];
      O[i](7,1)-=gravity[0]*thdt*fev[1][i];
      O[i](7,2)-=gravity[1]*thdt*fev[2][i];
      O[i](7,3)-=gravity[2]*thdt*fev[3][i];
      // operator part: sum_i A_i \phi_i/dx_i
      for(unsigned int d=0;d<dim;d++){
        for(unsigned int k=Nt;k<Ne;k++)  // time independent part
          for(unsigned int l=0;l<Nv;l++)
            O[i](k,l)+=A[d][k][l]*dt*feg[d][l][i];
          
        for(unsigned int k=0;k<Nt;k++)   // time dependent part
          for(unsigned int l=0;l<Nv;l++)
            O[i](k,l)+=A[d][k][l]*thdt*feg[d][l][i];
      }
    }
  }
  
  template <int dim>
  void MHDequations<dim>::set_operators_diag(FullMatrix<double> *O)
  {
    for(unsigned int i=0;i<stv2dof.Nstv;i++){

      for(unsigned int k=0;k<Ne;k++)
        for(unsigned int l=0;l<Nv;l++)
          O[i](k,l)=0.0;
      // diagonal part 1
      for(unsigned int l=Nt;l<Nv;l++) O[i](l,l)+=dt*fev[l][i];
      for(unsigned int l=0;l<Nt;l++) O[i](l,l)+=fev[l][i];

      for(unsigned int d=0;d<dim;d++)
        for(unsigned int k=Nt;k<Ne;k++)  // time independent part
          for(unsigned int l=0;l<Nv;l++)
            O[i](k,l)+=A[d][k][l]*dt*feg[d][l][i];
    }
  }

  template <int dim>
  void MHDequations<dim>::set_rhs_CN(Vector<double> &F)
  {
    double sum[Nt],sum2[Nt],dtth,dtoth;
    
    JacobiM(V[1]);
    
    for(unsigned int k=0;k<Nt;k++){
      F[k]=V[1][k];
      sum[k]=sum2[k]=0.0;
    }
    for(unsigned int k=Nt;k<Ne;k++) F[k]=0.0;
      
    if (NRLin)
      for(unsigned int k=0;k<Nt;k++)
        for(unsigned int l=0;l<Nv;l++)
          sum2[k]+=B[k][l]*V[0][l];
    
    for(unsigned int d=0;d<dim;d++)
      for(unsigned int k=0;k<Nt;k++)
        for(unsigned int l=0;l<Nv;l++)
          sum[k]+=A[d][k][l]*G[1][d][l];
      
    dtth=-dt*theta;
    dtoth=dt*(1.0-theta);
    for(unsigned int k=0;k<Nt;k++)
      F[k]-=dtoth*sum[k]+dtth*sum2[k];
    
    // add gravity terms
    F[1]+=gravity[0]*V[1][0]*dtoth;
    F[2]+=gravity[1]*V[1][0]*dtoth;
    F[3]+=gravity[2]*V[1][0]*dtoth;
    F[7]+=dtoth*(gravity[0]*V[1][1]+gravity[1]*V[1][2]+gravity[2]*V[1][3]);

    F[11]=dt*(this->*setEta)(1);  // calculate it from old time values
  }
  
    template <int dim>
  void MHDequations<dim>::set_rhs_DIRK(Vector<double> &F)
  {
    double sum[Nt],dtth;

    for(unsigned int k=0;k<Nt;k++){
      F[k]=V[1][k];
      sum[k]=0.0;
    }
    for(unsigned int k=Nt;k<Ne;k++) F[k]=0.0;
      
    if (NRLin)
      for(unsigned int k=0;k<Nt;k++)
        for(unsigned int l=0;l<Nv;l++)
          sum[k]+=B[k][l]*V[0][l];
    
    dtth=-dt*theta;
    for(unsigned int k=0;k<Nt;k++)
      F[k]-=dtth*sum[k];

    F[11]=dt*(this->*setEta)(1);
    
    for(unsigned int l=0;DIRK.stages[DIRK.stage]-1;l++){
      int lStage=1+DIRK.stage;
      JacobiM(V[lStage]);
      
      for(unsigned int k=0;k<Nt;k++) sum[k]=0.0;
        
      for(unsigned int d=0;d<dim;d++)
        for(unsigned int k=0;k<Nt;k++)
          for(unsigned int l=0;l<Nv;l++)
            sum[k]+=A[d][k][l]*G[lStage][d][l];
          
      for(unsigned int k=0;k<Nt;k++)
        F[k]-=dt*DIRK.ab[DIRK.method][DIRK.stage][l]*sum[k];
    }
  }
  
  template <int dim>
  bool MHDequations<dim>::checkOverflow(LA::MPI::Vector &v, LA::MPI::Vector &o)
  {
    /*double Uk,Um,buf,pc=1e-4/(GAMMA-1.0),rhs,rhc=1e-1;
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
        //buf=rhs/o[i+0];
        v[i+0]=rhs;
        buf=0.1;
        v[i+1]=buf*v[i+1]+(1.-buf)*o[i+1];
        v[i+2]=buf*v[i+2]+(1.-buf)*o[i+2];
        v[i+3]=buf*v[i+3]+(1.-buf)*o[i+3];
        
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
    return overflow!=0;*/
    return false;
  }
  
  template <int dim>
  void MHDequations<dim>::checkDt(LA::MPI::Vector &v)
  {
    /*double buf,p,a,vlc,cf,B2,iRh,iRh2;
    
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
*/
  }
  
  template <int dim>
  void MHDequations<dim>::checkDt()
  {
    double buf,p,a,vlc,cf,B2,iRh,iRh2;
    
    newdt=1e99;
    vmax=0.0;
    
    iRh=1.0/V[0][0];
    iRh2=iRh*iRh;
    B2=V[0][4]*V[0][4]+V[0][5]*V[0][5]+V[0][6]*V[0][6];
    buf=(V[0][1]*V[0][1]+V[0][2]*V[0][2]+V[0][3]*V[0][3])*iRh+B2;
    
    p=GAMMA*(GAMMA-1.0)*(V[0][7]-buf); // gamma*p
    a=(p+B2)*iRh;
    cf=sqrt(0.5*(a+sqrt(a*a-4.0*p*V[0][4]*V[0][4]*iRh2)));
    if (V[0][1]>0.0) vlc=V[0][1]*iRh+cf;
    else vlc=V[0][1]*iRh-cf;
    buf=vlc*vlc;
    cf=sqrt(0.5*(a+sqrt(a*a-4.0*p*V[0][5]*V[0][5]*iRh2)));
    if (V[0][2]>0.0) vlc=V[0][2]*iRh+cf;
    else vlc=V[0][2]*iRh-cf;
    buf+=vlc*vlc;
    cf=sqrt(0.5*(a+sqrt(a*a-4.0*p*V[0][6]*V[0][6]*iRh2)));
    if (V[0][3]>0.0) vlc=V[0][3]*iRh+cf;
    else vlc=V[0][3]*iRh-cf;
    buf+=vlc*vlc;
    
    vlc=sqrt(buf); // fast magnetoacustic wave speed
    buf = CFL*hmin/vlc;
    if (buf<newdt) newdt = buf;  // use minimum dt
    buf = CFL*hmin*hmin/(8.0*ETAmax);
    if (buf<newdt) newdt = buf;  // use minimum dt for resistivity
    if (vlc>vmax) vmax=vlc;

  }
  
  template <int dim>
  void MHDequations<dim>::setNewDt()
  {
    newdt=Utilities::MPI::min(newdt,mpi_communicator);
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
    vmax=Utilities::MPI::max(vmax,mpi_communicator);
    return vmax;
  }
  
//   template <int dim>
//   void MHDequations<dim>::fluxes(double v[])
//   {
//     double iRh, Uk,Um, p, E1, E2, E3;
// 
//     iRh = 1.0 / v[0];
//     Uk = iRh*(v[1] * v[1] + v[2] * v[2] + v[3] * v[3]);
//     Um = v[4] * v[4] + v[5] * v[5] + v[6] * v[6];
//     p = (GAMMA-1.0)*(v[7] - Um - Uk);
//     E1 = (v[3] * v[5] - v[2] * v[6])*iRh + ETA*v[8];
//     E2 = (v[1] * v[6] - v[3] * v[4])*iRh + ETA*v[9];
//     E3 = (v[2] * v[4] - v[1] * v[5])*iRh + ETA*v[10];
// 
//     switch(dim){
//       case 3:
//         Flx[2][0] = v[3];
//         Flx[2][1] = v[3]*v[1]*iRh-v[6]*v[4];
//         Flx[2][2] = v[3]*v[2]*iRh-v[6]*v[5];
//         Flx[2][3] = v[3]*v[3]*iRh-v[6]*v[6]+0.5*(Um+p);
//         Flx[2][4] = -E2;
//         Flx[2][5] =  E1;
//         //Flx[2][6] = 0;
//         Flx[2][7] = (p*GAMMA/(GAMMA-1.0)+Uk)*v[3]*iRh + 2.0*(E1*v[5]-E2*v[4]);
//       case 2:
//         Flx[1][0] = v[2];
//         Flx[1][1] = v[2]*v[1]*iRh-v[5]*v[4];
//         Flx[1][2] = v[2]*v[2]*iRh-v[5]*v[5]+0.5*(Um+p);
//         Flx[1][3] = v[2]*v[3]*iRh-v[5]*v[6];
//         //Flx[1][4] = 0;
//         Flx[1][5] =  E3;
//         Flx[1][6] = -E1;
//         Flx[1][7] = (p*GAMMA/(GAMMA-1.0)+Uk)*v[2]*iRh + 2.0*(E3*v[4]-E1*v[6]);
//       case 1:
//         Flx[0][0] = v[1];
//         Flx[0][1] = v[1]*v[1]*iRh-v[4]*v[4]+0.5*(Um+p);
//         Flx[0][2] = v[1]*v[2]*iRh-v[4]*v[5];
//         Flx[0][3] = v[1]*v[3]*iRh-v[4]*v[6];
//         //Flx[0][4] = 0;
//         Flx[0][5] = -E3;
//         Flx[0][6] =  E2;
//         Flx[0][7] = (p*GAMMA/(GAMMA-1.0)+Uk)*v[1]*iRh + 2.0*(E2*v[6]-E3*v[5]);
//     }
//   }

} //end of namespace mhd