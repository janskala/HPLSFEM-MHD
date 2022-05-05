#include "initcon.h"
#include "problem.h"
//--- GNU Scientific library: Elliptical integrals
//#include "gsl/gsl_sf_ellint.h"
#include"complete_elliptic_integrals.h"

namespace mhd
{
  /***************************************************************************
          Initial Condition
  ***************************************************************************/
  template <int dim>
  InitialValues<dim>::InitialValues(Params &pars)
  {
    pars.prm.enter_subsection("Simulation");
    {
      GAMMA=pars.prm.get_double("gamma");
    }
    pars.prm.leave_subsection();
  }
  
  template <int dim>
  void InitialValues<dim>::vector_value_list(const std::vector<Point<dim> > &points,
                                              std::vector<Vector<double> >   &value_list) const
  {
    for(unsigned int p=0; p<points.size(); ++p)
      this->point_value(points[p], value_list[p]);
  }
  
  
  /***************************************************************************
          MHD Blast initial condition
  ***************************************************************************/
  template <int dim>
  mhdBlast<dim>::mhdBlast(Params &pars):InitialValues<dim>(pars)
  {
    pars.prm.enter_subsection("Simulation");
    {
      pars.prm.enter_subsection("Box");
      {
        box[0][0]=pars.prm.get_double("x_min");
        box[0][1]=pars.prm.get_double("y_min");
        if (dim==3) box[0][2]=pars.prm.get_double("z_min");
        box[1][0]=pars.prm.get_double("x_max");
        box[1][1]=pars.prm.get_double("y_max");
        if (dim==3) box[1][2]=pars.prm.get_double("z_max");
      }
      pars.prm.leave_subsection();
    }
    pars.prm.leave_subsection();
  }
  
  
  template <int dim>
  void mhdBlast<dim>::point_value(const Point<dim> &p,
                                        Vector<double> &v) const
  {
    if (p.norm()<0.1){ // 0.1
        v(7)=10.0/(this->GAMMA-1.0)+1.0; // U
    }else{
        v(7)=0.1/(this->GAMMA-1.0)+1.0; // U
    } 
    v(0)=1.0; // rho
    v(1)=0.0; // v
    v(2)=0.0;
    v(3)=0.0;
    v(4)=1.0/sqrt(2); // B
    v(5)=1.0/sqrt(2);
    v(6)=0.0;
    
    v(8)=0.0;  // J
    v(9)=0.0;
    v(10)=0.0;
    
    v(11)=0.0; // eta
  }
  
  /***************************************************************************
          Harris Current Sheet initial condition
  ***************************************************************************/
  template <int dim>
  harris<dim>::harris(Params &pars):InitialValues<dim>(pars)
  {
    pars.prm.enter_subsection("Simulation");
    {      
      pars.prm.enter_subsection("Box");
      {
        box[0][0]=pars.prm.get_double("x_min");
        box[0][1]=pars.prm.get_double("y_min");
        if (dim==3) box[0][2]=pars.prm.get_double("z_min");
        box[1][0]=pars.prm.get_double("x_max");
        box[1][1]=pars.prm.get_double("y_max");
        if (dim==3) box[1][2]=pars.prm.get_double("z_max");
      }
      pars.prm.leave_subsection();
    }
    pars.prm.leave_subsection();
  }
  
  template <int dim>
  void harris<dim>::point_value(const Point<dim> &p,
                                        Vector<double> &v) const
  {
    double yy,xx;
    double pressure;
    
    xx=p[0]-(box[1][0]+box[0][0])*0.5;
    yy=p[1]-(box[1][1]+box[0][1])*0.5;
    
    v(0)=1.0; // rho
    v(1)=0.0;//-.08*p[0]*p[0]*p[0]*exp(-p[0]*p[0]/12.0)*exp(-(p[1]-20)*(p[1]-20)/8.0); // v
    v(2)=0.0;
    v(3)=0.0;
    v(4)=4e-2*yy*exp(-(xx*xx+yy*yy)/8.0); // B
    v(5)=std::tanh(xx)-4e-2*xx*exp(-(xx*xx+yy*yy)/8.0);
    v(6)=0.0;
    pressure=0.05+1.0-v(5)*v(5);
    v(7)=pressure/(this->GAMMA-1.0)+v(5)*v(5)+v(1)*v(1); // U
    v(8)=0.0;  // J
    v(9)=0.0;
    v(10)=1.0/(std::cosh(xx)*std::cosh(xx));
    
    v(11)=0.0; // eta
  }
  
  /***************************************************************************
          Debugging initial condition
  ***************************************************************************/
  template <int dim>
  debug<dim>::debug(Params &pars):InitialValues<dim>(pars)
  {
  }
  
  template <int dim>
  void debug<dim>::point_value(const Point<dim> &/*p*/,
                                        Vector<double> &v) const
  {
    v(0)=1.0;
    v(1)=2.0;
    v(2)=3.0;
    v(3)=4.0;
    v(4)=5.0;
    v(5)=6.0;
    v(6)=7.0;
    v(7)=8.0/(this->GAMMA-1.0)+v(1)*v(1)+v(2)*v(2)+v(3)*v(3)
                        +v(4)*v(4)+v(5)*v(5)+v(6)*v(6);
    v(8)=9.0;
    v(9)=10.0;
    v(10)=11.0;
    
    v(11)=0.0; // eta
  }
  
  /***************************************************************************
          Calculate the field according to TD paper (A&A 351, 707, 1999)
          Fill the structure with gravity-stratified plasma. 
  ***************************************************************************/
  template <int dim>
  TitovDemoulin<dim>::TitovDemoulin(Params &pars):InitialValues<dim>(pars)
  {
    pars.prm.enter_subsection("Simulation");
    {      
      pars.prm.enter_subsection("Box");
      {
        box[0][0]=pars.prm.get_double("x_min");
        box[0][1]=pars.prm.get_double("y_min");
        if (dim==3) box[0][2]=pars.prm.get_double("z_min");
        box[1][0]=pars.prm.get_double("x_max");
        box[1][1]=pars.prm.get_double("y_max");
        if (dim==3) box[1][2]=pars.prm.get_double("z_max");
      }
      pars.prm.leave_subsection();
    }
    pars.prm.leave_subsection();
    
    // plasma beta
    beta=0.05;
    Lg=0.0;          

    invLg=0.0;
    if(Lg > 0.0) invLg=1.0/Lg;
    
    //======================== TD-model specific parameters

    // Torus winding number
    N_t=1.0;

    // Torus major radius
    R_t=4.0;
  
    // Submerging of torus main axis in units of R_t
    d2R_t=2.0/R_t;

    // Distance of magnetic charges from x=0 plane in units of R_t
    L2R_t=2.0/R_t;
      
    
    //======================= Calculate dependent TD model parameters

    // Normalised magnetic charge corresponding to global equilibrium (Eq. 6)
    q_mag=0.25*fabs(N_t)*(log(8.0*R_t)-1.25)
      *(1+L2R_t*L2R_t)*sqrt(1+L2R_t*L2R_t)/L2R_t;

    // Sign of winding: corresponds to sign of I_O in TD paper
    iSgn=(N_t>=0)?1.0:-1.0;

    // "Helicity" factor inside tho loop (used later in B_theta_internal calcs)
    heliFactor=2.0*(N_t*N_t)/(R_t*R_t);


    //======================= Parameters for plasma

    // The coronal/prominence temperature ratio and its inverse value.
    Tc2Tp=1.0;
    //  Tp2Tc=1.0/Tc2Tp;

    //======================= TCPR definition

    // density jump half-width...
    t_rho=0.12;
    
    // ...and its inverse value
    densGrad=1.0/t_rho;
  }
  
  template <int dim>
  void TitovDemoulin<dim>::point_value(const Point<dim> & p,
                                        Vector<double> &v) const
  {
    //========== Calculate the vector potential for I_t-generated toroidal field 
    double xx,yy,zz;
    
    xx=p[0]-(box[1][0]+box[0][0])*0.5;
    yy=p[1]-(box[1][1]+box[0][1])*0.5;
    zz=p[2]-box[0][2];

    Vector<double> theta0(3);
    double r_maj,r_min;
    const double dd=1e-8;  // precision of numerical derivatives
    const double idd=0.5/dd;
    double df[6][3];
    double P[6][3];
    double dP[3][3];
    
    // calculate vector potentil in 6 close points (+dx,-dx,+dy,-dy,+dz,-dz)
    for(unsigned int i=0;i<6;++i){
      df[i][0]=df[i][1]=df[i][2]=0.0;
      df[i][int(i/2)]=double(1-2*int(i%2))*dd;
    }
    
    for(unsigned int i=0;i<6;++i){
      double x=xx+df[i][0];
      double y=yy+df[i][1];
      double z=zz+df[i][2];
      
      // Distances from torus major and minor axes
      r_maj=sqrt(y*y+(z+d2R_t*R_t)*(z+d2R_t*R_t));
      r_min=sqrt(x*x+(r_maj-R_t)*(r_maj-R_t));

      // Unit vector of toroidal coordinate theta
      theta0[0]=0.0;
      theta0[1]=-(z+d2R_t*R_t)/r_maj;
      theta0[2]=y/r_maj;

      // Common radial factor for A_tor
      double rFactor=fabs(N_t)*sqrt(1.0/(R_t*r_maj));

      // Argument of elliptical integral
      double kr=2.0*sqrt(r_maj*R_t/((r_maj+R_t)*(r_maj+R_t)+x*x));

      //---- Sew-up internal and external solutions
      if(r_min>1.0){ //---- external region 
          // Elliptical integrals
          double Ek,Kk;
          Complete_Elliptic_Integrals_Modulus(kr, Kk, Ek);

          double Ak=((2.0-kr*kr)*Kk-2.0*Ek)/kr;

          P[i][0]=(rFactor*Ak)*theta0[0];
          P[i][1]=(rFactor*Ak)*theta0[1];
          P[i][2]=(rFactor*Ak)*theta0[2];
          
      }else{ //---- inside the torus

          // ka=kr at r_min=1 (=torus surface) 
          double ka=2.0*sqrt(r_maj*R_t/(4.0*r_maj*R_t+1.0));

          double Ek,Kk;
          Complete_Elliptic_Integrals_Modulus(ka, Kk, Ek);

          double Ak=((2.0-ka*ka)*Kk-2.0*Ek)/ka;
          double Ak_prime=((2.0-ka*ka)*Ek-2.0*(1.0-ka*ka)*Kk)/
                          (ka*ka*(1.0-ka*ka));

          double cf=(rFactor*(Ak+Ak_prime*(kr-ka)));
          P[i][0]=cf*theta0[0];
          P[i][1]=cf*theta0[1];
          P[i][2]=cf*theta0[2];
      }
    }
    
    // calculate derivatives of vector potential
    for(unsigned int i=0;i<3;++i){
      dP[i][0]=(P[2*i][0]-P[2*i+1][0])*idd;
      dP[i][1]=(P[2*i][1]-P[2*i+1][1])*idd;
      dP[i][2]=(P[2*i][2]-P[2*i+1][2])*idd;
    }
    
    //====================== Calculate the full state field

    double pressure;
    
    // Distances from torus major and minor axes
    r_maj=sqrt(yy*yy+(zz+d2R_t*R_t)*(zz+d2R_t*R_t));
    r_min=sqrt(xx*xx+(r_maj-R_t)*(r_maj-R_t));

    // Unit vector of toroidal coordinate theta
    theta0[0]=0.0;
    theta0[1]=-(zz+d2R_t*R_t)/r_maj;
    theta0[2]=yy/r_maj;

    // Radius vectors originating in magnetic charges
    Vector<double> r_plus(3);//(xx-L2R_t*R_t,yy,zz+d2R_t*R_t);
    r_plus[0]=xx-L2R_t*R_t;
    r_plus[1]=yy;
    r_plus[2]=zz+d2R_t*R_t;
    Vector<double> r_minus(3);//(xx+L2R_t*R_t,yy,zz+d2R_t*R_t);
    r_minus[0]=xx+L2R_t*R_t;
    r_minus[1]=yy;
    r_minus[2]=zz+d2R_t*R_t;

    double rp=r_plus.l2_norm();
    double rm=r_minus.l2_norm();

    //========== Calculate the magnetic field in TD equilibrium by parts

    //--- Q-generated part of field

    double cf1=q_mag/(rp*rp*rp);
    double cf2=q_mag/(rm*rm*rm);
    Vector<double> B_loc(3);//=cf1*r_plus-cf2*r_minus;
    B_loc.add(cf1,r_plus,-cf2,r_minus);
 
    // add vector potential part B = curl A
    B_loc[0]+=dP[1][2]-dP[2][1];
    B_loc[1]+=dP[2][0]-dP[0][2];
    B_loc[2]+=dP[0][1]-dP[1][0];
    
    /*
      barta@asu.cas.cz
      10/02/2012

      With the following density-scaling the velocities are normalised 
      to V_A outside the flux-rope; L_g is the coronal height-scale.

      ----------

      The density (and consequent temperature) jump rho~H(r_min-1) 
      replaced by smoother rho~tgh(r_min-1) profile (TPCR-like).
    */

    double rho_0=0.5*(1.0-Tc2Tp)*tanh(densGrad*(r_min-1.0))+0.5*(1+Tc2Tp);
    
    if(r_min>1.0){ // external region

      v[0]=rho_0*exp(-zz*invLg);         // mass density outside
      
      B_loc.add(iSgn*R_t/r_maj,theta0);

      pressure=beta*exp(-zz*invLg);
    }else{ // inside the torus

      v[0]=rho_0*exp(-zz*Tc2Tp*invLg);   // mass density in the loop

      B_loc.add((iSgn*(sqrt(1.0+heliFactor*(1.0-r_min*r_min))+
                    R_t/r_maj-1.0)),theta0);

      pressure=beta*exp(-zz*invLg);
    }
    
    v[1]=0.0;                    
    v[2]=0.0;                          // momentum density
    v[3]=0.0;
    
    v[4]=B_loc[0];
    v[5]=B_loc[1];                  // magnetic field
    v[6]=B_loc[2];
    // energy density
    v[7]=pressure/(this->GAMMA-1.0)+
              v[4]*v[4]+v[5]*v[5]+v[6]*v[6];
    
    v[8]=0.0;                    
    v[9]=0.0;                          // current density
    v[10]=0.0;
    
    v[11]=0.0; // eta
  }
  
  
  /***************************************************************************
          Projection of initial condition
  ***************************************************************************/
  template <int dim>
  void MHDProblem<dim>::project_initial_conditions()
  {
    QGauss<dim>quadrature(FEO+gausIntOrd);// QGaussLobatto<dim>  -- qps are in interpolation points
    FEValues<dim> fe_values(fe, quadrature,
                             update_values   | //update_gradients |
                             update_quadrature_points | update_JxW_values);
    const unsigned int dofs_per_cell = fe_values.dofs_per_cell,
                       n_q_points    = fe_values.n_quadrature_points;
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<Vector<double> > rhs_values(n_q_points, Vector<double>(Ne));
    
    system_matrix=0.0;
    system_rhs=0.0;
    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for(; cell!=endc; ++cell)
      if(cell->is_locally_owned()){
          fe_values.reinit(cell);
          initial_values->vector_value_list(fe_values.get_quadrature_points(),
                                          rhs_values);
          cell_rhs = 0;
          cell_matrix = 0;
          
          for(unsigned int point=0; point<n_q_points; ++point){
            
            for(unsigned int i=0; i<stv2dof.Nstv; i++){
              for(unsigned int j=0; j<stv2dof.Nstv; j++){
                for(unsigned int k=0; k<Nv; k++){
                  const int dof_i=stv2dof.stateD[i][k];
                  if (dof_i<0) continue;
                  const int dof_j=stv2dof.stateD[j][k];
                  if (dof_j<0) continue;
                  cell_matrix(dof_i,dof_j) +=
                        fe_values.shape_value_component(dof_i,point,k)*
                        fe_values.shape_value_component(dof_j,point,k)*
                        fe_values.JxW(point);
                }
              }
              // Assembling the right hand side
              for(unsigned int k=0; k<Nv; k++){
                const int dof_i=stv2dof.stateD[i][k];
                if (dof_i<0) continue;
                cell_rhs(dof_i) += rhs_values[point][k]*
                    fe_values.shape_value_component(dof_i,point,k)*
                    fe_values.JxW(point);
              }
            } // end of i-loop
          }  // end of point-loop
          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(cell_matrix,
                                                cell_rhs,
                                                local_dof_indices,
                                                system_matrix,
                                                system_rhs);       
        }
    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
    
    solve();
  }
  
  // Projection of the initial condition where the magnetic field is defined by a vector potential
  // in this case the operator is set to identity except of the part for magnetic field which
  // is set according to the curl of A
  template <int dim>
  void MHDProblem<dim>::project_init_cnd_vecpot()
  {
    QGauss<dim>quadrature(FEO+gausIntOrd);// QGaussLobatto<dim>  -- qps are in interpolation points
    FEValues<dim> fe_values(fe, quadrature,
                             update_values   | update_gradients |
                             update_quadrature_points | update_JxW_values);
    const unsigned int dofs_per_cell = fe_values.dofs_per_cell,
                       n_q_points    = fe_values.n_quadrature_points;
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<Vector<double> > rhs_values(n_q_points, Vector<double>(Ne));
    double O1[Nv][Nv],O2[Nv][Nv];
    
    for(unsigned int i=0; i<Nv; i++)
      for(unsigned int j=0; j<Nv; j++)
        O1[i][j]=O2[i][j]=0.0;
        
    
    system_matrix=0.0;
    system_rhs=0.0;
    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for(; cell!=endc; ++cell)
      if(cell->is_locally_owned()){
          fe_values.reinit(cell);
          initial_values->vector_value_list(fe_values.get_quadrature_points(),
                                          rhs_values);
          cell_rhs = 0;
          cell_matrix = 0;
          
          for(unsigned int point=0; point<n_q_points; ++point){
            
            for(unsigned int i=0; i<stv2dof.Nstv; i++){
              // set first operator - diagonal componetns
              for(unsigned int k=0; k<Nv; k++)
                if (stv2dof.stateD[i][k]>=0)
                  O1[k][k]=fe_values.shape_value_component(stv2dof.stateD[i][k],point,k);
              // and curl of vector potential (stored at position of J)
              if (stv2dof.stateD[i][8]>=0){
                O1[5][ 8]=-fe_values.shape_grad_component(stv2dof.stateD[i][ 8],point, 8)[2];
                O1[6][ 8]= fe_values.shape_grad_component(stv2dof.stateD[i][ 8],point, 8)[1];
              }
              if (stv2dof.stateD[i][9]>=0){
                O1[4][ 9]= fe_values.shape_grad_component(stv2dof.stateD[i][ 9],point, 9)[2];
                O1[6][ 9]=-fe_values.shape_grad_component(stv2dof.stateD[i][ 9],point, 9)[0];
              }
              if (stv2dof.stateD[i][10]>=0){
                O1[4][10]=-fe_values.shape_grad_component(stv2dof.stateD[i][10],point,10)[1];
                O1[5][10]= fe_values.shape_grad_component(stv2dof.stateD[i][10],point,10)[0];
              }
              
              for(unsigned int j=0; j<stv2dof.Nstv; j++){
                // set first operator - diagonal componetns
                for(unsigned int k=0; k<Nv; k++)
                  if (stv2dof.stateD[j][k]>=0)
                    O2[k][k]=fe_values.shape_value_component(stv2dof.stateD[j][k],point,k);
                // and curl of vector potential (stored at position of J)
                if (stv2dof.stateD[j][8]>=0){
                  O2[5][ 8]=-fe_values.shape_grad_component(stv2dof.stateD[j][ 8],point, 8)[2];
                  O2[6][ 8]= fe_values.shape_grad_component(stv2dof.stateD[j][ 8],point, 8)[1];
                }
                if (stv2dof.stateD[j][9]>=0){
                  O2[4][ 9]= fe_values.shape_grad_component(stv2dof.stateD[j][ 9],point, 9)[2];
                  O2[6][ 9]=-fe_values.shape_grad_component(stv2dof.stateD[j][ 9],point, 9)[0];
                }
                if (stv2dof.stateD[j][10]>=0){
                  O2[4][10]=-fe_values.shape_grad_component(stv2dof.stateD[j][10],point,10)[1];
                  O2[5][10]= fe_values.shape_grad_component(stv2dof.stateD[j][10],point,10)[0];
                }
                // multiplication of the operator matrices
                for(unsigned int k=0; k<Nv; k++){
                  int dof_i=stv2dof.stateD[i][k];
                  if (dof_i<0) continue;
                  for(unsigned int l=0; l<Nv; l++){
                    int dof_j=stv2dof.stateD[j][l];
                    if (dof_j<0) continue;
                    for(unsigned int m=0; m<Nv; m++){
                        cell_matrix(dof_i,dof_j) +=
                            O1[m][k]*O2[m][l]*fe_values.JxW(point);
                    }
                  }
                }
              }
              // Assembling the right hand side
              for(unsigned int k=0; k<Nv; k++){
                int dof_i=stv2dof.stateD[i][k];
                if (dof_i<0) continue;
                for(unsigned int l=0; l<Nv; l++)
                  cell_rhs(dof_i) += O1[l][k]*
                        rhs_values[point](l)*fe_values.JxW(point);
              }

            } // end of i-loop
          }  // end of point-loop
          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(cell_matrix,
                                                cell_rhs,
                                                local_dof_indices,
                                                system_matrix,
                                                system_rhs);       
        }
    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
    
    solve();
  }

} // end of namespace mhd
