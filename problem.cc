#include <iostream>
#include <cstdio>

#include "problem.h"

namespace mhd
{
  template <int dim>  // One dimension - all MHD variables are aproximated by Lagrange elements
  MHDProblem<dim>::MHDProblem(Params &pars) :
      mpi_communicator(MPI_COMM_WORLD),
      triangulation(mpi_communicator,
                    typename Triangulation<dim>::MeshSmoothing
                    (Triangulation<dim>::smoothing_on_refinement |
                    Triangulation<dim>::smoothing_on_coarsening)),
      dof_handler(triangulation), //dof_handler_s(triangulation),
      FEO(pars.getMinElementDegree()),fe(FE_Q<dim>(FEO), Nv),//fes(FEO),
      pcout(std::cout,
          (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
#ifdef USE_TIMER
      ,computing_timer(mpi_communicator, pcout,
                    TimerOutput::summary,
                    TimerOutput::wall_times)
#endif
  {
    setup_parameters(pars);
  }
  /*
  template <> // velocity is Arnold-Boffi-Falk (2D vectors) elements, B is Nedelec (2D vectors) elm. and the rest is Lagrange
  MHDProblem<2>::MHDProblem(Params &pars) :
      mpi_communicator(MPI_COMM_WORLD),
      triangulation(mpi_communicator,
                    typename Triangulation<2>::MeshSmoothing
                    (Triangulation<2>::smoothing_on_refinement |
                    Triangulation<2>::smoothing_on_coarsening)),
      dof_handler(triangulation), //dof_handler_s(triangulation),
      FEO(pars.getMinElementDegree()),
      fe(FE_DGQ<2>(FEO), 1, FE_Q<2>(FEO), 7,FE_Q<2>(FEO), 4),
      //fe(FE_Q<2>(FEO), 1, FE_RaviartThomas<2>(FEO), 1, FE_Q<2>(FEO), 1, FE_Nedelec<2>(FEO), 1, FE_Q<2>(FEO), 6),
      //fes(FEO),
      pcout(std::cout,
          (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
#ifdef USE_TIMER
      ,computing_timer(mpi_communicator, pcout,
                    TimerOutput::summary,
                    TimerOutput::wall_times)
#endif
  {
    setup_parameters(pars);
  }
  */
//   template <>  // velocity is Arnold-Boffi-Falk (3D vectors) elements, B is Nedelec (3D vectors) elm. and the rest is Lagrange
//   MHDProblem<3>::MHDProblem(Params &pars) :
//       mpi_communicator(MPI_COMM_WORLD),
//       triangulation(mpi_communicator,
//                     typename Triangulation<3>::MeshSmoothing
//                     (Triangulation<3>::smoothing_on_refinement |
//                     Triangulation<3>::smoothing_on_coarsening)),
//       dof_handler(triangulation), //dof_handler_s(triangulation),
//       FEO(pars.getMinElementDegree()),
//       fe(FE_Q<3>(FEO), 1, FE_RaviartThomas<3>(FEO), 1, FE_Nedelec<3>(FEO), 1, FE_Q<3>(FEO), 5),
//       //fes(FEO),
//       pcout(std::cout,
//           (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
// #ifdef USE_TIMER
//       ,computing_timer(mpi_communicator, pcout,
//                     TimerOutput::summary,
//                     TimerOutput::wall_times)
// #endif
//   {
//     setup_parameters(pars);
//   }
  
  template <int dim>
  MHDProblem<dim>::~MHDProblem()
  {
    dof_handler.clear();
    //dof_handler_s.clear();
    delete [] operator_matrixes;
    delete [] DIRK;
    delete mhdeq;
  }
  
  template <int dim>
  void MHDProblem<dim>::setup_parameters(Params &pars)
  {
    pcout<<"Element degree: "<<fe.degree<<std::endl;
    
    pcout<<"Parsing parameters"<<std::endl;
    pars.prm.enter_subsection("Output");
    {
      outputFreq=pars.prm.get_double("Output frequency");
    }
    pars.prm.leave_subsection();
    pars.prm.enter_subsection("Simulation");
    {
      totalTime=pars.prm.get_double("Total time");
      initCond=pars.prm.get_integer("Initial condition");
    }
    pars.prm.leave_subsection();
    pars.prm.enter_subsection("Numerics");
    {
      CGMmaxIt=pars.prm.get_integer("CGM interations");
      CGMprec=pars.prm.get_double("CGM tolerance");
      linmaxIt=pars.prm.get_integer("Number of linearizations");
      linPrec=pars.prm.get_double("Linearization tolerance");
      linLevel=pars.prm.get_integer("Simple level");
      gausIntOrd=pars.prm.get_integer("Gauss int ord");
      intMethod=pars.prm.get_integer("Time integration");
    }
    pars.prm.leave_subsection();
    pars.prm.enter_subsection("Mesh refinement");
    {
      meshMinLev=pars.prm.get_integer("Minimum level");
      meshMaxLev=pars.prm.get_integer("Maximum level");
      meshRefGrad=-pars.prm.get_double("Refining gradient"); // switch sign for human convenience
      meshCoaGrad=-pars.prm.get_double("Coarsing gradient");
      initSplit=pars.prm.get_integer("Initial division");
      initRefin=pars.prm.get_integer("Initial refinement");
    }
    pars.prm.leave_subsection();
    pars.prm.enter_subsection("Simulation");
    {
      pars.prm.enter_subsection("Box");
      {
        boxP1[0]=pars.prm.get_double("x_min");
        boxP1[1]=pars.prm.get_double("y_min");
        if (dim==3) boxP1[2]=pars.prm.get_double("z_min");
        boxP2[0]=pars.prm.get_double("x_max");
        boxP2[1]=pars.prm.get_double("y_max");
        if (dim==3) boxP2[2]=pars.prm.get_double("z_max");
      }
      pars.prm.leave_subsection();
    }
    pars.prm.leave_subsection();
    
    pars.setBC(&BCmap[0]);  // sets kind of BC for six box sides
    
    initial_values.setParameters(pars);
    
    mhdeq = new MHDequations<dim>(pars, stv2dof, mpi_communicator);
    mhdeq->setBoxRef(&boxP1,&boxP2);
    
    switch(intMethod){
      case 1:
      case 2:
      case 3:
      case 4:
      case 5:
        timeStepInt=&MHDProblem::DIRKmethod;
        mhdeq->setDIRKMethod(intMethod-1);
        break;
      default:
        timeStepInt=&MHDProblem::thetaMethod;
        mhdeq->setThetaMethod();
        break;
    }
    
    DIRK = new LA::MPI::Vector[mhdeq->DIRK.maxStageAll];
  }
  
  template <int dim>
  void MHDProblem<dim>::setDofMapping()
  {
    // Set mapping from dof to MHD system component and vice versa
    QGauss<dim> quadrature(FEO+gausIntOrd);
    double intFEval;
    FEValues<dim> fe_values(fe, quadrature, update_values | update_quadrature_points);
    fe_values.reinit(dof_handler.begin_active());
    for(unsigned int i=0;i<fe_values.dofs_per_cell;i++)
      for(unsigned int k=0;k<Nv;k++){
        intFEval=0.0;
        for(unsigned int l=0;l<fe_values.n_quadrature_points;l++)     // sum it for all q_points
          intFEval+=std::fabs(fe_values.shape_value_component(i,l,k));
        if (intFEval>1e-6){ 
          stv2dof.cmpInx.push_back(k);
          stv2dof.dof.push_back(i);
        }
      }
    stv2dof.Ndofs=stv2dof.cmpInx.size();
    
    //std::vector<std::array<int, Nv>> stateD;
    std::vector<unsigned int> indx(stv2dof.cmpInx);
    std::vector<unsigned int> dof(stv2dof.dof);
    unsigned int i=0; // index of the state vector
    for(;;){  // create state vector mapping
      if (indx.size()==0) break;
      stv2dof.stateD.push_back(std::array<unsigned int, Nv>());
      for(unsigned int k=0;k<Nv;k++){ // we need to find all variables to construct one state vector
        unsigned int l,ins=indx.size();
        for(l=0;l<ins;l++)
          if (indx[l]==k){
            stv2dof.stateD[i][k]=dof[l];
            indx.erase(indx.begin()+l);
            dof.erase(dof.begin()+l);
            break;
          }
        if (l==ins) stv2dof.stateD[i][k]=-1; // nothing is found - no more corresponding basis fce.
      }
      i++;
    }
    stv2dof.Nstv=stv2dof.stateD.size();
    
    mhdeq->reinitFEval();
    
    operator_matrixes = new FullMatrix<double>[stv2dof.Nstv];
    for(unsigned int i=0;i<stv2dof.Nstv;i++)
        operator_matrixes[i].reinit(Ne,Nv);
  }
  
  template <int dim>
  void MHDProblem<dim>::setup_system()
  {
#ifdef USE_TIMER
    TimerOutput::Scope t(computing_timer, "setup");
#endif

    dof_handler.distribute_dofs(fe);
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler,
                                         locally_relevant_dofs);

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();

    //CompressedSimpleSparsityPattern csp(locally_relevant_dofs);
    DynamicSparsityPattern sparsity_pattern(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity_pattern,
                                   constraints, false);
    SparsityTools::distribute_sparsity_pattern(sparsity_pattern,
                                              dof_handler.n_locally_owned_dofs_per_processor(),
                                              mpi_communicator,
                                              locally_relevant_dofs);
    system_matrix.reinit(locally_owned_dofs,
                        locally_owned_dofs,
                        sparsity_pattern,
                        mpi_communicator);
    
    solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    lin_solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    old_solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    if (intMethod>0)
      for(unsigned int i=0;i<mhdeq->DIRK.maxStageAll;i++)
        DIRK[i].reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    system_rhs.reinit(locally_owned_dofs, mpi_communicator);  // no ghosts
    residue.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    distributed_solution.reinit(locally_owned_dofs, mpi_communicator);  // for solver - no ghosts
    shockIndicator.reinit(triangulation.n_active_cells());
    shockWeights.reinit(triangulation.n_active_cells()*Nv);
    
    shockIndicator=1.0;
    shockWeights=1.0;
    // make a array for eta
    //dof_handler_s.distribute_dofs(fes);
    //local_dofs = dof_handler_s.locally_owned_dofs();
    //DoFTools::extract_locally_relevant_dofs(dof_handler_s,
    //                                     local_relevant_dofs);
//     eta.reinit(local_dofs, local_relevant_dofs, mpi_communicator); 
//     eta_dist.reinit(local_dofs, mpi_communicator);  // for solver - no ghosts
  }
  
  template <int dim>
  void MHDProblem<dim>::assemble_system(const int /*iter*/)
  {
#ifdef USE_TIMER
    TimerOutput::Scope t(computing_timer, "assembly");
#endif
    double *weights=mhdeq->getWeights();
    QGauss<dim>  quadrature_formula(FEO+gausIntOrd);
    QGauss<dim-1> face_quadrature_formula(FEO+gausIntOrd);
    
    
//     FEValues<dim> fes_values(fes, quadrature_formula,
//                              update_values   | update_gradients);
//     FEFaceValues<dim> fes_face_values(fes, face_quadrature_formula,
//                              update_values   | update_gradients);

    FEValues<dim> fe_values(fe, quadrature_formula,
                             update_values   | update_gradients |
                             update_quadrature_points | update_JxW_values);
    
    FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
                                  update_values | update_gradients | update_quadrature_points |
                                  update_normal_vectors | update_JxW_values);

    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();
    const unsigned int   n_f_q_points   = face_quadrature_formula.size();

    Vector<double>       cell_rhs_lin(Ne);
    FullMatrix<double>   cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs(dofs_per_cell);

    std::vector<Vector<double> >  old_sv(n_q_points, Vector<double>(Nv));
    std::vector<Vector<double> >  lin_sv(n_q_points, Vector<double>(Nv));
    std::vector<Vector<double> >  dirk1v(n_q_points, Vector<double>(Nv));
    std::vector<Vector<double> >  dirk2v(n_q_points, Vector<double>(Nv));
    std::vector<Vector<double> >  dirk3v(n_q_points, Vector<double>(Nv));
    
    std::vector<std::vector<Tensor<1,dim> > >  old_sg(n_q_points,
                                        std::vector<Tensor<1,dim> > (Nv));
    std::vector<std::vector<Tensor<1,dim> > >  lin_sg(n_q_points,
                                        std::vector<Tensor<1,dim> > (Nv));
    std::vector<std::vector<Tensor<1,dim> > >  dirk1g(n_q_points,
                                        std::vector<Tensor<1,dim> > (Nv));
    std::vector<std::vector<Tensor<1,dim> > >  dirk2g(n_q_points,
                                        std::vector<Tensor<1,dim> > (Nv));
    std::vector<std::vector<Tensor<1,dim> > >  dirk3g(n_q_points,
                                        std::vector<Tensor<1,dim> > (Nv));
    std::vector<Vector<double> >* pVecVec[5];
    std::vector<std::vector<Tensor<1,dim> > >* pVecTen[5];
        
    pVecVec[0]=&old_sv;
    pVecVec[1]=&lin_sv;
    pVecVec[2]=&dirk1v;
    pVecVec[3]=&dirk2v;
    pVecVec[4]=&dirk3v;
    
    pVecTen[0]=&old_sg;
    pVecTen[1]=&lin_sg;
    pVecTen[2]=&dirk1g;
    pVecTen[3]=&dirk2g;
    pVecTen[4]=&dirk3g;
        
    std::vector<Vector<double> >  old_svf(n_f_q_points, Vector<double>(Nv));
    std::vector<Vector<double> >  lin_svf(n_f_q_points, Vector<double>(Nv));
    std::vector<Vector<double> >  dirk1vf(n_f_q_points, Vector<double>(Nv));
    std::vector<Vector<double> >  dirk2vf(n_f_q_points, Vector<double>(Nv));
    std::vector<Vector<double> >  dirk3vf(n_f_q_points, Vector<double>(Nv));
    
    std::vector<std::vector<Tensor<1,dim> > >  old_sgf(n_f_q_points,
                                        std::vector<Tensor<1,dim> > (Nv));
    std::vector<std::vector<Tensor<1,dim> > >  lin_sgf(n_f_q_points,
                                        std::vector<Tensor<1,dim> > (Nv));
    std::vector<std::vector<Tensor<1,dim> > >  dirk1gf(n_f_q_points,
                                        std::vector<Tensor<1,dim> > (Nv));
    std::vector<std::vector<Tensor<1,dim> > >  dirk2gf(n_f_q_points,
                                        std::vector<Tensor<1,dim> > (Nv));
    std::vector<std::vector<Tensor<1,dim> > >  dirk3gf(n_f_q_points,
                                        std::vector<Tensor<1,dim> > (Nv));
    std::vector<Vector<double> >* pVecVecf[5];
    std::vector<std::vector<Tensor<1,dim> > >* pVecTenf[5];
    pVecVecf[0]=&old_svf;
    pVecVecf[1]=&lin_svf;
    pVecVecf[2]=&dirk1vf;
    pVecVecf[3]=&dirk2vf;
    pVecVecf[4]=&dirk3vf;
    
    pVecTenf[0]=&old_sgf;
    pVecTenf[1]=&lin_sgf;
    pVecTenf[2]=&dirk1gf;
    pVecTenf[3]=&dirk2gf;
    pVecTenf[4]=&dirk3gf;
    
//     std::vector<Vector<double> >  cell_residue(n_q_points, Vector<double>(Nv));
    
//     std::vector<double>  eta_v(n_q_points);
//     std::vector<Tensor<1,dim> >  eta_g(n_q_points);
//     std::vector<double>  eta_vf(n_f_q_points);
//     std::vector<Tensor<1,dim> >  eta_gf(n_f_q_points);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    //typename MHDequations<dim>::RightHandSide      right_hand_side;
    std::vector<Vector<double> > rhs_values(n_q_points, Vector<double>(Ne));
    std::vector<Vector<double> > init_values(n_q_points, Vector<double>(Nv));
    //std::vector<Vector<double> > feval(n_q_points, Vector<double>(4));

    system_matrix=0.0;
    system_rhs=0.0;
    
    // Now we can begin with the loop over all cells:
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
                                                   endc = dof_handler.end();//,
//                                                    cells = dof_handler_s.begin_active();
    unsigned int cellNo=0;
    for(; cell!=endc; ++cell,++cellNo)
      if (cell->is_locally_owned()){
        
        cell_matrix = 0;
        cell_rhs = 0;

        fe_values.reinit(cell);
//         fes_values.reinit(cells);
        
        fe_values.get_function_values(old_solution, old_sv);
        fe_values.get_function_gradients(old_solution, old_sg);
        fe_values.get_function_values(lin_solution, lin_sv);
        fe_values.get_function_gradients(lin_solution, lin_sg);
        for(int i=0; i<=mhdeq->DIRK.stage; i++){
          fe_values.get_function_values(DIRK[i], *(pVecVec[2+i]));
          fe_values.get_function_gradients(DIRK[i], *(pVecTen[2+i]));
        }
//         fe_values.get_function_values(residue, cell_residue);
//         fes_values.get_function_values(eta, eta_v);
//         fes_values.get_function_gradients(eta, eta_g);

        mhdeq->rhs.vector_value_list(fe_values.get_quadrature_points(),
                                           rhs_values);

        // determine error on the cell for choosing way of constructing matrix
        /*double maxErr=0.0;
        for(unsigned int q_point=0; q_point<n_q_points; ++q_point)
          for(unsigned int i=0; i<Nv; i++){
            double er=std::fabs(cell_residue[q_point](i));
            if (er>maxErr) maxErr=er;
          }

        if (maxErr<linPrec*1e-1){  // directly copy old linearization value into new linearization
          for(unsigned int point=0; point<n_q_points; ++point){
          
            for(unsigned int i=0; i<dofs_per_cell; ++i){
              const unsigned int cmp_i = fe.system_to_component_index(i).first;
              cell_rhs(i) += lin_sv[point][cmp_i]*
                                fe_values.shape_value(i,point)*
                                fe_values.JxW(point);

              cell_matrix(i,i) += fe_values.shape_value(i,point)*
                                fe_values.shape_value(i,point)*
                                fe_values.JxW(point);
            }
          }
        }else{*/  // regular matrix construction
          mhdeq->useNRLinearization(cell->level()>linLevel/* || maxErr>linPrec*1e3*/);

          // Then assemble the entries of the local stiffness matrix and right
          // hand side vector.
          for(unsigned int q_point=0; q_point<n_q_points; ++q_point){
            mhdeq->set_state_vector_for_qp(pVecVec,pVecTen, q_point);
            mhdeq->setFEvals(fe_values, q_point);
            mhdeq->calucate_matrix_rhs(operator_matrixes,cell_rhs_lin);
            
            for(unsigned int i=0; i<stv2dof.Nstv; i++){
              for(unsigned int j=0; j<stv2dof.Nstv; j++){
                //if (mhdeq->isDiagonal() && i!=j) continue;
                for(unsigned int k=0; k<Nv; k++){
                  int dof_i=stv2dof.stateD[i][k];
                  if (dof_i<0) continue;
                  for(unsigned int l=0; l<Nv; l++){
                    int dof_j=stv2dof.stateD[j][l];
                    if (dof_j<0) continue;
                    for(unsigned int m=0; m<Ne; m++){
                        cell_matrix(dof_i,dof_j) +=
                            operator_matrixes[i](m,k)*
                            operator_matrixes[j](m,l)*
                            weights[m]*fe_values.JxW(q_point);
                    }
                  }
                }
              }
              // Assembling the right hand side
              for(unsigned int k=0; k<Nv; k++){
                int dof_i=stv2dof.stateD[i][k];
                if (dof_i<0) continue;
                for(unsigned int l=0; l<Ne; l++)
                  cell_rhs(dof_i) += operator_matrixes[i](l,k)*
                        (rhs_values[q_point](l)+cell_rhs_lin(l))*
                        weights[l]*fe_values.JxW(q_point);
              }

            } // end of i-loop
          }   // end of q-points

          // Boundary conditions
          for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
            if (cell->face(face_number)->at_boundary() && !mhdeq->isDiagonal()){
                fe_face_values.reinit(cell, face_number);
//                 fes_face_values.reinit(cells, face_number);
                
                fe_face_values.get_function_values(old_solution, old_svf);
                fe_face_values.get_function_gradients(old_solution, old_sgf);
                fe_face_values.get_function_values(lin_solution, lin_svf);
                fe_face_values.get_function_gradients(lin_solution, lin_sgf);
                for(int i=0; i<=mhdeq->DIRK.stage; i++){
                  fe_face_values.get_function_values(DIRK[i], *(pVecVecf[2+i]));
                  fe_face_values.get_function_gradients(DIRK[i], *(pVecTenf[2+i]));
                }
                
                mhdeq->rhs.vector_value_list(fe_face_values.get_quadrature_points(),
                                            rhs_values);
                initial_values.vector_value_list(fe_face_values.get_quadrature_points(),
                                            init_values);
                
                for(unsigned int q_point=0; q_point<n_f_q_points; ++q_point){
                  mhdeq->setFEvals(fe_face_values,q_point); // call it before BC
                  
                  // get pointer to function which setup BC for given type defined in params
                  int bi = cell->face(face_number)->boundary_id();
                  const Tensor<1,dim> &nrm=fe_face_values.normal_vector(q_point);
                  const Point<dim> &pt=fe_face_values.quadrature_point(q_point);
                  // call BC function and setup state vectors
                  (mhdeq->*(mhdeq->BCp[BCmap[bi] ])) (
                              pVecVecf,pVecTenf, init_values,
                              nrm, pt, q_point);

                  mhdeq->setFEvals(fe_face_values,q_point);
                  mhdeq->calucate_matrix_rhs(operator_matrixes,cell_rhs_lin);
                  
                  for(unsigned int i=0; i<stv2dof.Nstv; i++){ // unfortunately we have to go all over the dofs_per_cell
                    for(unsigned int j=0; j<stv2dof.Nstv; j++){ //   dofs_per_face does not cointain proper basis fce
                      for(unsigned int k=0; k<Nv; k++){
                      int dof_i=stv2dof.stateD[i][k];
                      if (dof_i<0) continue;
                        for(unsigned int l=0; l<Nv; l++){
                          int dof_j=stv2dof.stateD[j][l];
                          if (dof_j<0) continue;
                            for(unsigned int m=0; m<Ne; m++){
                              cell_matrix(dof_i,dof_j) +=
                                  operator_matrixes[i](m,k)*
                                  operator_matrixes[j](m,l)*
                                  weights[m]*fe_face_values.JxW(q_point); // /cell->diameter()
                          }
                        }
                      }
                    }
                    // Assembling the right hand side
                    for(unsigned int k=0; k<Nv; k++){
                      int dof_i=stv2dof.stateD[i][k];
                      if (dof_i<0) continue;
                      for(unsigned int l=0; l<Ne; l++)
                        cell_rhs(dof_i) += operator_matrixes[i](l,k)*
                              (rhs_values[q_point](l)+cell_rhs_lin(l))*
                              weights[l]*fe_face_values.JxW(q_point); // /cell->diameter()
                      }
                          
                  }  // end of i
                }  // end of q_point
            } // end of boundary
        //}  // no linearization condition

        // The transfer from local degrees of freedom into the global matrix
        cell->get_dof_indices(local_dof_indices); // TODO: Matrix is symetric, optimalization is possible!
        constraints.distribute_local_to_global(cell_matrix,
                                              cell_rhs,
                                              local_dof_indices,
                                              system_matrix,
                                              system_rhs);
    } // end of cell loop
    
    // TODO: using numerical flux instead of constraints - tutorial 33

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }
  
  template <int dim>
  unsigned int MHDProblem<dim>::solve()
  {
#ifdef USE_TIMER
    TimerOutput::Scope t(computing_timer, "solve");
#endif
    SolverControl solver_control(CGMmaxIt, CGMprec); // 1000, 1e-14
    
    LA::SolverCG solver(solver_control);//, mpi_communicator);
    
    //LA::MPI::PreconditionAMG preconditioner;
    //LA::MPI::PreconditionAMG::AdditionalData data;
    
    LA::PreconditionJacobi preconditioner;
    LA::PreconditionJacobi::AdditionalData data;
#ifdef USE_PETSC_LA
    //data.symmetric_operator = true;
#else
    /* Trilinos defaults are good */
#endif
    preconditioner.initialize(system_matrix, data);
    solver.solve (system_matrix, distributed_solution, system_rhs,
                  preconditioner);

    constraints.distribute(distributed_solution);
    
    solution=distributed_solution;
    
    return solver_control.last_step();
  }
  
  template <int dim>
  void MHDProblem<dim>::run()
  {
    unsigned int sitr,output_counter=0;
    unsigned int time_step=0;
    unsigned int iter,linReset;
//     double lastErr,err;
    //bool overflow=false;
    double sz[dim],maxSiz=0.0;
    std::vector<unsigned int> rep(dim);

    pcout<<"Initial mesh refinement..."<<std::endl;
    for(unsigned int i=0;i<dim;i++){
      sz[i]=std::fabs(boxP1[i]-boxP2[i]);
      if (sz[i]>maxSiz) maxSiz=sz[i];
    }
    // initial refinement of the box is non-homogenous in order to have cube cells (if possible)
    for(unsigned int i=0;i<dim;i++){
      rep[i]=(unsigned int)(4*sz[i]/maxSiz+0.5);
      if (rep[i]==0) rep[i]=1;
    }
    
    GridGenerator::subdivided_hyper_rectangle(triangulation, rep, boxP1, boxP2, true); // set box and colorize boundaries
    triangulation.refine_global(initSplit);
    mhdeq->setMinh(GridTools::minimal_cell_diameter(triangulation)/FEO);
    setup_system();
    setDofMapping();
    pcout << "   initial mesh, ndofs: "<< dof_handler.n_dofs() << std::endl;
    
    pcout<<"Refine gradients in initial conditions..."<<std::endl;
    for(unsigned int i=0;i<initRefin;i++){ // 5
      project_initial_conditions();
      //mhdeq->checkOverflow(distributed_solution,distributed_solution); //check overflow
      //(mhdeq->*(mhdeq->setEta))(distributed_solution,eta,eta_dist);
      solution = distributed_solution;  // copy results to the solution (it includes ghosts)
      //refine_grid_simple();  // error estimator does not work
      setShockSmoothCoef(); // set refinement coeficients
      refine_grid_rule();
      pcout << "   "<<i<<". initial refinement, ndofs: "
                  << dof_handler.n_dofs() << std::endl;
    }
    pcout << "   Number of active cells:       "
                  << triangulation.n_active_cells()<< std::endl;
    pcout << "   Number of degrees of freedom: "
                  << dof_handler.n_dofs() << std::endl;
    pcout << "   Setting initial conditions..."<<std::endl;

    project_initial_conditions();
    //mhdeq->checkOverflow(distributed_solution,distributed_solution); //check overflow
    corrections();
    solution = distributed_solution;  // copy results to the solution (it includes ghosts)
    //(mhdeq->*(mhdeq->setEta))(distributed_solution,eta,eta_dist);
    //mhdeq->checkDt(solution);         // find and set dt
    mhdeq->setNewDt();
    
    pcout << "   First output."<<std::endl;
    old_solution=solution;
    setShockSmoothCoef(); // set refinement coeficients
    output_results(output_counter++);
    double time;
    mhdeq->setTimeRef(&time);
    for(time=0.0; time<totalTime; ){
        if (time_step%10==0) pcout << "Time: " << time <<", time step: "<<time_step<< std::endl;

        lin_solution=old_solution;
// 	residue=9e99;
        //lastErr=9e99;
        //linReset=iter=sitr=0;
        /*
        for(;;){ // linearization - do one time step
            assemble_system(iter);
            sitr += solve();
            //overflow=mhdeq->checkOverflow(distributed_solution,old_solution);
            corrections();
            //solution=distributed_solution;
            //mhdeq->checkDt(solution);
            system_rhs=lin_solution;
            system_rhs-=distributed_solution;
            err=system_rhs.linfty_norm();//norm_sqr();
            if (err<linPrec) break;  //linfty_norm
            if ((err>8.*lastErr && iter>1) || false){                 // linearization do not converge
              //if (err<1e-7*dof_handler.n_dofs()) break; // still ok error...
              mhdeq->setDt(mhdeq->getDt()*0.5);
              lin_solution=old_solution;
              iter=sitr=0;
              lastErr=9e99;
              linReset++;
              pcout<< linReset<<" linearization reset. dt= "<<mhdeq->getDt()<<std::endl;
              if (linReset>=7){
                pcout<< "Linearization do not converge."<<std::endl;
                return;
              }
              continue;
            }
            
            lastErr=err;
            iter++;
            if (iter>=linmaxIt) break;
            solution.swap(lin_solution);
            //pcout<<sitr<<" ";
//             pcout<< "---t: " << time<<" it: "<<iter<<" cgm:"<<sitr
//                  << " res: "<<err<<
//                  " vmax="<<mhdeq->getVmax()<<" "<<overflow<<std::endl;
        }
        */
        (this->*timeStepInt)(iter,sitr);
        //(mhdeq->*(mhdeq->setEta))(distributed_solution,eta,eta_dist);
        //mhdeq->checkDt(solution);
        if (time_step%1==0) pcout << "l. it.: " << iter << 
                                          " s. it.:"<< sitr << 
                                          " dt="<<mhdeq->getDt()<<
                                          " vmax="<<mhdeq->getVmax()<<std::endl;

        if (output_counter*outputFreq<=time){
            pcout << output_counter << ". output" << std::endl;
            setShockSmoothCoef(); // set refinement coeficients
            output_results(output_counter++);
        }

        time_step++;
        time+=mhdeq->getDt();
        mhdeq->setNewDt();

        if (time_step%5==0){
          setShockSmoothCoef(); // set refinement coeficients
          refine_grid_rule();
          void_step();          // clean div B
          corrections();
          /*distributed_solution=solution;
          mhdeq->checkOverflow(distributed_solution,solution);
          old_solution=distributed_solution;
          (mhdeq->*(mhdeq->setEta))(distributed_solution,eta,eta_dist);*/
          pcout << "   refinement, ndofs: "<< dof_handler.n_dofs() << std::endl;
          continue;
        }

       old_solution=solution;
    }
  }

  template <int dim>
  void MHDProblem<dim>::thetaMethod(unsigned int &iter, unsigned int &sitr)
  {
    lin_solution=old_solution;
    iter=sitr=0;
    for(;;){ // linearization
      assemble_system(iter);
      sitr += solve();
      corrections();
      system_rhs=lin_solution;
      system_rhs-=distributed_solution;
      double err=system_rhs.linfty_norm();//norm_sqr();
      if (err<linPrec) break;  //linfty_norm
      iter++;
      if (iter>=linmaxIt) break;
      solution.swap(lin_solution);
    }
  }

  template <int dim>
  void MHDProblem<dim>::DIRKmethod(unsigned int &iter, unsigned int &sitr)
  {
    lin_solution=old_solution;
    iter=sitr=0;
    for(int i=0;i<mhdeq->DIRK.maxStage;i++){
      //pcout<<"DIRK stage: "<<i<<"\n";
      mhdeq->setDIRKStage(i);
      for(;;){ // linearization
        //pcout<<"lin. step: "<<iter<<"\n";
        assemble_system(i);
        sitr += solve();
        corrections();
        system_rhs=lin_solution;
        system_rhs-=distributed_solution;
        double err=system_rhs.linfty_norm();//norm_sqr();
        if (err<linPrec) break;  //linfty_norm
        iter++;
        if (iter>=linmaxIt) break;
        solution.swap(lin_solution);
      }
      DIRK[i]=solution;
    }
//     output_results(1);
//     pcout<<iter<<" lin it, DIRK end stage: "<<mhdeq->DIRK.maxStage<<"\n";
    mhdeq->setDIRKStage(mhdeq->DIRK.maxStage);
    assemble_system(mhdeq->DIRK.maxStage);
    sitr += solve();
    corrections();
//     output_results(2);
  }
  
  template <int dim>
  void MHDProblem<dim>::void_step()  // perform one time step with dt=0
  {                                  // Calculate current density and clean div B
    old_solution=solution;
    lin_solution=old_solution;
    mhdeq->setDt(0.0);
    if (intMethod!=0) mhdeq->setDIRKStage(0);
    assemble_system(0);
    solve();
    //mhdeq->checkOverflow(distributed_solution,old_solution);
    solution=distributed_solution;
    //(mhdeq->*(mhdeq->setEta))(distributed_solution,eta,eta_dist);
    //mhdeq->checkDt(solution);
    mhdeq->setNewDt();
  }
  
  template <int dim>
  void MHDProblem<dim>::corrections()
  {
    double RHSvalue,Uk,Um,buf;
    QGaussLobatto<dim> quadrature(FEO+gausIntOrd);// QGaussLobatto<dim>  -- qps are in interpolation points
    FEValues<dim> fe_values(fe, quadrature,
                             update_values   | //update_gradients |
                             update_quadrature_points | update_JxW_values);
    const unsigned int dofs_per_cell = fe_values.dofs_per_cell,
                       n_q_points    = fe_values.n_quadrature_points;
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<Vector<double> >  ov(n_q_points, Vector<double>(Nv));
    std::vector<Vector<double> >  lv(n_q_points, Vector<double>(Nv));
    
    
    system_matrix=0.0;
    system_rhs=0.0;
    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for(; cell!=endc; ++cell)
      if(cell->is_locally_owned()){
          fe_values.reinit(cell);
          
          fe_values.get_function_values(old_solution, ov);
          fe_values.get_function_values(solution, lv);
          
          cell_rhs = 0;
          cell_matrix = 0;
              
          for(unsigned int p=0; p<n_q_points; ++p){
          
            for(unsigned int i=0; i<stv2dof.Nstv; i++){
              for(unsigned int j=0; j<stv2dof.Nstv; j++){ // Assemble system for identity operator
                for(unsigned int k=0; k<Nv; k++){
                  const int dof_i=stv2dof.stateD[i][k];
                  if (dof_i<0) continue;
                  const int dof_j=stv2dof.stateD[j][k];
                  if (dof_j<0) continue;
                  
                  cell_matrix(dof_i,dof_j) += fe_values.shape_value_component(dof_i,p,k)*
                                    fe_values.shape_value_component(dof_j,p,k)*
                                    fe_values.JxW(p);
                }
              }
              for(unsigned int k=0; k<Nv; k++){      // Check values for underflow (density and pressue)
                const int dof_i=stv2dof.stateD[i][k];
                if (dof_i<0) continue;
                switch(k){
                  case 0:  // density
                    RHSvalue=lv[p][k];
                    if (RHSvalue<0.25) RHSvalue=0.25;
                    break;
                  case 7:  // pressure
                    RHSvalue=lv[p][k];
                    Uk=(lv[p][1]*lv[p][1]+lv[p][2]*lv[p][2]+lv[p][3]*lv[p][3])/(lv[p][0]);
                    Um=lv[p][4]*lv[p][4]+lv[p][5]*lv[p][5]+lv[p][6]*lv[p][6];
                    buf=Um+Uk;
                    if ((RHSvalue-buf)<1e-6) RHSvalue=buf+1e-6;
                    break;
//                   case 1:
//                   case 2:
//                   case 3:
//                   case 4:
//                   case 5:
//                   case 6:
//                   case 8:
//                   case 9:
//                   case 10:
//                   case 11:
                  default:
                    RHSvalue=lv[p][k];
                  break;
                }
                cell_rhs(dof_i) += RHSvalue*
                                  fe_values.shape_value_component(dof_i,p,k)*
                                  fe_values.JxW(p);
              }

            } // end of i-loop
          }  // end of p-loop
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
