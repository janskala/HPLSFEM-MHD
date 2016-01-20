#include <iostream>
#include <cstdio>

#include "problem.h"

namespace mhd
{
  template <int dim>
  MHDProblem<dim>::MHDProblem(Params &pars) :
      mpi_communicator(MPI_COMM_WORLD),
      triangulation(mpi_communicator,
                    typename Triangulation<dim>::MeshSmoothing
                    (Triangulation<dim>::smoothing_on_refinement |
                    Triangulation<dim>::smoothing_on_coarsening)),
      dof_handler(triangulation), dof_handler_s(triangulation),
      pcout(std::cout,
          (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
#ifdef USE_TIMER
      ,computing_timer(mpi_communicator, pcout,
                    TimerOutput::summary,
                    TimerOutput::wall_times)
#endif
  {
    int initCond;
    
    mhdeq = new MHDequations<dim>(/*dof_handler,*/ pars, mpi_communicator);
    
    pcout<<"Parsing parameters"<<std::endl;
    pars.prm.enter_subsection("Output");
    {
      outputFreq=pars.prm.get_double("Output frequency");
    }
    pars.prm.leave_subsection();
    pars.prm.enter_subsection("Simulation");
    {
      totalTime=pars.prm.get_double("Total time");
    }
    pars.prm.leave_subsection();
    pars.prm.enter_subsection("Numerics");
    {
      CGMmaxIt=pars.prm.get_integer("CGM interations");
      CGMprec=pars.prm.get_double("CGM tolerance");
      linmaxIt=pars.prm.get_integer("Number of linearizations");
      linPrec=pars.prm.get_double("Linearization tolerance");
      linLevel=pars.prm.get_integer("Simple level");
    }
    pars.prm.leave_subsection();
    pars.prm.enter_subsection("Mesh refinement");
    {
      meshMinLev=pars.prm.get_integer("Minimum level");
      meshMaxLev=pars.prm.get_integer("Maximum level");
      meshRefGrad=pars.prm.get_double("Refining gradient");
      meshCoaGrad=pars.prm.get_double("Coarsing gradient");
      initSplit=pars.prm.get_integer("Initial division");
      initRefin=pars.prm.get_integer("Initial refinement");
      afDegMin=pars.prm.get_integer("aprox deg min");
      afDegMax=pars.prm.get_integer("aprox deg max");
    }
    pars.prm.leave_subsection();
    pars.prm.enter_subsection("Simulation");
    {
      initCond=pars.prm.get_integer("Initial condition");
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
    
    initial_values.setInitialCondition(initCond);
    
    pcout<<"Min and max element degree: "<<afDegMin<<" and "<<afDegMax<<std::endl;
    // prepare colection od FE basis functions and vectors for 
    //   old/linearization solutions needed during assembling
    fesys=new FESystem<dim>*[afDegMax-afDegMin+1];
    qpVals= new QPVals[afDegMax-afDegMin+1];
    for(unsigned int degree=afDegMin,i=0; degree<=afDegMax; ++degree,++i){
      fesys[i]=new FESystem<dim>(FE_Q<dim>(degree),Nv);
      fe_collection.push_back(*fesys[i]);
      fe_collection_s.push_back(FE_Q<dim>(degree));
      quadrature_collection.push_back(QGauss<dim>(degree+1));
      face_quadrature_collection.push_back(QGauss<dim-1>(degree+1));
      quadrature_col_proj.push_back(QGaussLobatto<dim>(degree+1));
      
      int nqp=quadrature_collection[i].size();
      int nqpf=quadrature_collection[i].size();
      qpVals[i].old_sv.resize(nqp, Vector<double>(Nv));
      qpVals[i].lin_sv.resize(nqp, Vector<double>(Nv));
      qpVals[i].rhs_values.resize(nqp, Vector<double>(Ne));
      qpVals[i].init_values.resize(nqp, Vector<double>(Nv));
      qpVals[i].old_svf.resize(nqpf, Vector<double>(Nv));
      qpVals[i].lin_svf.resize(nqpf, Vector<double>(Nv));
      
      qpVals[i].old_sg.resize(nqp, std::vector<Tensor<1,dim> >(Nv));
      qpVals[i].lin_sg.resize(nqp, std::vector<Tensor<1,dim> >(Nv));
      qpVals[i].old_sgf.resize(nqpf, std::vector<Tensor<1,dim> >(Nv));
      qpVals[i].lin_sgf.resize(nqpf, std::vector<Tensor<1,dim> >(Nv));
      
      qpVals[i].eta_v.resize(nqp);
      qpVals[i].eta_g.resize(nqp);
      qpVals[i].eta_vf.resize(nqpf);
      qpVals[i].eta_gf.resize(nqpf);
    }
    
    unsigned int maxDofs=fesys[afDegMax-afDegMin]->dofs_per_cell/Nv;
    operator_matrixes = new FullMatrix<double>[maxDofs];
    for(unsigned int i=0;i<maxDofs;i++)
        operator_matrixes[i].reinit(Ne,Nv);
  }
  
  template <int dim>
  MHDProblem<dim>::~MHDProblem()
  {
    dof_handler.clear();
    dof_handler_s.clear();
    
    delete mhdeq;
    
    for(unsigned int degree=afDegMin,i=0; degree<=afDegMax; ++degree,++i)
      delete fesys[i];
    delete [] fesys;
    
    delete [] operator_matrixes;
    delete [] qpVals;
  }
  
  template <int dim>
  void MHDProblem<dim>::setup_system()
  {
#ifdef USE_TIMER
    TimerOutput::Scope t(computing_timer, "setup");
#endif

    dof_handler.distribute_dofs(fe_collection);
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
    system_rhs.reinit(locally_owned_dofs, mpi_communicator);  // no ghosts
    residue.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    distributed_solution.reinit(locally_owned_dofs, mpi_communicator);  // for solver - no ghosts
    shockIndicator.reinit(triangulation.n_active_cells());
    shockWeights.reinit(triangulation.n_active_cells()*Nv);
    
    shockIndicator=1.0;
    shockWeights=1.0;
    // make a array for eta
    dof_handler_s.distribute_dofs(fe_collection_s);
    local_dofs = dof_handler_s.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler_s,
                                         local_relevant_dofs);
    eta.reinit(local_dofs, local_relevant_dofs, mpi_communicator); 
    eta_dist.reinit(local_dofs, mpi_communicator);  // for solver - no ghosts
  }
  
  template <int dim>
  void MHDProblem<dim>::assemble_system(const int /*iter*/)
  {
#ifdef USE_TIMER
    TimerOutput::Scope t(computing_timer, "assembly");
#endif
    double *weights=mhdeq->getWeights();
    
    hp::FEValues<dim> hp_fes_values(fe_collection_s, quadrature_collection,
                             update_values   | update_gradients);
    hp::FEFaceValues<dim> hp_fes_face_values(fe_collection_s, face_quadrature_collection,
                             update_values   | update_gradients);

    hp::FEValues<dim> hp_fe_values(fe_collection, quadrature_collection,
                             update_values   | update_gradients |
                             update_quadrature_points | update_JxW_values);
    
    hp::FEFaceValues<dim> hp_fe_face_values(fe_collection, face_quadrature_collection,
                                  update_values | update_gradients | update_quadrature_points |
                                  update_normal_vectors | update_JxW_values);

    Vector<double>       cell_rhs_lin(Ne);
    FullMatrix<double>   cell_matrix;
    Vector<double>       cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;

    RightHandSide<dim> right_hand_side;

    system_matrix=0.0;
    system_rhs=0.0;
    
    // Now we can begin with the loop over all cells:
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
                                                   endc = dof_handler.end(),
                                                   cells = dof_handler_s.begin_active();
    unsigned int cellNo=0;
    for(; cell!=endc; ++cell,++cells,++cellNo)
      if (cell->is_locally_owned()){
        unsigned int d = cell->active_fe_index();  // degree
        const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
        
        cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
        cell_rhs.reinit(dofs_per_cell);
        cell_matrix = 0;
        cell_rhs = 0;
        
        hp_fe_values.reinit(cell);
        hp_fes_values.reinit(cell);
        
        const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();
        const FEValues<dim> &fes_values = hp_fes_values.get_present_fe_values();
        
        //int celllevel = cell->level();//0*int(1.0/cell->diameter());
        mhdeq->useNRLinearization(cell->level()>linLevel);
        
        fe_values.get_function_values(old_solution, qpVals[d].old_sv);
        fe_values.get_function_gradients(old_solution, qpVals[d].old_sg);
        fe_values.get_function_values(lin_solution, qpVals[d].lin_sv);
        fe_values.get_function_gradients(lin_solution, qpVals[d].lin_sg);
        
        fes_values.get_function_values(eta,qpVals[d].eta_v);
        fes_values.get_function_gradients(eta, qpVals[d].eta_g);

        right_hand_side.vector_value_list(fe_values.get_quadrature_points(),
                                           qpVals[d].rhs_values);

        // Then assemble the entries of the local stiffness matrix and right
        // hand side vector.
        for(unsigned int q_point=0; q_point<fe_values.n_quadrature_points; ++q_point){
          
          mhdeq->set_state_vector_for_qp(qpVals[d].lin_sv, qpVals[d].lin_sg, qpVals[d].old_sv,
                                         qpVals[d].old_sg, qpVals[d].eta_v, qpVals[d].eta_g, q_point);
          
          mhdeq->setFEvals(fe_values,dofs_per_cell,q_point);
          
          mhdeq->set_operator_matrixes(operator_matrixes, dofs_per_cell);
          
          mhdeq->set_rhs(cell_rhs_lin);
          for(unsigned int i=0; i<dofs_per_cell/Nv; i++){
            for(unsigned int j=0; j<dofs_per_cell/Nv; j++){
              for(unsigned int k=0; k<Nv; k++){
                for(unsigned int l=0; l<Nv; l++){
                  for(unsigned int m=0; m<Ne; m++){
                      cell_matrix(i*Nv+k,j*Nv+l) +=
                          operator_matrixes[i](m,k)*
                          operator_matrixes[j](m,l)*
                          weights[m]*fe_values.JxW(q_point);
                  }
                }
              }
            }
            // Assembling the right hand side
            for(unsigned int l=0; l<Ne; l++){
              for(unsigned int k=0; k<Nv; k++){
                //pcout<<k<<" "<<l<<' '<<cell_rhs(i*Ne+k)<<"\n";
                cell_rhs(i*Nv+k) += operator_matrixes[i](l,k)*
                      (qpVals[d].rhs_values[q_point](l)+cell_rhs_lin(l))*
                      weights[l]*fe_values.JxW(q_point);
              }
                //pcout<<"i:"<<i*Ne<<" "<<k<<' '<<cell_rhs(i*Ne+k)<<"\n";
            }

          } // end of i-loop
        }   // end of q-points

        // Boundary conditions
        for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
          if (cell->face(face_number)->at_boundary()){
              hp_fe_face_values.reinit(cell, face_number);
              hp_fes_face_values.reinit(cells, face_number);
              
              const FEFaceValues<dim> &fe_face_values = hp_fe_face_values.get_present_fe_values();
              const FEFaceValues<dim> &fes_face_values = hp_fes_face_values.get_present_fe_values();
              
              fe_face_values.get_function_values(old_solution, qpVals[d].old_svf);
              fe_face_values.get_function_gradients(old_solution, qpVals[d].old_sgf);
              fe_face_values.get_function_values(lin_solution, qpVals[d].lin_svf);
              fe_face_values.get_function_gradients(lin_solution, qpVals[d].lin_sgf);

              fes_face_values.get_function_values(eta, qpVals[d].eta_vf);
              fes_face_values.get_function_gradients(eta, qpVals[d].eta_gf);

              right_hand_side.vector_value_list(fe_face_values.get_quadrature_points(),
                                           qpVals[d].rhs_values);
              initial_values.vector_value_list(fe_face_values.get_quadrature_points(),
                                          qpVals[d].init_values);
              
              for(unsigned int q_point=0; q_point<fe_face_values.n_quadrature_points; ++q_point){
                mhdeq->setFEvals(fe_face_values,dofs_per_cell,q_point); // call it before BC
                
                // get pointer to function which setup BC for given type defined in params
                int bi = cell->face(face_number)->boundary_indicator();
                
                const Tensor<1,dim> &nrm=fe_face_values.normal_vector(q_point);
                // call BC function and setup state vectors
                (mhdeq->*(mhdeq->BCp[BCmap[bi] ])) (
                            qpVals[d].lin_svf, qpVals[d].lin_sgf, qpVals[d].old_svf, 
                            qpVals[d].old_sgf, qpVals[d].init_values, qpVals[d].eta_vf,
                            qpVals[d].eta_gf,  nrm, q_point);
                
                mhdeq->setFEvals(fe_face_values,dofs_per_cell,q_point);
                
                mhdeq->set_operator_matrixes(operator_matrixes, dofs_per_cell);
          
                mhdeq->set_rhs(cell_rhs_lin);
                for(unsigned int i=0; i<dofs_per_cell/Nv; i++){ // unfortunately we have to go all over the dofs_per_cell
                  for(unsigned int j=0; j<dofs_per_cell/Nv; j++){ //   dofs_per_face do no cointains proper basis fce
                    for(unsigned int k=0; k<Nv; k++){
                      for(unsigned int l=0; l<Nv; l++){
                        for(unsigned int m=0; m<Ne; m++){
                            cell_matrix(i*Nv+k,j*Nv+l) +=
                                operator_matrixes[i](m,k)*
                                operator_matrixes[j](m,l)*
                                weights[m]*fe_face_values.JxW(q_point);
                        }
                      }
                    }
                  }
                  // Assembling the right hand side
                  for(unsigned int l=0; l<Ne; l++)
                    for(unsigned int k=0; k<Nv; k++){
                      cell_rhs(i*Nv+k) += operator_matrixes[i](l,k)*
                            (qpVals[d].rhs_values[q_point](l)+cell_rhs_lin(l))*
                            weights[l]*fe_face_values.JxW(q_point);
                    }
                        
                }  // end of i
              }  // end of q_point
          } // end of boundary


        // The transfer from local degrees of freedom into the global matrix
        local_dof_indices.resize(dofs_per_cell);
        cell->get_dof_indices(local_dof_indices); // TODO: Matrix is symetric, optimalization is needed!
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
    SolverControl    solver_control(CGMmaxIt, CGMprec); // 1000, 1e-14
    
    LA::SolverCG solver(solver_control, mpi_communicator);
    
    LA::MPI::PreconditionAMG preconditioner;
    LA::MPI::PreconditionAMG::AdditionalData data;
#ifdef USE_PETSC_LA
    data.symmetric_operator = true;
#else
    / * Trilinos defaults are good * /
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
    double lastErr,err;
    bool overflow;
    double sz[dim],maxSiz=0.0;
    std::vector<unsigned int> rep(dim);

    pcout<<"Initial mesh refinement..."<<std::endl;
    for(unsigned int i=0;i<dim;i++){ 
      sz[i]=std::fabs(boxP1[i]-boxP2[i]);
      if (sz[i]>maxSiz) maxSiz=sz[i];
    }
    // initial refinement of the box is non-homogenous in order to have cube cells (if possible)
    for(unsigned int i=0;i<dim;i++) rep[i]=(unsigned int)(4*sz[i]/maxSiz+0.5);
    GridGenerator::subdivided_hyper_rectangle(triangulation, rep, boxP1, boxP2, true); // set box and colorize boundaries
    triangulation.refine_global(initSplit); // 6
    pcout << "   initial mesh, ndofs: "<< dof_handler.n_dofs() << std::endl;
    mhdeq->setMinh(GridTools::minimal_cell_diameter(triangulation));
    setup_system();
    pcout<<"Refine gradients in initial conditions..."<<std::endl;
    for(unsigned int i=0;i<initRefin;i++){ // 5
      project_initial_conditions();
      mhdeq->checkOverflow(distributed_solution,distributed_solution); //check overflow
      (mhdeq->*(mhdeq->setEta))(distributed_solution,eta,eta_dist);
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
    mhdeq->checkOverflow(distributed_solution,distributed_solution); //check overflow
    solution = distributed_solution;  // copy results to the solution (it includes ghosts)
    mhdeq->checkDt(solution);         // find and set dt
    mhdeq->setNewDt();
    
    pcout << "   First output."<<std::endl;
    old_solution=solution;
    setShockSmoothCoef(); // set refinement coeficients
    output_results(output_counter++);
    for(double time=0.0; time<totalTime; ){
        if (time_step%10==0) pcout << "Time: " << time <<", time step: "<<time_step<< std::endl;

        lin_solution=old_solution;
        lastErr=9e99;
        linReset=iter=0;
        for(;;){ // linearization - do one time step
            assemble_system(iter);
            sitr = solve();
            overflow=mhdeq->checkOverflow(distributed_solution,old_solution);
            solution=distributed_solution;
            mhdeq->checkDt(solution);
            system_rhs=lin_solution;
            system_rhs-=distributed_solution;
            err=system_rhs.linfty_norm();//norm_sqr();
            if (err<linPrec) break;  //linfty_norm
            if ((err>8.*lastErr && iter>1)){                 // linearization do not converge
              //if (err<1e-7*dof_handler.n_dofs()) break; // still ok error...
              mhdeq->setDt(mhdeq->getDt()*0.25);
              lin_solution=old_solution;
              iter=0;
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
//             pcout<< "---t: " << time<<" it: "<<iter<<" cgm:"<<sitr
//                  << " res: "<<err<<
//                  " vmax="<<mhdeq->getVmax()<<" "<<overflow<<std::endl;
        }
        (mhdeq->*(mhdeq->setEta))(distributed_solution,eta,eta_dist);
        if (time_step%1==0) pcout << "No. l. it.: " << iter << 
                                      " No. s. it.:"<< sitr << 
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
          distributed_solution=solution;
          mhdeq->checkOverflow(distributed_solution,solution);
          (mhdeq->*(mhdeq->setEta))(distributed_solution,eta,eta_dist);
          old_solution=distributed_solution;
          pcout << "   refinement, ndofs: "<< dof_handler.n_dofs() << std::endl;
          continue;
        }

       old_solution=solution;
    }
  }
  
  template <int dim>
  void MHDProblem<dim>::project_initial_conditions()
  {
    // QGaussLobatto<dim>  -- qps are in interpolation points
    hp::FEValues<dim> hp_fe_values(fe_collection, quadrature_col_proj,
                             update_values   | update_gradients |
                             update_quadrature_points | update_JxW_values);

    std::vector<types::global_dof_index> local_dof_indices;
    Vector<double> cell_rhs;
    FullMatrix<double> cell_matrix;
    
    system_matrix=0.0;
    system_rhs=0.0;
    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    unsigned int cellNo=0;
    for(; cell!=endc; ++cell,++cellNo)
      if(cell->is_locally_owned()){
        unsigned int d = cell->active_fe_index();  // degree
        const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
        
        cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
        cell_rhs.reinit(dofs_per_cell);
        cell_rhs = 0;
        cell_matrix = 0;
        
        hp_fe_values.reinit(cell);
        const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();
        
        initial_values.vector_value_list(fe_values.get_quadrature_points(),
                                        qpVals[d].rhs_values);
              
        for(unsigned int point=0; point<fe_values.n_quadrature_points; ++point){
          for(unsigned int i=0; i<dofs_per_cell; ++i){
            const unsigned int cmp_i = cell->get_fe().system_to_component_index(i).first;
            cell_rhs(i) += qpVals[d].rhs_values[point][cmp_i]*
                              fe_values.shape_value(i,point)*
                              fe_values.JxW(point);

            for(unsigned int j=0; j<dofs_per_cell; ++j){
              const unsigned int cmp_j = cell->get_fe().system_to_component_index(j).first;
              if (cmp_i==cmp_j)
                cell_matrix(i,j) += fe_values.shape_value(i,point)*
                                  fe_values.shape_value(j,point)*
                                  fe_values.JxW(point);
            }

          }
        }
        local_dof_indices.resize(dofs_per_cell);
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