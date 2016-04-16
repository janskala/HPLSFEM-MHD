#include "problem.h"

namespace mhd
{

  template <int dim>
  void MHDProblem<dim>::refine_grid_simple()
  {
    // simple rule for mesh refinement
#ifdef USE_TIMER
    TimerOutput::Scope t(computing_timer, "refine-simple");
#endif
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate(dof_handler,
                                        QGauss<dim-1>(fe.degree+1),
                                        typename FunctionMap<dim>::type(),
                                        distributed_solution,
                                        estimated_error_per_cell);
     parallel::distributed::GridRefinement::
     refine_and_coarsen_fixed_fraction(triangulation,
                                    estimated_error_per_cell,
                                    0.8, 0.05);
    // bound maximum and minimum refinement levels
    typename Triangulation<dim>::active_cell_iterator cell;
    if(triangulation.n_levels() > meshMaxLev)
      for(cell = triangulation.begin_active(meshMaxLev);
          cell != triangulation.end(); ++cell)
        cell->clear_refine_flag();
    for(cell = triangulation.begin_active(meshMinLev);
        cell != triangulation.end_active(meshMinLev); ++cell)
      cell->clear_coarsen_flag();
      
    transfer_solution();
    
    //cell = triangulation.begin_active(triangulation.n_levels()-1);
    mhdeq->setMinh(GridTools::minimal_cell_diameter(triangulation)/FEO);
  }
  
  template <int dim>
  void MHDProblem<dim>::refine_grid_rule()
  {
    // refine cell according to the shock indicator
#ifdef USE_TIMER
    TimerOutput::Scope t(computing_timer, "refine-shocks");
#endif
    typename Triangulation<dim>::active_cell_iterator
            cell = triangulation.begin_active(),
            endc = triangulation.end();
    for (unsigned int cell_no=0; cell!=endc; ++cell,++cell_no)
      if(cell->is_locally_owned()){
        cell->clear_coarsen_flag();
        cell->clear_refine_flag();
        if ((cell->level() < int(meshMaxLev)) &&
            (shockIndicator(cell_no) > meshRefGrad))
          cell->set_refine_flag();
        else if ((cell->level() > int(meshMinLev)) &&
                (shockIndicator(cell_no) < meshCoaGrad))
          cell->set_coarsen_flag();
      }
      
    transfer_solution();
    
    //cell = triangulation.begin_active(triangulation.n_levels()-1);
    mhdeq->setMinh(GridTools::minimal_cell_diameter(triangulation)/FEO);
  }
  
  template <int dim>
  void MHDProblem<dim>::transfer_solution()
  {
    // trasfer solution from old mesh to the new one
#ifdef USE_TIMER
    TimerOutput::Scope t(computing_timer, "solution transfer");
#endif
    parallel::distributed::SolutionTransfer<dim, LA::MPI::Vector> 
        solution_transfer(dof_handler);
    
    triangulation.prepare_coarsening_and_refinement();
    solution_transfer.prepare_for_coarsening_and_refinement(solution);
    
    triangulation.execute_coarsening_and_refinement();  // Actual mesh refinement
    setup_system();  // resize vectors
    
    LA::MPI::Vector distributed_solution(locally_owned_dofs, mpi_communicator);
    solution_transfer.interpolate(distributed_solution);
    
    solution=distributed_solution;
    //old_solution=solution;
  }
  
  template <int dim>
  void MHDProblem<dim>::setShockSmoothCoef()
  {
    // trasfer solution from old mesh to the new one
#ifdef USE_TIMER
    TimerOutput::Scope t(computing_timer, "set Shock Smooth Coef");
#endif
    float locShockIndx[Nv], locMean[Nv], max[Nv], min[Nv], grad;
    
    QGaussLobatto<dim> quadrature(FEO+1);// QGaussLobatto<dim>  -- qps are in interpolation points
    //QGauss<dim-1> face_quadrature_formula(FEO+1);
    FEValues<dim> fe_values(fe, quadrature, update_values | update_gradients | update_quadrature_points);
    /*FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
                                  update_values | update_gradients | update_quadrature_points |
                                  update_normal_vectors);*/
    
    const unsigned int n_q_points    = fe_values.n_quadrature_points;
    //const unsigned int n_f_q_points  = face_quadrature_formula.size();
    
    std::vector<Vector<double> > values(n_q_points, Vector<double>(Nv));
    std::vector<std::vector<Tensor<1,dim> > >  gadients(n_q_points,
                                        std::vector<Tensor<1,dim> > (Nv));
    /*std::vector<Vector<double> > fvalues(n_f_q_points, Vector<double>(Ne));
    std::vector<std::vector<Tensor<1,dim> > >  fgadients(n_f_q_points,
                                        std::vector<Tensor<1,dim> > (Nv));*/
    
    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    unsigned int cellNo=0;
    for(; cell!=endc; ++cell,++cellNo)
      if(cell->is_locally_owned()){
        /*for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face){
          fe_face_values.reinit(cells, face);
              
          fe_face_values.get_function_values(solution, fvalues);
          fe_face_values.get_function_gradients(solution, fgadients);
          
          for(unsigned int q_point=0; q_point<n_f_q_points; ++q_point){
          }
        }*/
        // std::cout<<triangulation.locally_owned_subdomain()<<" start proj "<<cellNo<<std::endl;
        fe_values.reinit(cell);
        
        fe_values.get_function_values(solution, values);
        fe_values.get_function_gradients(solution, gadients);
        
        for(unsigned int l=0; l<Nv; l++){
          /*locShockIndx[l]= */locMean[l]=max[l]=-9e99;
          min[l]=9e99;
        }
        
        for(unsigned int point=0; point<n_q_points; ++point)
          for(unsigned int l=1; l<Nv; l++){
            locMean[l]+=values[point][l];
            if (values[point][l]>max[l]) max[l]=values[point][l];
            if (values[point][l]<min[l]) min[l]=values[point][l];
          }
            
        //for(unsigned int point=0; point<n_q_points; ++point){
          // integrate gradients over cell for shock recognition
          for(unsigned int l=1; l<Nv; l++){
              //grad=values[point][l]-locMean[l]/n_q_points;
              grad=max[l]-min[l];
              locShockIndx[l]=std::fabs(grad);
          }
        //}

        // find the greatest gradient
        grad=-9e99;
        for(unsigned int l=1; l<Nv; l++){
          if (locShockIndx[l]<1e-8) locShockIndx[l]=1e-8;
          double hlp=std::log(locShockIndx[l]*cell->diameter()); // /std::pow(cell->diameter(),dim)
          
          //if (hlp<1.0) hlp=1.0; // for smooth region
          if (hlp>50.0) hlp=50.0; // remove extrema
          if (hlp>grad) grad=hlp; // maximum on cell
        }
        // shock indicator is given by average value of gradient on the cell
        //if (grad>1)
//         std::cout<<triangulation.locally_owned_subdomain()<<" cellNo: "<<cellNo
//                  <<", g="<<grad<<" d="<<cell->diameter()<<std::endl;
        shockIndicator[cellNo]=grad;
      }
  }

} // end of namespace mhd