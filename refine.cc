#include "problem.h"

#include <algorithm>

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
                                        {},
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
    double max[Nv], min[Nv];
    
    QGaussLobatto<dim> quadrature(FEO+1);// QGaussLobatto<dim>  -- qps are in interpolation points
    FEValues<dim> fe_values(fe, quadrature, update_values | update_gradients | update_quadrature_points);
    
    const unsigned int n_q_points    = fe_values.n_quadrature_points;
    //const unsigned int n_f_q_points  = face_quadrature_formula.size();
    
    std::vector<Vector<double> > values(n_q_points, Vector<double>(Nv));
    std::vector<std::vector<Tensor<1,dim> > >  gadients(n_q_points,
                                        std::vector<Tensor<1,dim> > (Nv));
    
    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    unsigned int cellNo=0;
    for(; cell!=endc; ++cell,++cellNo)
        if(cell->is_locally_owned()){
            fe_values.reinit(cell);

            fe_values.get_function_values(solution, values);
            fe_values.get_function_gradients(solution, gadients);

            for(unsigned int l=1; l<7; l++){ // Only B is used for max-grad
                max[l]=-9e99;
                min[l]=9e99;
            } 

            for(unsigned int point=0; point<n_q_points; ++point)
                for(unsigned int l=1; l<7; l++){
                    min[l]=std::min(min[l],values[point][l]);
                    max[l]=std::max(max[l],values[point][l]);
                }

            float grad=-9e99;
            for(unsigned int l=1; l<7; l++){
                double hlp=std::log(1.0+std::fabs(max[l]-min[l]))*std::pow(1.0+cell->diameter(),4);
                if (hlp>grad) grad=hlp; // maximum on cell
            }

            shockIndicator[cellNo]=grad;
        }
  }

} // end of namespace mhd
