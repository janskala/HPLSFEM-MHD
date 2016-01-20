#ifndef _PROBLEM_
#define _PROBLEM_

#define USE_TIMER
#ifdef USE_TIMER
#include <deal.II/base/timer.h>
#endif

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
//#include <deal.II/distributed/solution_transfer.h>

#define USE_PETSC_LA
namespace LA
{
#ifdef USE_PETSC_LA
  using namespace dealii::LinearAlgebraPETSc;
#else
  using namespace dealii::LinearAlgebraTrilinos;
#endif
}

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/compressed_simple_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>  // vector-valued finite elements
#include <deal.II/fe/fe_q.h>       // Q1 elements
#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/fe_values.h>

#include "equations.h"
#include "initcon.h"
#include "params.h"

namespace mhd
{
  using namespace dealii;

  template <int dim>
  class MHDProblem
  {
  public:
    MHDProblem(Params&);
    ~MHDProblem();
    void run();

  private:
    void setup_system();
    void assemble_system(const int);
    
    
    unsigned int solve();
    void refine_grid_simple();
    void refine_grid_rule();
    void setShockSmoothCoef();
    void transfer_solution();
    void output_results(const unsigned int cycle);
    void project_initial_conditions();

    MPI_Comm             mpi_communicator;
    
    parallel::distributed::
      Triangulation<dim> triangulation;
      
    hp::DoFHandler<dim>      dof_handler;
    hp::DoFHandler<dim>      dof_handler_s;
    
    hp::FECollection<dim>    fe_collection;
    hp::FECollection<dim>    fe_collection_s;
    
    hp::QCollection<dim>     quadrature_collection;
    hp::QCollection<dim-1>   face_quadrature_collection;
    hp::QCollection<dim>     quadrature_col_proj;
    
    //DoFHandler<dim>      dof_handler;
    //DoFHandler<dim>      dof_handler_s;
    
    FESystem<dim>**      fesys;
    //FE_Q<dim>            fes;  // FE for single variable

    ConstraintMatrix     constraints;
    
    IndexSet             locally_owned_dofs;
    IndexSet             locally_relevant_dofs;
    IndexSet             local_dofs;  // eta dofs
    IndexSet             local_relevant_dofs;  // eta

    LA::MPI::SparseMatrix system_matrix;

    LA::MPI::Vector       solution;
    LA::MPI::Vector       old_solution;
    LA::MPI::Vector       lin_solution;
    LA::MPI::Vector       system_rhs;
    LA::MPI::Vector       residue;
    LA::MPI::Vector       distributed_solution; // for solver
    LA::MPI::Vector       eta,eta_dist;
    Vector<float>         shockIndicator;
    //BlockVector<float>   shockWeights;
    Vector<float>         shockWeights;
    
    FullMatrix<double>   *operator_matrixes;
    
    ConditionalOStream   pcout;
#ifdef USE_TIMER
    TimerOutput          computing_timer;
#endif   
    MHDequations<dim>*   mhdeq;
    InitialValues<dim>   initial_values;
    
    const static int     FEO=2;
    
    int                  BCmap[6]; 
    Point<dim> boxP1,boxP2;  // Box definition
    
    double totalTime;
    double outputFreq;
    double CGMprec;
    double linPrec;
    int CGMmaxIt;
    unsigned int linmaxIt;
    int linLevel;
    unsigned int afDegMin,afDegMax;   // max and min degrees of aprox. fce.
    
    unsigned int meshMinLev;
    unsigned int meshMaxLev;
    double meshRefGrad;
    double meshCoaGrad;
    unsigned int initSplit;
    unsigned int initRefin;
    
    typedef struct{
      std::vector<Vector<double> >  old_sv,lin_sv,old_svf,lin_svf,rhs_values,init_values;
      std::vector<std::vector<Tensor<1,dim> > >  old_sg,lin_sg,old_sgf,lin_sgf;
      
      std::vector<double>  eta_v,eta_vf;
      std::vector<Tensor<1,dim> >  eta_g,eta_gf;
    } QPVals;
    
    QPVals*     qpVals;
  };
  
  //template class MHDProblem<1>;  // parallel version of deal does not support 1D
  template class MHDProblem<2>;  // create code for 1D to be able link it....
  template class MHDProblem<3>;
  
} // end of namespace mhd

#endif //_PROBLEM_