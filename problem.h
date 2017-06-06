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
#include <deal.II/distributed/solution_transfer.h>

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
//#include <deal.II/lac/compressed_simple_sparsity_pattern.h>
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

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>  // vector-valued finite elements
#include <deal.II/fe/fe_q.h>       // Q1 elements - H1
#include <deal.II/fe/fe_dgq.h>     // Discontinuous elements 
#include <deal.II/fe/fe_nedelec.h> // Nedelec elements - Hcurl -- up to 7th order
//#include <deal.II/fe/fe_abf.h>     // Arnold-Boffi-Falk (ABF) elements - Hdiv -- only supports 2nd order!
#include <deal.II/fe/fe_raviart_thomas.h>  // Raviart-Thomas elements

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
    void setup_parameters(Params&);
    void setup_system();
    void assemble_system(const int);
    
    
    unsigned int solve();
    void refine_grid_simple();
    void refine_grid_rule();
    void setShockSmoothCoef();
    void transfer_solution();
    void output_results(const unsigned int cycle);
    void project_initial_conditions();  // implemented in initcon.cc
    void project_init_cnd_vecpot();     // implemented in initcon.cc
    void corrections();
    void void_step();
    void setDofMapping();

    MPI_Comm             mpi_communicator;
    
    parallel::distributed::
      Triangulation<dim> triangulation;
    DoFHandler<dim>      dof_handler;
//     DoFHandler<dim>      dof_handler_s;
    
    int                  FEO;
    FESystem<dim>        fe;
//     FE_Q<dim>            fes;  // FE for single variable

    ConstraintMatrix     constraints;
    
    FullMatrix<double>*  operator_matrixes;
    std::vector<Vector<double> >*  DIRKv;
    std::vector<std::vector<Tensor<1,dim> > >*  DIRKg;
    
    IndexSet             locally_owned_dofs;
    IndexSet             locally_relevant_dofs;
//     IndexSet             local_dofs;  // eta dofs
//     IndexSet             local_relevant_dofs;  // eta

    LA::MPI::SparseMatrix system_matrix;

    LA::MPI::Vector*      DIRK;  // Diagonal Implicit Runge-Kutta y_1 ... y_4 (up to 4th stage)
    LA::MPI::Vector       solution;
    LA::MPI::Vector       old_solution;
    LA::MPI::Vector       lin_solution;
    LA::MPI::Vector       system_rhs;
    LA::MPI::Vector       residue;
    LA::MPI::Vector       distributed_solution; // for solver
//     LA::MPI::Vector       eta,eta_dist;
    Vector<float>         shockIndicator;
    //BlockVector<float>   shockWeights;
    Vector<float>         shockWeights;
    
    void thetaMethod(unsigned int &, unsigned int &);
    void DIRKmethod(unsigned int &, unsigned int &);
    typedef void (MHDProblem::*p2TimeStepInt)(unsigned int&, unsigned int&);
    p2TimeStepInt timeStepInt;
    
    ConditionalOStream   pcout;
#ifdef USE_TIMER
    TimerOutput          computing_timer;
#endif   
    MHDequations<dim>*   mhdeq;
    InitialValues<dim>   initial_values;
    
    int                  BCmap[6]; 
    Point<dim> boxP1,boxP2;  // Box definition
    
    double totalTime;
    double outputFreq;
    double CGMprec;
    double linPrec;
    int CGMmaxIt;
    unsigned int linmaxIt;
    int linLevel;
    
    unsigned int meshMinLev;
    unsigned int meshMaxLev;
    double meshRefGrad;
    double meshCoaGrad;
    unsigned int initSplit;
    unsigned int initRefin;
    unsigned int intMethod;
    unsigned int gausIntOrd;
    unsigned int initCond;
    
    mapDoFs stv2dof;    // state vectors to the dofs and dofs to state vectors
    
    //class Postprocessor;  // the derived quatities for output
  };
  
  //template class MHDProblem<1>;  // parallel version of deal does not support 1D
  template class MHDProblem<2>;  // create code for 1D to be able link it....
  template class MHDProblem<3>;
  
} // end of namespace mhd

#endif //_PROBLEM_
