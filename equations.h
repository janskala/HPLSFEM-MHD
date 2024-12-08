#ifndef _EQUATIONS_
#define _EQUATIONS_

#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/generic_linear_algebra.h>

namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
  !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
  using namespace dealii::LinearAlgebraPETSc;
#  define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
  using namespace dealii::LinearAlgebraTrilinos;
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA

#include "params.h"

#define Ne  12   // Number of the equations  - MHD + div B
#define Nv  11   // Number of the variables
#define Nt  8    // Number of time dependent equations

namespace mhd
{
  using namespace dealii;
  
  typedef struct{  // mapping from dof to system component and vice versa
  std::vector<unsigned int> cmpInx;
  std::vector<unsigned int> dof;
  std::vector<std::array<unsigned int, Nv>> stateD; // stores dofs for state vector for operator
  unsigned int Ndofs;
  unsigned int Nstv;
  } mapDoFs;
  
  template <int dim>
  class MHDequations
  {
  public:
    MHDequations(Params&, MPI_Comm);
    ~MHDequations();
    
    struct{
      double ab[5][4][3];
      double tau[5][3];
      const unsigned int orders[5]={2,2,3,3,4};
      const unsigned int stages[5]={1,2,2,3,3};
      const unsigned int maxStageAll=3;
      int maxStage;
      int stage;
      int method;
    }DIRK;
    
    void setDIRKMethod(unsigned int);
    void setDIRKStage(unsigned int);
    void setThetaMethod();
    
    void dumpenOsc(const bool isMax, const bool isSteep);
    void set_state_vector_for_qp(std::vector<Vector<double> >**,
                        std::vector<std::vector<Tensor<1,dim> > >**,
                        const unsigned int);
    void setFEvals(const FEValues<dim>&, const unsigned int);
    void setFEvals(const FEFaceValues<dim>&, const unsigned int);
    
    void calucate_matrix_rhs(std::vector<FullMatrix<double>>&, Vector<double>&);
    void set_operators_full(std::vector<FullMatrix<double>>&);
    void set_operators_diag(std::vector<FullMatrix<double>>&);
    void set_rhs_old(Vector<double>&);
    void set_rhs_theta(Vector<double>&);
    void set_rhs_DIRK(Vector<double>&);
    void set_rhs_DIRK_last(Vector<double>&);
    
    void JacobiM(const Vector<double>&);
    void dxA(const Vector<double>& v, const std::vector<Tensor<1,dim>>&);
//     void fluxes(double v[]);
//    bool checkOverflow(LA::MPI::Vector&, LA::MPI::Vector&);
//    void checkDt(LA::MPI::Vector&);
    void checkDt();
    void setNewDt();
    double* getWeights();
    double getDt();
    void setMinh(double minh);
    void setDt(double);
    double getVmax();
    double getEtaMax();
    double getEta(const Vector<double>& V) {return  (this->*setEta)(V);};
    void useNRLinearization(bool);
//     void setEtaConst(LA::MPI::Vector &, LA::MPI::Vector &, LA::MPI::Vector &);
//     void setEtaVD(LA::MPI::Vector &, LA::MPI::Vector &, LA::MPI::Vector &);
//     void setEtaJ(LA::MPI::Vector &, LA::MPI::Vector &, LA::MPI::Vector &);
    void setTimeRef(double*); // reference on the simulation time
    void setBoxRef(Point<dim>*,Point<dim>*);  // reference on the simulation box geometry
    double setEtaConst(const Vector<double>&);
    double setEtaVD(const Vector<double>&);
    double setEtaJ(const Vector<double>&);
    bool isDiagonal();  // for DIRK - last step uses diagonal matrix
    
//     typedef void (MHDequations::*p2setEta)(LA::MPI::Vector &, LA::MPI::Vector &, LA::MPI::Vector &);
    
    typedef void (MHDequations::*p2BC)(std::vector<Vector<double> >*[],
                        std::vector<std::vector<Tensor<1,dim> > >*[],
                        std::vector<Vector<double> >&, // initial values
                        const Tensor<1,dim>&,  // normal
                        const Point<dim>&,     // point
                        const unsigned int); // qp
    p2BC        BCp[5];
    void constantBC(std::vector<Vector<double> >*[],
                    std::vector<std::vector<Tensor<1,dim> > >*[],
                    std::vector<Vector<double> >&, // initial values
                    const Tensor<1,dim>&,  // normal
                    const Point<dim>&,     // point
                    const unsigned int); // qp
    void freeBC(std::vector<Vector<double> >*[],
                std::vector<std::vector<Tensor<1,dim> > >*[],
                std::vector<Vector<double> >&, // initial values
                const Tensor<1,dim>&,  // normal
                    const Point<dim>&,     // point
                const unsigned int); // qp
    void noFlowBC(std::vector<Vector<double> >*[],
                  std::vector<std::vector<Tensor<1,dim> > >*[],
                  std::vector<Vector<double> >&, // initial values
                  const Tensor<1,dim>&,  // normal
                    const Point<dim>&,     // point
                  const unsigned int); // qp
    void mirrorBC(std::vector<Vector<double> >*[],
                  std::vector<std::vector<Tensor<1,dim> > >*[],
                  std::vector<Vector<double> >&, // initial values
                  const Tensor<1,dim>&,  // normal
                    const Point<dim>&,     // point
                  const unsigned int); // qp
    void vortexBC(std::vector<Vector<double> >*[],
                  std::vector<std::vector<Tensor<1,dim> > >*[],
                  std::vector<Vector<double> >&, // initial values
                  const Tensor<1,dim>&,  // normal
                    const Point<dim>&,     // point
                  const unsigned int); // qp
  
    void reinitFEval(const unsigned int);
    
    // RHS of MHD equations
    class RightHandSide :  public Function<dim>
    {
    public:
      RightHandSide();

      virtual void vector_value(const Point<dim> &p,
                                Vector<double>   &values) const override;

      virtual void vector_value_list(const std::vector<Point<dim> > &points,
                                      std::vector<Vector<double> >   &value_list) const override;
    };
    RightHandSide rhs;
    
  private:
    MPI_Comm            mpi_communicator;
    
    double               A[3][Ne][Nv]; // Jacobi matrix from fluxes Fx, Fy and Fz
    double               B[Ne][Nv];    // \sum_i dA_i/dx_i
    Vector<double>*      V[2+3];       // prev state vectors \Psi 0:(t-1), 1:(lin), 2..4:DIRK
    std::vector<Tensor<1,dim>> *   G[2+3];       // d\Psi/dx_i
    double               *fev[Nv],*feg[3][Nv];
    double weights[Ne]={1e4,1.0,1.0, 1.0,1.0,1.0, 1.0,1.0, 1e-0, 1e-0, 1e-0, 1e0};
    
    double        dt;                  // time step size
    double        newdt;
    double        theta=0.6;
    double        GAMMA=5.0/3.0;
    double        gravity[3];
    int           ETAmet;              // which method is used to set resistivity
    double        ETA,ETAg[3],ETAmax;
    double        ETApar1,ETApar2;
    double        CFL=0.2;
    double        *time;               // reference on simulation time
    const Point<dim> *boxP1;
    const Point<dim> *boxP2;
    
    double        vmax;
    double        hmin;
    
    bool          NRLin;               // Newton-Raphson linearization will be used
    bool          NRLinBck;            // Used in DIRK for backup NRLin in last part of DIRK
    bool          diagonal;            // when DIRK is evaluating new values then use diagonal matrix
    
    typedef void (MHDequations::*r2clcMat)(std::vector<FullMatrix<double>>&);
    r2clcMat clcMat;
    typedef void (MHDequations::*p2clcRhs)(Vector<double>&);
    p2clcRhs clcRhs;
    
    typedef double (MHDequations::*p2setEta)(const Vector<double>&);
    p2setEta setEta;
  };
  
  //template class MHDequations<1>;  // create code for 1D to be able link it....
  template class MHDequations<2>;
  template class MHDequations<3>;

} // end of namespace mhd

#endif //_EQUATIONS_
