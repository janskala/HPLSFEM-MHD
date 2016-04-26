#ifndef _EQUATIONS_
#define _EQUATIONS_

#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/generic_linear_algebra.h>

#define USE_PETSC_LA
namespace LA
{
#ifdef USE_PETSC_LA
  using namespace dealii::LinearAlgebraPETSc;
#else
  using namespace dealii::LinearAlgebraTrilinos;
#endif
}

#include "params.h"

#define Ne  13   // Number of the equations  - MHD + eta + div B
#define Nv  12   // Number of the variables
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
    MHDequations(Params&, mapDoFs&, MPI_Comm);
    ~MHDequations();
    
    struct{
      double ab[5][4][3];
      double tau[5][3];
      const unsigned int orders[5]={2,2,3,3,4};
      const unsigned int stages[5]={1,2,2,3,3};
      const unsigned int maxStageAll=3;
      unsigned int maxStage;
      unsigned int stage;
      unsigned int method;
    }DIRK;
    
    void setDIRKMethod(unsigned int);
    void setDIRKStage(unsigned int);
    void setCNMethod();
    
    void set_state_vector_for_qp(std::vector<Vector<double> >**,
                        std::vector<std::vector<Tensor<1,dim> > >**,
                        const unsigned int);
    void setFEvals(const FEValues<dim>&, const unsigned int);
    void setFEvals(const FEFaceValues<dim>&, const unsigned int);
    
    void calucate_matrix_rhs(FullMatrix<double> *, Vector<double>&);
    void set_operators_full(FullMatrix<double> *);
    void set_operators_diag(FullMatrix<double> *);
    void set_rhs_CN(Vector<double>&);
    void set_rhs_DIRK(Vector<double>&);
    
    void JacobiM(double v[]);
    void dxA(double v[], double dv[][Nv]);
//     void fluxes(double v[]);
    bool checkOverflow(LA::MPI::Vector&, LA::MPI::Vector&);
    void checkDt(LA::MPI::Vector&);
    void checkDt();
    void setNewDt();
    double* getWeights();
    double getDt();
    void setMinh(double minh);
    void setDt(double);
    double getVmax();
    void useNRLinearization(bool);
//     void setEtaConst(LA::MPI::Vector &, LA::MPI::Vector &, LA::MPI::Vector &);
//     void setEtaVD(LA::MPI::Vector &, LA::MPI::Vector &, LA::MPI::Vector &);
//     void setEtaJ(LA::MPI::Vector &, LA::MPI::Vector &, LA::MPI::Vector &);
    double setEtaConst(int);
    double setEtaVD(int);
    double setEtaJ(int);
    
//     typedef void (MHDequations::*p2setEta)(LA::MPI::Vector &, LA::MPI::Vector &, LA::MPI::Vector &);
    
    typedef void (MHDequations::*p2BC)(std::vector<Vector<double> >*[],
                        std::vector<std::vector<Tensor<1,dim> > >*[],
                        std::vector<Vector<double> >&, // initial values
                        const Tensor<1,dim>&,  // normal
                        const unsigned int); // qp
    p2BC        BCp[4];
    void constantBC(std::vector<Vector<double> >*[],
                    std::vector<std::vector<Tensor<1,dim> > >*[],
                    std::vector<Vector<double> >&, // initial values
                    const Tensor<1,dim>&,  // normal
                    const unsigned int); // qp
    void freeBC(std::vector<Vector<double> >*[],
                std::vector<std::vector<Tensor<1,dim> > >*[],
                std::vector<Vector<double> >&, // initial values
                const Tensor<1,dim>&,  // normal
                const unsigned int); // qp
    void noFlowBC(std::vector<Vector<double> >*[],
                  std::vector<std::vector<Tensor<1,dim> > >*[],
                  std::vector<Vector<double> >&, // initial values
                  const Tensor<1,dim>&,  // normal
                  const unsigned int); // qp
    void mirrorBC(std::vector<Vector<double> >*[],
                  std::vector<std::vector<Tensor<1,dim> > >*[],
                  std::vector<Vector<double> >&, // initial values
                  const Tensor<1,dim>&,  // normal
                  const unsigned int); // qp
  
    void reinitFEval();
    
    // RHS of MHD equations
    class RightHandSide :  public Function<dim>
    {
    public:
      RightHandSide();

      virtual void vector_value(const Point<dim> &p,
                                Vector<double>   &values) const;

      virtual void vector_value_list(const std::vector<Point<dim> > &points,
                                      std::vector<Vector<double> >   &value_list) const;
    };
    RightHandSide rhs;
    
  private:
    MPI_Comm            mpi_communicator;
    
    double               A[3][Ne][Nv];
    double               B[Ne][Nv];
    double               V[2+3][Nv];
    double               G[2+3][3][Nv];
    double               *fev[Nv],*feg[3][Nv];
    double weights[Ne]={1e4,1.0,1.0, 1.0,1.0,1.0, 1.0,1.0, 1e-0, 1e-0,1e-0, 1e0,1e0};
    
    double        dt;                    // time step size
    double        newdt;
    double        theta=0.6;
    double        GAMMA=5.0/3.0;
    double        gravity[3];
    int           ETAmet;               // which method is used to set resistivity
    double        ETA,ETAg[3],ETAmax;
    double        ETApar1,ETApar2;
    double        CFL=0.2;
    
    double        vmax;
    double        hmin;
    
    bool          NRLin;               // Newton-Raphson linearization will be used
    bool          NRLinBck;            // Used in DIRK for backup NRLin in last part of DIRK
    
    mapDoFs&      stv2dof;
    
    typedef void (MHDequations::*p2clcMat)(FullMatrix<double> *);
    p2clcMat clcMat;
    typedef void (MHDequations::*p2clcRhs)(Vector<double>&);
    p2clcRhs clcRhs;
    
    typedef double (MHDequations::*p2setEta)(int);
    p2setEta setEta;
  };
  
  //template class MHDequations<1>;  // create code for 1D to be able link it....
  template class MHDequations<2>;
  template class MHDequations<3>;

} // end of namespace mhd

#endif //_EQUATIONS_