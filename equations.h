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
    
    void set_state_vector_for_qp(std::vector<Vector<double> >&,
                        std::vector<std::vector<Tensor<1,dim> > >&,
                        std::vector<Vector<double> > &,
                        std::vector<std::vector<Tensor<1,dim> > >&,
//                         std::vector<double > &,  // eta values
//                         std::vector<Tensor<1,dim> >&, // eta gradients
                        const unsigned int);
    void setFEvals(const FEValues<dim>&,
                    const unsigned int,
                    const unsigned int);
    void setFEvals(const FEFaceValues<dim>&,
                    const unsigned int,
                    const unsigned int);
    void set_operator_matrixes(FullMatrix<double> *,
                            const unsigned int);
    void set_rhs(Vector<double>&);
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
    double setEtaConst();
    double setEtaVD();
    double setEtaJ();
    
//     typedef void (MHDequations::*p2setEta)(LA::MPI::Vector &, LA::MPI::Vector &, LA::MPI::Vector &);
    typedef double (MHDequations::*p2setEta)();
    p2setEta setEta;
    
    typedef void (MHDequations::*p2BC)(std::vector<Vector<double> >&,  // lin values
                        std::vector<std::vector<Tensor<1,dim> > >&, // lin gradients
                        std::vector<Vector<double> > &,  // old values
                        std::vector<std::vector<Tensor<1,dim> > >&, // old gradients
                        std::vector<Vector<double> >&, // initial values
//                         std::vector<double > &,  // eta values
//                         std::vector<Tensor<1,dim> >&, // eta gradients
                        const Tensor<1,dim>&,  // normal
                        const unsigned int); // qp
    p2BC        BCp[4];
    void constantBC(std::vector<Vector<double> >&,  // lin values
                        std::vector<std::vector<Tensor<1,dim> > >&, // lin gradients
                        std::vector<Vector<double> > &,  // old values
                        std::vector<std::vector<Tensor<1,dim> > >&, // old gradients
                        std::vector<Vector<double> >&, // initial values
//                         std::vector<double > &,  // eta values
//                         std::vector<Tensor<1,dim> >&, // eta gradients
                        const Tensor<1,dim>&,  // normal
                        const unsigned int); // qp
    void freeBC(std::vector<Vector<double> >&,  // lin values
                        std::vector<std::vector<Tensor<1,dim> > >&, // lin gradients
                        std::vector<Vector<double> > &,  // old values
                        std::vector<std::vector<Tensor<1,dim> > >&, // old gradients
                        std::vector<Vector<double> >&, // initial values
//                         std::vector<double > &,  // eta values
//                         std::vector<Tensor<1,dim> >&, // eta gradients
                        const Tensor<1,dim>&,  // normal
                        const unsigned int); // qp
    void noFlowBC(std::vector<Vector<double> >&,  // lin values
                        std::vector<std::vector<Tensor<1,dim> > >&, // lin gradients
                        std::vector<Vector<double> > &,  // old values
                        std::vector<std::vector<Tensor<1,dim> > >&, // old gradients
                        std::vector<Vector<double> >&, // initial values
//                         std::vector<double > &,  // eta values
//                         std::vector<Tensor<1,dim> >&, // eta gradients
                        const Tensor<1,dim>&,  // normal
                        const unsigned int); // qp
    void mirrorBC(std::vector<Vector<double> >&,  // lin values
                        std::vector<std::vector<Tensor<1,dim> > >&, // lin gradients
                        std::vector<Vector<double> > &,  // old values
                        std::vector<std::vector<Tensor<1,dim> > >&, // old gradients
                        std::vector<Vector<double> >&, // initial values
//                         std::vector<double > &,  // eta values
//                         std::vector<Tensor<1,dim> >&, // eta gradients
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
//     double               F[3][Ne];
//     double               Flx[3][Ne];
    double               vl[Nv],vo[Nv];
    double               dvx[3][Nv],dox[3][Nv];
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
    
    mapDoFs&      stv2dof;
  };
  
  //template class MHDequations<1>;  // create code for 1D to be able link it....
  template class MHDequations<2>;
  template class MHDequations<3>;

} // end of namespace mhd

#endif //_EQUATIONS_