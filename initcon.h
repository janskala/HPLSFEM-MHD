#ifndef _INITCON_
#define _INITCON_

#include <deal.II/numerics/vector_tools.h>
#include "equations.h"

namespace mhd
{
  
  // Initial conditions
  template <int dim>
  class InitialValues : public Function<dim>
  {
  public:
    InitialValues(Params &);
    virtual void point_value(const Point<dim> &, Vector<double>&) const{};
    virtual void vector_value_list(const std::vector<Point<dim> >&,
                                    std::vector<Vector<double> >&) const override;
  protected:
    double     GAMMA;
  };
  
  // MHD Blast
  template <int dim>
  class mhdBlast : public InitialValues<dim>
  {
  public:
    mhdBlast(Params &);
    
    void point_value(const Point<dim> &, Vector<double>&) const override;
  
  private:
    Point<dim> box[2];
  };
  
  // Harris Current Sheet
  template <int dim>
  class harris : public InitialValues<dim>
  {
  public:
    harris(Params &);
    
    void point_value(const Point<dim> &, Vector<double>&) const override;
  
  private:                                  
    Point<dim> box[2];
  };
  
  // Titov & Demoulin flux rope
  template <int dim>
  class TitovDemoulin : public InitialValues<dim>
  {
  public:
    TitovDemoulin(Params &);
    
    void point_value(const Point<dim> &, Vector<double>&) const override;
  
  private:
    Point<dim> box[2];
    double beta;
    double Lg;
    double invLg;
    double N_t;
    double R_t;
    double d2R_t;
    double L2R_t;
    double q_mag;
    double iSgn;
    double heliFactor;
    double Tc2Tp;
    double t_rho;
    double densGrad;
  };
  
  // Debugging 
  template <int dim>
  class debug : public InitialValues<dim>
  {
  public:
    debug(Params &);
    
    void point_value(const Point<dim> &, Vector<double>&) const override;
  };
  
  
  //template class InitialValues<1>;  // parallel version of deal does not support 1D
  template class InitialValues<2>;   // create code for 1D to be able link it....
  template class InitialValues<3>;
  template class mhdBlast<2>;
  template class mhdBlast<3>;
  template class harris<2>;
  template class harris<3>;
  template class TitovDemoulin<2>;
  template class TitovDemoulin<3>;
  template class debug<2>;
  template class debug<3>;
  
} // end of namespace mhd


#endif  // _INITCON_
