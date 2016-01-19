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
    InitialValues();
    
    void setInitialCondition(int);
    
    typedef void (InitialValues::*p2vctVal)(const Point<dim>&,
                                        Vector<double>&) const;
    p2vctVal vectorValue;

    void mhdBlast(const Point<dim>&, Vector<double>&) const;
    void harris(const Point<dim>&, Vector<double>&) const;
    
    virtual void vector_value(const Point<dim> &, Vector<double>&) const;
                                        
    virtual void vector_value_list(const std::vector<Point<dim> >&,
                                    std::vector<Vector<double> >&) const;
  };
  
  //template class InitialValues<1>;  // parallel version of deal does not support 1D
  template class InitialValues<2>;   // create code for 1D to be able link it....
  template class InitialValues<3>;
  
} // end of namespace mhd


#endif  // _INITCON_