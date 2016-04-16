#ifndef _POSTPROCESSOR_
#define _POSTPROCESSOR_

#include <deal.II/numerics/data_out.h>
//#include <deal.II/numerics/data_postprocessor.h>

namespace mhd
{
  using namespace dealii;
    
  template <int dim>
  class Postprocessor : public DataPostprocessor<dim>
  {
  public:
    Postprocessor();
    virtual void
    compute_derived_quantities_vector(const std::vector<Vector<double> >              &,
                                      const std::vector<std::vector<Tensor<1,dim> > > &,
                                      const std::vector<std::vector<Tensor<2,dim> > > &,
                                      const std::vector<Point<dim> >                  &,
                                      const std::vector<Point<dim> >                  &,
                                      std::vector<Vector<double> >                    &) const;
    virtual std::vector<std::string> get_names() const;
    virtual std::vector<DataComponentInterpretation::DataComponentInterpretation>
            get_data_component_interpretation() const;
    virtual UpdateFlags get_needed_update_flags() const;
  private:
    
  };

  //template class Postprocessor<1>;  // parallel version of deal does not support 1D
  template class Postprocessor<2>;   // create code for 2D to be able link it....
  template class Postprocessor<3>;   // as well as for 3D

}
#endif // _POSTPROCESSOR_