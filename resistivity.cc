#include "equations.h"

namespace mhd
{
  
  template <int dim>
  void MHDequations<dim>::setEtaConst(LA::MPI::Vector& , LA::MPI::Vector &eta, LA::MPI::Vector &eta_dist)
  {
    std::pair<types::global_dof_index, types::global_dof_index> range=eta_dist.local_range();
    for(unsigned int i=range.first;i<range.second;i++){
      eta_dist[i]=ETApar1;
    }
    eta_dist.compress(VectorOperation::insert);  // write changes in parallel vector
    eta=eta_dist;
    ETAmax=ETApar1;
  }
  
  template <int dim>
  void MHDequations<dim>::setEtaJ(LA::MPI::Vector &v, LA::MPI::Vector &eta, LA::MPI::Vector &eta_dist)
  {
    double J;
    ETAmax=0.0;
    std::pair<types::global_dof_index, types::global_dof_index> range=eta_dist.local_range();
    for(unsigned int i=range.first;i<range.second;i++){
      unsigned int j=i*Nv;
      J=std::sqrt(v[j+8]*v[j+8]+v[j+9]*v[j+9]+v[j+10]*v[j+10]);
      if (J>ETApar2){
        eta_dist[i]=(J-ETApar2)*ETApar1;
        if (eta_dist[i]>ETAmax) ETAmax=eta_dist[i];
      }else eta_dist[i]=0.0;
    }
    eta_dist.compress(VectorOperation::insert);  // write changes in parallel vector
    eta=eta_dist;
  }
  
  template <int dim>
  void MHDequations<dim>::setEtaVD(LA::MPI::Vector &v, LA::MPI::Vector &eta, LA::MPI::Vector &eta_dist)
  {
    double VD;
    ETAmax=0.0;
    std::pair<types::global_dof_index, types::global_dof_index> range=eta_dist.local_range();
    for(unsigned int i=range.first;i<range.second;i++){
      unsigned int j=i*Nv;
      VD=std::sqrt(v[j+8]*v[j+8]+v[j+9]*v[j+9]+v[j+10]*v[j+10])/v[j+0];
      if (VD>ETApar2){
        eta_dist[i]=(VD-ETApar2)*ETApar1;
        if (eta_dist[i]>ETAmax) ETAmax=eta_dist[i];
      }else eta_dist[i]=0.0;
    }
    eta_dist.compress(VectorOperation::insert);  // write changes in parallel vector
    eta=eta_dist;
  }
} // end of namespace mhd