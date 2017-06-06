#include <fstream>

#include "problem.h"
#include "postprocessor.h"

namespace mhd
{
  
  template <int dim>
  void MHDProblem<dim>::output_results(const unsigned int fileNum)
  {
#ifdef USE_TIMER
    TimerOutput::Scope t(computing_timer, "output");
#endif
    const std::string filename = ("results/o" +
                                Utilities::int_to_string (fileNum, 5) +
                                "." +
                                Utilities::int_to_string
                                (triangulation.locally_owned_subdomain(), 4) +
                                ".vtu");
    std::ofstream output(filename);

    Postprocessor<dim> postprocessor; // Needs to be before DataOut which holds poiter to Postprocessor,
    DataOut<dim> data_out;            // otherwise we get error because Postprocessor is destroyed before DataOut
    data_out.attach_triangulation(triangulation);
    
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution,postprocessor);
    
    Vector<float> subdomain(triangulation.n_active_cells());
    for(unsigned int i=0; i<subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");
    data_out.add_data_vector(shockIndicator, "refine");

//     data_out.add_data_vector(dof_handler_s,eta, "eta");
    
    data_out.build_patches(FEO);
    data_out.write_vtu(output);

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0){
      std::vector<std::string> filenames;
      for (unsigned int i=0; i<Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); ++i)
        filenames.push_back (std::string("o") +
                             Utilities::int_to_string (fileNum, 5) +
                             "." +
                             Utilities::int_to_string(i, 4) +
                             ".vtu");
      const std::string
      pvtu_master_filename = ("results/f" +
                              Utilities::int_to_string (fileNum, 5) +
                              ".pvtu");
      std::ofstream pvtu_master (pvtu_master_filename.c_str());
      data_out.write_pvtu_record (pvtu_master, filenames);
      const std::string
      visit_master_filename = ("results/f" +
                               Utilities::int_to_string (fileNum, 5) +
                               ".visit");
      std::ofstream visit_master (visit_master_filename.c_str());
      data_out.write_pvtu_record(visit_master, filenames);
    }
  }
  
} // end of namespace mhd
