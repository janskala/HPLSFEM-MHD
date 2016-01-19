#include "problem.h"
#include "params.h"

int main(int argc, char **argv)
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1); // MPI, PETSc init
      dealii::deallog.depth_console(0);

      mhd::Params pars(argc,argv);
      if (pars.getDim()==2){
        mhd::MHDProblem<2> MHD_problem(pars);
        MHD_problem.run();
      }else{
        mhd::MHDProblem<3> MHD_problem(pars);
        MHD_problem.run();
      }
    }
  catch(std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch(...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }

  return 0;
}


