#include "params.h"
#include <deal.II/base/mpi.h>

namespace mhd
{

  void Params::print_usage_message()
  {
    static const char *message
      =
        "\n"
        "MHD simulation based on mesh library deal.ii and LSFEM.\n"
        "\n"
        "Usage:\n"
        "    ./HPLSFEM [-p parameter_file] [-h]\n"
        "\n"
        "Parameter sequences in brackets can be omitted. If the parameter file\n"
        "is not specifed then default name params.prm is used.\n"
        "\n"
        "The parameter file has the following format and allows the following\n"
        "values(you can cut and paste this and use it for your own parameter\n"
        "file):\n"
        "\n";
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0){
      std::cout << message;
      prm.print_parameters(std::cout, ParameterHandler::PRM);
    }
  }
  
  
  void Params::declare_parameters()
  {
    prm.enter_subsection("Mesh refinement");
    {
      prm.declare_entry("Refinement method", "1",
                         Patterns::Integer(0,1),
                         "0 - simple method fraction of cells with highest/lowest gradients "
                         "are refined/coarsed. 1 - Refine when gradients are steep enough.");
      prm.declare_entry("Minimum level", "2",
                         Patterns::Integer(0,100),
                         "Minimum level of refinement.");
      prm.declare_entry("Maximum level", "8",
                         Patterns::Integer(0,100),
                         "Maximum level of refinement.");
      prm.declare_entry("Coarsing gradient", "0",
                         Patterns::Double(),
                         "The gradient from which the mesh is coarsed.");
      prm.declare_entry("Refining gradient", "1",
                         Patterns::Double(),
                         "The gradient from which the mesh is refined.");
      prm.declare_entry("Initial division", "2",
                         Patterns::Integer(0,1000),
                         "The number of initial cells sub-domain division/refinement.");
      prm.declare_entry("Initial refinement", "2",
                         Patterns::Integer(0,1000),
                         "The number of refinments applied on initial conditions.");
      prm.declare_entry("aprox deg min", "1",
                         Patterns::Integer(0,10),
                         "Minimum degrees of approximation element function.");
      prm.declare_entry("aprox deg max", "2",
                         Patterns::Integer(0,10),
                         "Maximum degrees of approximation element function.");
      
    }
    prm.leave_subsection();

    prm.enter_subsection("Numerics");
    {
      prm.declare_entry("Time integration", "0",
                         Patterns::Integer(0,5),
                         "Time integration method: 0 - Crank-Nicolson, 1 - DIRK(1,2),"
                         "2 - DIRK(2,2), 3 - DIRK(2,3), 4 - DIRK(3,3), 5 - DIRK(3,4)");
      prm.declare_entry("theta", "0.6",
                         Patterns::Double(0.0,1.0),
                         "Coeficient from theta scheme - sets implicit/explicit formulation.");
      prm.declare_entry("CFL", "0.2",
                         Patterns::Double(0.0,10.0),
                         "Cournat-Fridrich-Lewy coeficient.");
      prm.declare_entry("CGM interations", "1000",
                         Patterns::Integer(0,1000000),
                         "Maximum number of CGS interations");
      prm.declare_entry("CGM tolerance", "1e-12",
                         Patterns::Double(0.0,1.0),
                         "CGM is exited when this number is reached.");
      prm.declare_entry("Number of linearizations", "8",
                         Patterns::Integer(0,100),
                         "Maximum number of interations for linearization.");
      prm.declare_entry("Linearization tolerance", "1e-12",
                         Patterns::Double(0.0,1.0),
                         "Linearization is exited when this number is reached.");
      prm.declare_entry("Simple level", "4",
                         Patterns::Integer(0,100),
                         "From this refinement level Newton-Rapson linearization is used."
                         "Otherwise, a simple linearization method is used.");
      prm.declare_entry("Gauss int ord", "1",
                         Patterns::Integer(0,100),
                         "Gaussian quadrature formula use (aprox deg) + (Gauss int ord)"
                         "order for integration.");
      prm.enter_subsection("LS weights");
      {
        prm.declare_entry("rho", "1.0", Patterns::Double(0,9e99), "Density.");
        prm.declare_entry("Pi", "1.0", Patterns::Double(0,9e99), "Momentum.");
        prm.declare_entry("B", "1.0", Patterns::Double(0,9e99), "Magnetic field.");
        prm.declare_entry("U", "1.0", Patterns::Double(0,9e99), "Total energy.");
        prm.declare_entry("J", "1.0", Patterns::Double(0,9e99), "Current density.");
        prm.declare_entry("div B", "1.0", Patterns::Double(0,9e99), "Additional div B = 0 equation.");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
    
    prm.enter_subsection("Output");
    {
      prm.declare_entry("Output format", "vtk",
                        Patterns::Selection("vtk|h5"),
                        "The output format: vtk or h5");
      prm.declare_entry("Output frequency", "1.0",
                        Patterns::Double(0,9e99),
                        "Determine time between outputs");
    }
    prm.leave_subsection();
    
    prm.enter_subsection("Simulation");
    {
      prm.declare_entry("Dimension", "2",
                         Patterns::Integer(2,3),
                         "Number of dimensions");
      prm.declare_entry("Initial condition", "0",
                         Patterns::Integer(0,3),
                         "Initial condition"
                         "0 - MHD blast, 1 - Harris CS, 2 - Titov&Demoulin, 3 - debug");
      prm.declare_entry("Total time", "1.0",
                         Patterns::Double(0,9e99),
                         "The time when the simulation will be terminated.");
      prm.declare_entry("gamma", "1.6666666666666666666666666666666",
                         Patterns::Double(0,9e99),
                         "Specific heat.");
      prm.declare_entry("eta method", "0",
                         Patterns::Integer(0,2),
                         "The method used for setting resistivity eta. 0 - constant, 1 - switch-on J, 2 - switch-on V_D");
      prm.declare_entry("eta param 1", "1e-3",
                         Patterns::Double(),
                         "First parameter for resistivity model - constant value of resist or steepness");
      prm.declare_entry("eta param 2", "1.0",
                         Patterns::Double(),
                         "Second parameter for resistivity model - threshold");
      prm.enter_subsection("Gravity");
      {
        prm.declare_entry("g_x", "0.0", Patterns::Double(), "Acceleration in x direction");
        prm.declare_entry("g_y", "0.0", Patterns::Double(), "Acceleration in x direction");
        prm.declare_entry("g_z", "0.0", Patterns::Double(), "Acceleration in x direction");
      }
      prm.leave_subsection();
      prm.enter_subsection("Box");
      {
        prm.declare_entry("x_min", "-1.0", Patterns::Double(), "Minimum of x");
        prm.declare_entry("x_max", "1.0", Patterns::Double(), "Maximum of x");
        prm.declare_entry("y_min", "-1.0", Patterns::Double(), "Minimum of y");
        prm.declare_entry("y_max", "1.0", Patterns::Double(), "Maximum of y");
        prm.declare_entry("z_min", "-1.0", Patterns::Double(), "Minimum of z");
        prm.declare_entry("z_max", "1.0", Patterns::Double(), "Maximum of z");
      }
      prm.leave_subsection();
      prm.enter_subsection("BC");
      {
        prm.declare_entry("x_min", "0", Patterns::Integer(0,4), "Minimum of x, 0 - constant BC, 1 - free BC, 2 - no flow BC, 3 - mirror BC, 4 - vortex");
        prm.declare_entry("x_max", "0", Patterns::Integer(0,4), "Maximum of x");
        prm.declare_entry("y_min", "0", Patterns::Integer(0,4), "Minimum of y");
        prm.declare_entry("y_max", "0", Patterns::Integer(0,4), "Maximum of y");
        prm.declare_entry("z_min", "0", Patterns::Integer(0,4), "Minimum of z");
        prm.declare_entry("z_max", "0", Patterns::Integer(0,4), "Maximum of z");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }
  
  int Params::getDim()
  {
    int dim;
    prm.enter_subsection("Simulation");
    {
      dim=prm.get_integer("Dimension");
    }
    prm.leave_subsection();
    return dim;
  }
  
  int Params::getMinElementDegree()
  {
    int deg;
    prm.enter_subsection("Mesh refinement");
    {
      deg=prm.get_integer("aprox deg min");
    }
    prm.leave_subsection();
    return deg;
  }
  
  int Params::getMaxElementDegree()
  {
    int deg;
    prm.enter_subsection("Mesh refinement");
    {
      deg=prm.get_integer("aprox deg max");
    }
    prm.leave_subsection();
    return deg;
  }
  
  void Params::setWeights(double *w)
  {
    prm.enter_subsection("Numerics");
    {
      prm.enter_subsection("LS weights");
      {
        w[0]=prm.get_double("rho");
        w[1]=prm.get_double("Pi");
        w[2]=prm.get_double("Pi");
        w[3]=prm.get_double("Pi");
        w[4]=prm.get_double("B");
        w[5]=prm.get_double("B");
        w[6]=prm.get_double("B");
        w[7]=prm.get_double("U");
        w[8]=prm.get_double("J");
        w[9]=prm.get_double("J");
        w[10]=prm.get_double("J");
        w[12]=prm.get_double("div B");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }

  void Params::setGravity(double *g)
  {
    prm.enter_subsection("Simulation");
    {
      prm.enter_subsection("Gravity");
      {
        g[0]=prm.get_double("g_x");
        g[1]=prm.get_double("g_y");
        g[2]=prm.get_double("g_z");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }
  
    void Params::setBC(int *bc)
  {
    prm.enter_subsection("Simulation");
    {
      prm.enter_subsection("BC");
      {
        bc[0]=prm.get_integer("x_min");
        bc[1]=prm.get_integer("x_max");
        bc[2]=prm.get_integer("y_min");
        bc[3]=prm.get_integer("y_max");
        bc[4]=prm.get_integer("z_min");
        bc[5]=prm.get_integer("z_max");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }
  
  
  Params::Params(const int argc, char *const *argv)
  {
    std::string paramFile("params.prm");
    
    declare_parameters();
    
    std::list<std::string> args;
    for(int i=1; i<argc; ++i)
      args.push_back(argv[i]);
    while(args.size()){
      if(args.front() == std::string("-p")){
        if(args.size() == 1){
          std::cerr << "Error: flag '-p' must be followed by the "
                    << "name of a parameter file."
                    << std::endl;
          print_usage_message();
          exit(1);
        }
        args.pop_front();
        paramFile = args.front();
        args.pop_front();
      }else if(args.front() == std::string("-h")){
        print_usage_message();
        args.pop_front();
      }
    }  // end of while - parameters
    // read parameters from the file
    prm.parse_input(paramFile);
  }
  
  Params::~Params()
  {
  }
  
} // end of namespace mhd
