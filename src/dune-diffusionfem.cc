#ifdef HAVE_CONFIG_H

#include "config.h"
#endif

#include <iostream>
#include <array>
#include <dune/fem/misc/mpimanager.hh>

#include "poissonPDE.hh"

#define GRIDSELECTOR false

int main(int argc, char** argv)
{
    try{
        // Maybe initialize MPI
        //Dune::MPIHelper& helper = Dune::MPIHelper::instance(argc, argv);
        Dune::Fem::MPIManager::initialize(argc, argv);

        const std::string problemNames [] = {"sin", "cos"};

        if(!GRIDSELECTOR) {
            const int dim = 3;

            Dune::FieldVector<double, dim> L(1.0);
            Dune::array<int, dim> N(Dune::fill_array<int, dim>(1));
            std::bitset<dim> B(false);
            Dune::YaspGrid<dim> grid(L, N, B, false);

            solvePoissonPDE<Dune::YaspGrid<dim>>(grid, 2, 0, 2, 1);

        }
        else {
            // append overloaded parameters from the command line
            Dune::Fem::Parameter::append(argc, argv);

            // append possible given parameter files
            for (int i = 1; i < argc; ++i)
                Dune::Fem::Parameter::append(argv[i]);

            // append default parameter file
            Dune::Fem::Parameter::append("/home/js/dune/dune-diffusionfem/data/parameter");

            // type of hierarchical grid
            typedef Dune::GridSelector::GridType HGridType;

            // create grid from DGF file
            const std::string gridkey = Dune::Fem::IOInterface::defaultGridKey(HGridType::dimension);
            const std::string gridfile = Dune::Fem::Parameter::getValue<std::string>(gridkey);

            // the method rank and size from MPIManager are static
            if (Dune::Fem::MPIManager::rank() == 0)
                std::cout << "Loading macro grid: " << gridfile << std::endl;

            // construct macro using the DGF Parser
            Dune::GridPtr<HGridType> gridPtr(gridfile);
            HGridType &grid = *gridPtr;

            // do initial load balance
            grid.loadBalance();

            // initial grid refinement
            const int level = Dune::Fem::Parameter::getValue<int>("poisson.level");

            // number of global refinements to bisect grid width
            const int refineStepsForHalf = Dune::DGFGridInfo<HGridType>::refineStepsForHalf();

            // refine grid
            grid.globalRefine(level * refineStepsForHalf);

            const int repeats = Dune::Fem::Parameter::getValue<int>("poisson.repeats", 0);
            const int problemNumber = Dune::Fem::Parameter::getEnum("poisson.problem", problemNames, 0);

            solvePoissonPDE<HGridType>(grid, refineStepsForHalf, level, repeats, problemNumber);

        }
        return 0;
    }

    catch (std::string &e){
        std::cerr << e << std::endl;
        return 1;
    }
    catch (Dune::Exception &e){
        std::cerr << "Dune reported error: " << e << std::endl;
        return 1;
    }
    catch (std::exception &e){
        std::cerr << "STL reported error: " << e.what() << std::endl;
        return 1;
    }
    catch (...){
        std::cerr << "Unknown exception thrown!" << std::endl;
        return 1;
    }
}

