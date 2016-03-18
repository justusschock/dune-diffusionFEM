#ifdef HAVE_CONFIG_H

#include "config.h"
#endif

#include <iostream>
#include <dune/fem/misc/mpimanager.hh>
#include <dune/grid/io/file/dgfparser/dgfyasp.hh>

#include "poissonPDE.hh"



int main(int argc, char** argv)
{
    try{
        // Maybe initialize MPI
        //Dune::MPIHelper& helper = Dune::MPIHelper::instance(argc, argv);
        Dune::Fem::MPIManager::initialize(argc, argv);

        enum {cos, sin};

        const int dim = 3;

        Dune::FieldVector<double, dim> L(1.0);
        Dune::array<int, dim> N (Dune::fill_array<int, dim>(1));
        std::bitset<dim> B(false);
        Dune::YaspGrid<dim> grid(L,N,B,false);



        solvePoissonPDE<Dune::YaspGrid<dim>>(grid, 1, 0, 2, sin);
        

        return 0;
    }

    catch (std::string &e){
        std::cerr << e << std::endl;
    }
    catch (Dune::Exception &e){
        std::cerr << "Dune reported error: " << e << std::endl;
    }
    catch (std::exception &e){
        std::cerr << "STL reported error: " << e.what() << std::endl;
    }
    catch (...){
        std::cerr << "Unknown exception thrown!" << std::endl;
    }
}

