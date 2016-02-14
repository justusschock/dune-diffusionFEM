#ifdef HAVE_CONFIG_H
#endif
#include <iostream>
#include <dune/common/parallel/mpihelper.hh> // An initializer of MPI
#include <dune/grid/yaspgrid.hh>

#include "diffusionPDE.hh"

int main(int argc, char** argv)
{
    try{
        // Maybe initialize MPI
        Dune::MPIHelper& helper = Dune::MPIHelper::instance(argc, argv);

        const int dim = 2;
        Dune::FieldVector<double, dim> L(1.0);
        Dune::array<int, dim> N (Dune::fill_array<int, dim>(1));
        std::bitset<dim> B(false);
        Dune::YaspGrid<dim> grid(L,N,B,false);


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

