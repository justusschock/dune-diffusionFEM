//#pragma once
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <iostream>
#include <time.h>
#include <dune/grid/yaspgrid.hh>
#include <dune/pdelab/finiteelementmap/qkfem.hh>
#include <dune/pdelab/constraints/conforming.hh>
#include <dune/common/parallel/mpihelper.hh> // An initializer of MPI
#include <dune/common/exceptions.hh> // We use exceptions
#include <dune/grid/utility/structuredgridfactory.hh>

#include "poisson.hh"
#include "stationaryDiffusion.hh"
//#include "instationaryDiffusion.h"
#include "PDETestSuite.h"


bool runTests(){

    const int dim = 2;
    stationaryDiffusion::SinkSourceDefault<Dune::YaspGrid<dim>::LeafGridView> sink_term, source_term;
    Test::Suite tests;
    tests.add(std::auto_ptr<Test::Suite>(new PDETestSuite<dim>(sink_term,source_term)));

    Test::TextOutput output(Test::TextOutput::Verbose);

    return tests.run(output);
}


int main(int argc, char** argv)
{
    try{
        // Maybe initialize MPI
        Dune::MPIHelper& helper = Dune::MPIHelper::instance(argc, argv);

        //sequential version
        if(helper.size()==1) {

           if(!runTests())
             throw std::string("At least one test failed");

            const int dim = 2;
            stationaryDiffusion::SinkSourceDefault<Dune::YaspGrid<dim>::LeafGridView> sink_term, source_term;
            Dune::FieldVector<double, dim> L(1.0);
            Dune::array<int, dim> N (Dune::fill_array<int, dim>(1));
            std::bitset<dim> B(false);
            Dune::YaspGrid<dim> grid(L,N,B,false);
            std::array<std::string,2> methods = {"SIPG", "FEM"};
            stationaryDiffusion::U0Initial<Dune::YaspGrid<dim>::LeafGridView, double> u(grid.leafGridView());
            //solvePoissonPDE<dim, 1>(3,grid);
            stationaryDiffusion::solveDiffusionPDE<dim, 2>(1, methods[0], grid,u, sink_term, source_term);
            stationaryDiffusion::solveDiffusionPDE<dim, 2>(1, methods[1], grid,u, sink_term, source_term);

        }


        return 0;
    }
    catch (Dune::Exception &e){
        std::cerr << "Dune reported error: " << e << std::endl;
    }

    catch (std::string &e) {
        std::cerr << "An error has been reported: " << e << std::endl;
    }

    catch(std::exception &e){
        std::cerr << "STL reported error: " << e.what() << std:: endl;
    }

    catch (...){
        std::cerr << "Unknown exception thrown!" << std::endl;
    }

}