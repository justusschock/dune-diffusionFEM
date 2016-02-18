#ifdef HAVE_CONFIG_H
#endif

#include <iostream>
#include <dune/common/parallel/mpihelper.hh> // An initializer of MPI
#include <dune/fem/gridpart/adaptiveleafgridpart.hh>
#include <dune/fem/space/common/adaptmanager.hh>
#include <dune/fem/

#include "diffusionPDE.hh"

template<class HGridType>
double algorithm(HGridType &grid, int step){


    return 0.0;
}

int main(int argc, char** argv)
{
    try{
        // Maybe initialize MPI
        //Dune::MPIHelper& helper = Dune::MPIHelper::instance(argc, argv);
        Dune::Fem::MPIManager::initialize(argc, argv);

        // append overloaded parameters from the command line
        Dune::Fem::Parameter::append( argc, argv );

        // append possible given parameter files
        for( int i = 1; i < argc; ++i )
            Dune::Fem::Parameter::append( argv[ i ] );

        // append possible given parameter files
        for( int i = 1; i < argc; ++i )
            Dune::Fem::Parameter::append( argv[ i ] );

        // append default parameter file
        Dune::Fem::Parameter::append( "../data/parameter" );

        // type of hierarchical grid
        typedef Dune::GridSelector::GridType HGridType;

        // create grid from DGF file
        const std::string gridkey = Dune::Fem::IOInterface::defaultGridKey( HGridType::dimension );
        const std::string gridfile = Dune::Fem::Parameter::getValue< std::string >( gridkey );

        // the method rank and size from MPIManager are static
        if( Dune::Fem::MPIManager::rank() == 0 )
            std::cout << "Loading macro grid: " << gridfile << std::endl;

        // construct macro using the DGF Parser
        Dune::GridPtr< HGridType > gridPtr( gridfile );
        HGridType& grid = *gridPtr ;

        // do initial load balance
        grid.loadBalance();

        // initial grid refinement
        const int level = Dune::Fem::Parameter::getValue< int >( "poisson.level" );

        // number of global refinements to bisect grid width
        const int refineStepsForHalf = Dune::DGFGridInfo< HGridType >::refineStepsForHalf();

        // refine grid
        Dune::Fem::GlobalRefine::apply( grid, level * refineStepsForHalf );

        // setup EOC loop
        const int repeats = Dune::Fem::Parameter::getValue< int >( "poisson.repeats", 0 );

        // calculate first step
        double oldError = algorithm( grid, (repeats > 0) ? 0 : -1 );

        for( int step = 1; step <= repeats; ++step )
        {
            // refine globally such that grid with is bisected
            // and all memory is adjusted correctly
            Dune::Fem::GlobalRefine::apply( grid, refineStepsForHalf );

            const double newError = algorithm( grid, step );
            const double eoc = log( oldError / newError ) / M_LN2;
            if( Dune::Fem::MPIManager::rank() == 0 )
            {
                std::cout << "Error: " << newError << std::endl;
                std::cout << "EOC( " << step << " ) = " << eoc << std::endl;
            }
            oldError = newError;
        }

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

