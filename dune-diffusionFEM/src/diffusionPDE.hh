#ifndef DUNE_DIFFUSIONFEM_DIFFUSIONPDE_HH
#define DUNE_DIFFUSIONFEM_DIFFUSIONPDE_HH

// include output
#include <dune/fem/io/file/dataoutput.hh>

#include "FEMscheme.hh"
#include "problemInterface.hh"
#include "diffusionModel.hh"

template<class GridType>
double algorithm(GridType &grid, int step){

    typedef Dune::Fem::AdaptiveLeafGridPart<GridType, Dune::InteriorBorder_Partition> GridPartType;
    GridPartType gridPart(grid);

    // use a scalar function space
    typedef Dune::Fem::FunctionSpace<double, double, GridType::dimensionworld, 1> FunctionSpaceType;

    // type of the mathematical model used
    typedef DiffusionModel<FunctionSpaceType, GridPartType> ModelType;

    typedef typename ModelType::ProblemType ProblemType;
    ProblemType problem;

    // implicit model for left hand side
    ModelType implicitModel(problem, gridPart);

    typedef FemScheme<ModelType> SchemeType;
    SchemeType scheme(gridPart, implicitModel);

    typedef Dune::Fem::GridFunctionAdapter< ProblemType, GridPartType > GridExactSolutionType;
    GridExactSolutionType gridExactSolution("exact solution", problem, gridPart, 5 );

    //! input/output tuple and setup datawriter
    typedef Dune::tuple< const typename SchemeType::DiscreteFunctionType *, GridExactSolutionType * > IOTupleType;
    typedef Dune::Fem::DataOutput<GridType, IOTupleType > DataOutputType;
    IOTupleType ioTuple( &(scheme.solution()), &gridExactSolution) ; // tuple with pointers
    DataOutputType dataOutput( grid, ioTuple, DataOutputParameters( step ) );

    // setup the right hand side
    scheme.prepare();
    // solve once
    scheme.solve( true );

    // write initial solve
    dataOutput.write();

    // calculate error
    double error = 0 ;

    // calculate standard error
    // select norm for error computation
    typedef Dune::Fem::L2Norm<GridPartType> NormType;
    NormType norm(gridPart);
    error = norm.distance(gridExactSolution, scheme.solution());

    return error;
}

template<class GridType>
bool solveDiffusionPDE(GridType &grid, const int refineSteps, const int level, const int repeats)
{
    // refine grid
    Dune::Fem::GlobalRefine::apply( grid, level * refineSteps );

    // setup EOC loop

    // calculate first step
    double oldError = algorithm( grid, (repeats > 0) ? 0 : -1 );

    for( int step = 1; step <= repeats; ++step )
    {
        // refine globally such that grid with is bisected
        // and all memory is adjusted correctly
        Dune::Fem::GlobalRefine::apply( grid, refineSteps );

        const double newError = algorithm( grid, step );
        const double eoc = log( oldError / newError ) / M_LN2;
        if( Dune::Fem::MPIManager::rank() == 0 )
        {
            std::cout << "Error: " << newError << std::endl;
            std::cout << "EOC( " << step << " ) = " << eoc << std::endl;
        }
        oldError = newError;
    }

}

#endif //DUNE_DIFFUSIONFEM_DIFFUSIONPDE_HH
