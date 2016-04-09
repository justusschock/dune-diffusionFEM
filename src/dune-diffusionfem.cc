#ifdef HAVE_CONFIG_H

#include "config.h"
#endif

#include <iostream>
#include <array>
#include <dune/fem/misc/mpimanager.hh>
#include <dune/fem/function/common/localfunctionadapter.hh>
#include <functional>

#include "poissonPDE.hh"

#define GRIDSELECTOR false



template<class FunctionSpace>
class initialLocalFunction
{
    typedef initialLocalFunction< FunctionSpace > ThisType;

    static const int dimDomain = FunctionSpace::dimDomain;
    static const int dimRange = FunctionSpace::dimRange;

public:
    typedef FunctionSpace FunctionSpaceType;
    typedef typename FunctionSpaceType::DomainType DomainType;
    typedef typename FunctionSpaceType::RangeType RangeType;
    typedef typename FunctionSpaceType::JacobianRangeType JacobianRangeType;

    void evaluate ( const DomainType &x, RangeType &value ) const
    {
        value = RangeType( 2 );
    }

    void jacobian ( const DomainType &x, JacobianRangeType &jacobian ) const
    {
        jacobian = JacobianRangeType( 0 );
    }
};

template<class DiscreteFunctionSpaceImp>
class init{
public:
    using RangeType = typename DiscreteFunctionSpaceImp::FunctionSpaceType::RangeType;
    using DomainType = typename DiscreteFunctionSpaceImp::FunctionSpaceType::DomainType;
    using EntityType = typename DiscreteFunctionSpaceImp::EntityType;

    RangeType operator()(const DomainType& x, const double& y, const EntityType& z)
    {
        return RangeType(2);
    }
};

int main(int argc, char** argv)
{


    try{
        // Maybe initialize MPI
        //Dune::MPIHelper& helper = Dune::MPIHelper::instance(argc, argv);
        Dune::Fem::MPIManager::initialize(argc, argv);

        const std::string problemNames [] = {"sin", "cos", "middle"};


        if(!GRIDSELECTOR) {
            // Default Grid-Setup
            const int dim = 2;
            typedef typename Dune::YaspGrid<dim> HGridType;
            Dune::FieldVector<double, dim> L(1.0);
            Dune::array<int, dim> N(Dune::fill_array<int, dim>(1));
            std::bitset<dim> B(false);
            Dune::YaspGrid<dim> grid(L, N, B, false);


            //Typedefs for initial Values
            //for all Versions
            using GridPartType = typename Dune::Fem::AdaptiveLeafGridPart<HGridType>;
            using FunctionSpace = Dune::Fem::FunctionSpace< double, double, HGridType::dimensionworld, 1 >;
            std::string initialname = "constant_values";
            GridPartType gridPart (grid);

            //Version 1:
            using DiscreteFunctionSpace = Dune::Fem::DiscreteFunctionSpaceAdapter<FunctionSpace , GridPartType >;
            using RangeType = DiscreteFunctionSpace ::FunctionSpaceType::RangeType;
            using DomainType = DiscreteFunctionSpace ::FunctionSpaceType::DomainType;
            using EntityType = DiscreteFunctionSpace ::EntityType;
            using localfunction = std::function<RangeType(const DomainType&, const double& ,const EntityType&)>;
            using LocalFunctionImp = Dune::Fem::LocalAnalyticalFunctionBinder<DiscreteFunctionSpace, localfunction >;
            using initialValues = Dune::Fem::LocalFunctionAdapter<LocalFunctionImp >;

            localfunction func = init<DiscreteFunctionSpace >();
            LocalFunctionImp funcImp(func);
            initialValues initial(initialname,funcImp,gridPart);
            solvePoissonPDE(grid, 2, 0, 2, 0, initial);

            /* //Version 2:
            using initialFunc = initialLocalFunction<FunctionSpace>;
            using initialValues = Dune::Fem::GridFunctionAdapter<initialFunc, GridPartType>;
            initialFunc initialFunction;
            initialValues initial(initialname,initialFunction,gridPart);
            solvePoissonPDE(grid, 2, 0, 2, 0, initial);
             */

            /*
            //Version 3 (without interpolate):
            DiscreteFunctionSpace dfSpace(gridPart);
            DiscreteFunctionSpace::IteratorType end =dfSpace.end();
            for(DiscreteFunctionSpace::IteratorType it = dfSpace.begin(); it!= end; ++it)
            {
                const DiscreteFunctionSpace::EntityType& entity = *it;
                auto localfunction = initial.localFunction(entity);
                localfunction[0] = 0;
            }
*/

        }
        else {
            // append overloaded parameters from the command line
            Dune::Fem::Parameter::append(argc, argv);

            // append possible given parameter files
            for (int i = 1; i < argc; ++i)
                Dune::Fem::Parameter::append(argv[i]);

            // append default parameter file
            Dune::Fem::Parameter::append("../data/parameter");

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

            typedef Dune::Fem::FunctionSpace< double, double, HGridType::dimensionworld, 1 > FunctionSpaceType;
            //initialValues<FunctionSpaceType, HGridType> initial;

            //solvePoissonPDE(grid, refineStepsForHalf, level, repeats, problemNumber, initial);

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

