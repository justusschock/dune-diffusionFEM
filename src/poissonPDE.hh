#ifndef DUNE_DIFFUSIONFEM_DIFFUSIONPDE_HH
#define DUNE_DIFFUSIONFEM_DIFFUSIONPDE_HH

#include <cassert>
#include <cmath>
// include output
#include <dune/fem/io/file/dataoutput.hh>

// iostream includes
#include <iostream>

// include grid part
#include <dune/fem/gridpart/adaptiveleafgridpart.hh>

// include output
#include <dune/fem/io/file/dataoutput.hh>
#include <dune/fem/io/file/vtkio.hh>

// include header of elliptic solver
#include "FEMscheme.hh"

#include "problemInterface.hh"

//include for data attaching
#include "elementdata.hh"
#include <dune/fem/space/common/interpolate.hh>
#include <dune/fem/space/finitevolume.hh>

const bool graphics = true;

// -laplace u + u = f with Dirichlet and Neumann boundary conditions
// Exact solution is u(x_1,...,x_d) = cos(2*pi*x_1)...cos(2*pi*x_d)
template <class FunctionSpace>
class CosinusProduct : public ProblemInterface < FunctionSpace >
{
    typedef ProblemInterface < FunctionSpace >  BaseType;
public:
    typedef typename BaseType :: RangeType            RangeType;
    typedef typename BaseType :: DomainType           DomainType;
    typedef typename BaseType :: JacobianRangeType    JacobianRangeType;
    typedef typename BaseType :: DiffusionTensorType  DiffusionTensorType;

    enum { dimRange  = BaseType :: dimRange };
    enum { dimDomain = BaseType :: dimDomain };

    //! the right hand side data (default = 0)
    virtual void f(const DomainType& x,
                   RangeType& phi) const
    {
        phi = 4*dimDomain*(M_PI*M_PI);
        for( int i = 0; i < dimDomain; ++i )
            phi *= std::cos( 2*M_PI*x[ i ] );
        RangeType uVal;
        u(x,uVal);
        phi += uVal;
    }

    //! the exact solution
    virtual void u(const DomainType& x,
                   RangeType& phi) const
    {
        phi = 1;
        for( int i = 0; i < dimDomain; ++i )
            phi *= std::cos( 2*M_PI*x[ i ] );
    }

    //! the jacobian of the exact solution
    virtual void uJacobian(const DomainType& x,
                           JacobianRangeType& ret) const
    {
        for( int r = 0; r < dimRange; ++ r )
        {
            for( int i = 0; i < dimDomain; ++i )
            {
                ret[ r ][ i ] = -2*M_PI*std::sin( 2*M_PI*x[ i ] );
                for( int j = 1; j < dimDomain; ++j )
                    ret[ r ][ i ] *= std::cos( 2*M_PI*x[ (i+j)%dimDomain ] );
            }
        }
    }

    //! mass coefficient has to be 1 for this problem
    virtual void m(const DomainType& x, RangeType &m) const
    {
        m = RangeType(1);
    }

    virtual void alpha(const DomainType& x, RangeType &a) const
    {
        a = RangeType(0.5);
    }
    //! the Dirichlet boundary data (default calls u)
    virtual void g(const DomainType& x,
                   RangeType& value) const
    {
        value = RangeType(1);
    }
    virtual bool hasDirichletBoundary () const
    {
        return true ;
    }
    virtual bool hasNeumanBoundary () const
    {
        return true ;
    }
    virtual bool isDirichletPoint( const DomainType& x ) const
    {
        // all boundaries except the x=0 plane are Dirichlet
        return (std::abs(x[0])>1e-8);

    }
    virtual void n(const DomainType& x,
                   RangeType& value) const
    {
        value = RangeType(1);
    }
};

// -laplace u = f with Dirichlet and Neumann boundary conditions
// Exact solution is u(x_1,...,x_d) = sin(2*pi*x_1)...sin(2*pi*x_d)
template <class FunctionSpace>
class SinusProduct : public ProblemInterface < FunctionSpace >
{
    typedef ProblemInterface < FunctionSpace >  BaseType;
public:
    typedef typename BaseType :: RangeType            RangeType;
    typedef typename BaseType :: DomainType           DomainType;
    typedef typename BaseType :: JacobianRangeType    JacobianRangeType;
    typedef typename BaseType :: DiffusionTensorType  DiffusionTensorType;

    enum { dimRange  = BaseType :: dimRange };
    enum { dimDomain = BaseType :: dimDomain };

    //! the right hand side data (default = 0)
    virtual void f(const DomainType& x,
                   RangeType& phi) const
    {
        phi = 4*dimDomain*(M_PI*M_PI);
        for( int i = 0; i < dimDomain; ++i )
            phi *= std::sin( 2*M_PI*x[ i ] );
    }

    //! the exact solution
    virtual void u(const DomainType& x,
                   RangeType& phi) const
    {
        phi = 1;
        for( int i = 0; i < dimDomain; ++i )
            phi *= std::sin( 2*M_PI*x[ i ] );
        phi[0] += x[0]*x[0]-x[1]*x[1]+x[0]*x[1];
        // phi[0] += x[0]*x[1];
        // phi[0] += 0.5;
    }

    //! the jacobian of the exact solution
    virtual void uJacobian(const DomainType& x,
                           JacobianRangeType& ret) const
    {
        for( int r = 0; r < dimRange; ++ r )
        {
            for( int i = 0; i < dimDomain; ++i )
            {
                ret[ r ][ i ] = 2*M_PI*std::cos( 2*M_PI*x[ i ] );
                for( int j = 1; j < dimDomain; ++j )
                    ret[ r ][ i ] *= std::sin( 2*M_PI*x[ (i+j)%dimDomain ] );
            }
        }
        ret[0][0] +=  2.*x[0]+x[1];
        ret[0][1] += -2.*x[1]+x[0];
        // ret[0][0] += x[1];
        // ret[0][1] += x[0];
    }
    //! mass coefficient has to be 1 for this problem
    virtual void m(const DomainType& x, RangeType &m) const
    {
        m = RangeType(1);
    }

    //DomainType = Type of input variable (e.g const double) (in this case: x is point in domain)
    //RangeType = Type of output variable (E.g. double) (in this case: value at x)
    virtual void alpha(const DomainType& x, RangeType &a) const
    {
        a = RangeType(0.5);
    }
    //! the Dirichlet boundary data (default calls u)
    virtual void g(const DomainType& x,
                   RangeType& value) const
    {
        value = RangeType(100);
    }
    virtual bool hasDirichletBoundary () const
    {
        return true ;
    }
    virtual bool hasNeumanBoundary () const
    {
        return false ;
    }
    virtual bool isDirichletPoint( const DomainType& x ) const
    {
        // all boundaries except the x=0 plane are Dirichlet
        return (std::abs(x[0])>1e-8);

    }
    virtual void n(const DomainType& x,
                   RangeType& value) const
    {
        /*u(x,value);
        value *= 0.5;
        JacobianRangeType jac;
        uJacobian(x,jac);
        value[0] -= jac[0][0];
         */
        value = RangeType(2);

    }
};

template<class FunctionSpace>
class SingleMiddleSource: public ProblemInterface<FunctionSpace>{
    typedef ProblemInterface < FunctionSpace >  BaseType;
public:
    typedef typename BaseType :: RangeType            RangeType;
    typedef typename BaseType :: DomainType           DomainType;
    typedef typename BaseType :: JacobianRangeType    JacobianRangeType;
    typedef typename BaseType :: DiffusionTensorType  DiffusionTensorType;

    enum { dimRange  = BaseType :: dimRange };
    enum { dimDomain = BaseType :: dimDomain };

    //! the right hand side data (default = 0)
    virtual void f(const DomainType& x,
                   RangeType& phi) const {
        phi = 0;
        if (std::abs(x[0] - 0.5) <= 1e-6 && std::abs(x[1] - 0.5) <= 1e-6)
            phi = 2;
    }

    //! mass coefficient has to be 1 for this problem
    virtual void m(const DomainType& x, RangeType &m) const
    {
        m = RangeType(1);
    }

    //DomainType = Type of input variable (e.g const double) (in this case: x is point in domain)
    //RangeType = Type of output variable (E.g. double) (in this case: value at x)
    virtual void alpha(const DomainType& x, RangeType &a) const
    {
        a = RangeType(0.5);
    }
    //! the Dirichlet boundary data (default calls u)
    virtual void g(const DomainType& x,
                   RangeType& value) const
    {
        value = RangeType(1);
    }
    virtual bool hasDirichletBoundary () const
    {
        return true ;
    }
    virtual bool hasNeumanBoundary () const
    {
        return false ;
    }
    virtual bool isDirichletPoint( const DomainType& x ) const
    {
        // all boundaries except the x=0 plane are Dirichlet
        return x[0]==1;

    }
    virtual void n(const DomainType& x,
                   RangeType& value) const
    {
        /*u(x,value);
        value *= 0.5;
        JacobianRangeType jac;
        uJacobian(x,jac);
        value[0] -= jac[0][0];
         */
        value = RangeType(2);

    }

};

// assemble-solve-estimate-mark-refine-IO-error-doitagain
template <class HGridType, class FunctionType>
double algorithm ( HGridType &grid, int step, const int problemNumber, FunctionType& initialValues )
{
    // we want to solve the problem on the leaf elements of the grid
    typedef Dune::Fem::AdaptiveLeafGridPart< HGridType > GridPartType;
    GridPartType gridPart(grid);

    // use a scalar function space
    typedef Dune::Fem::FunctionSpace< double, double, HGridType::dimensionworld, 1 > FunctionSpaceType;

    // type of the mathematical model used
    typedef nonlinearModel< FunctionSpaceType, GridPartType > ModelType;
    typedef typename ModelType::ProblemType ProblemType ;
    std::shared_ptr<ProblemType> problemPtr = 0 ;
    std::stringstream fullname;
    fullname << "poisson_";

    switch ( problemNumber )
    {
        case 0:
            problemPtr.reset(new SinusProduct< FunctionSpaceType >);
            fullname << "sin_problem_step_" << step;
            break ;
        case 1:
            problemPtr.reset(new CosinusProduct< FunctionSpaceType >);
            fullname << "cos_problem_step_" << step;
            break ;
        case 2:
            problemPtr.reset(new SingleMiddleSource<FunctionSpaceType>);
            fullname << "single-middle-source_problem_step" << step;
            break;
        default:
            problemPtr.reset(new SinusProduct< FunctionSpaceType >);
            fullname << "sin_problem_step_" << step;
    }
    assert( problemPtr );
    ProblemType& problem = *problemPtr.get() ;
    // implicit model for left hand side
    ModelType implicitModel( problem, gridPart );

    // poisson solver
    typedef FemScheme< ModelType, FunctionType > SchemeType;
    SchemeType scheme( gridPart, implicitModel, initialValues );

    typedef Dune::Fem::GridFunctionAdapter< ProblemType, GridPartType > GridExactSolutionType;
    GridExactSolutionType gridExactSolution("exact solution", problem, gridPart, 5 );

    //! input/output tuple and setup datawritter
    typedef Dune::tuple< const typename SchemeType::DiscreteFunctionType *, GridExactSolutionType * > IOTupleType;
    typedef Dune::Fem::DataOutput< HGridType, IOTupleType > DataOutputType;
    IOTupleType ioTuple( &(scheme.solution()), &gridExactSolution) ; // tuple with pointers
    DataOutputType dataOutput( grid, ioTuple, DataOutputParameters( step ) );
scheme.init();
    // setup the right hand side
   // scheme.prepare();
    // solve once (assemble matrix)
    //scheme.solve(true);

    // write initial solve
    dataOutput.write("~/ISEA/build/DiffusionTest/dune-diffusionFEM/src/test");

    // calculate error
    double error = 0 ;

    // calculate standard error
    // select norm for error computation
    typedef Dune::Fem::L2Norm< GridPartType > NormType;
    NormType norm( gridPart );
    error = norm.distance( gridExactSolution, scheme.solution() );

    // write vtk file
    if (graphics) {
        Dune::Fem::SubsamplingVTKIO<GridPartType> vtkio(gridPart);

        vtkio.addVertexData(scheme.solution(), "Numerical_solution");
        vtkio.addVertexData(gridExactSolution, "Exact_solution");
        //vtkio.addVertexData(, "Dirichlet_boundary_condition");

        vtkio.write(fullname.str(), Dune::VTK::appendedraw);
    }
    problemPtr.reset();
    return error ;
}



template<class GridType, class FunctionType>
bool solvePoissonPDE(GridType &grid, const int refineSteps, const int level, const int repeats, const double problemNumber, FunctionType& initialValues)
{
    try {
        // refine grid
        Dune::Fem::GlobalRefine::apply( grid, level * refineSteps );

       // initialvalues<typename GridType::ctype, GridType::dimension> initial;
        //elementdata(grid, initial);

        // setup EOC loop

        // calculate first step
        double oldError = algorithm( grid, (repeats > 0) ? 0 : -1, problemNumber, initialValues );

        for( int step = 1; step <= repeats; ++step )
        {
            // refine globally such that grid with is bisected
            // and all memory is adjusted correctly
            Dune::Fem::GlobalRefine::apply( grid, refineSteps );



            const double newError = algorithm( grid, step, problemNumber, initialValues );
            const double eoc = log( oldError / newError ) / M_LN2;
            if( Dune::Fem::MPIManager::rank() == 0 )
            {
                std::cout << "Error step " << step << " :" << newError << std::endl;
                std::cout << "EOC( " << step <<"/"<< step-1 <<" ) = " << eoc << std::endl;
            }
            oldError = newError;
        }
        return true;
    }
    catch (std::string &e){
        std::cerr << e << std::endl;
        return false;
    }
    catch (Dune::Exception &e){
        std::cerr << "Dune reported error: " << e << std::endl;
        return false;
    }
    catch (std::exception &e){
        std::cerr << "STL reported error: " << e.what() << std::endl;
        return false;
    }
    catch (...){
        std::cerr << "Unknown exception thrown!" << std::endl;
        return false;
    }
}

#endif //DUNE_DIFFUSIONFEM_DIFFUSIONPDE_HH
