
#ifndef DUNE_DIFFUSIONFEM_RHS_HH
#define DUNE_DIFFUSIONFEM_RHS_HH


#include <dune/fem/quadrature/cachingquadrature.hh>


// assembleRHS
// -----------

// assembleRHS
// -----------

template< class Model, class Function, class Neuman, class DiscreteFunction >
void assembleRHS ( const Model &model, const Function &function, const Neuman &neuman, DiscreteFunction &rhs )
{
    const bool neumanBnd = model.hasNeumanBoundary();
    rhs.clear();
    typedef typename DiscreteFunction::DiscreteFunctionSpaceType DiscreteFunctionSpaceType;
    typedef typename DiscreteFunction::LocalFunctionType LocalFunctionType;

    typedef typename DiscreteFunctionSpaceType::GridPartType GridPartType;
    typedef typename DiscreteFunctionSpaceType::IteratorType IteratorType;
    typedef typename IteratorType::Entity EntityType;
    typedef typename EntityType::Geometry GeometryType;
    typedef typename GridPartType::IntersectionIteratorType IntersectionIteratorType;
    typedef typename IntersectionIteratorType::Intersection IntersectionType;

    const DiscreteFunctionSpaceType &dfSpace = rhs.space();

    typedef typename DiscreteFunctionSpaceType::GridPartType GridPartType;
    typedef Dune::Fem::CachingQuadrature< GridPartType, 0 > QuadratureType;
    typedef Dune::Fem::ElementQuadrature< GridPartType, 1 > FaceQuadratureType;
    const int quadOrder = 2*dfSpace.order()+1;

    const IteratorType end = dfSpace.end();
    for( IteratorType it = dfSpace.begin(); it != end; ++it )
    {
        const EntityType &entity = *it;
        const GeometryType &geometry = entity.geometry();

        const typename Function::LocalFunctionType localFunction =
                function.localFunction( entity);
        LocalFunctionType rhsLocal = rhs.localFunction( entity );
        typedef typename Function::RangeType RangeType;

        QuadratureType quadrature( entity, quadOrder );
        const size_t numQuadraturePoints = quadrature.nop();
        for( size_t pt = 0; pt < numQuadraturePoints; ++pt )
        {
            // obtain quadrature point
            const typename QuadratureType::CoordinateType &x = quadrature.point( pt );

            // evaluate f
            RangeType f;
            localFunction.evaluate( quadrature[ pt ], f );

            // multiply by quadrature weight
            f *= quadrature.weight( pt ) * geometry.integrationElement( x );

            // add f * phi_i to rhsLocal[ i ]
            rhsLocal.axpy( quadrature[ pt ], f );
        }
        if (neumanBnd)
        {
            if ( !entity.hasBoundaryIntersections() )
                continue;

            const IntersectionIteratorType iitend = dfSpace.gridPart().iend( entity );
            for( IntersectionIteratorType iit = dfSpace.gridPart().ibegin( entity ); iit != iitend; ++iit ) // looping over intersections
            {
                const IntersectionType &intersection = *iit;
                if ( ! intersection.boundary() )
                    continue;
                Dune::FieldVector<bool,RangeType::dimension> components(true);
                // if ( model.isDirichletIntersection( intersection, components) )
                //   continue;
                bool hasDirichletComponent = model.isDirichletIntersection( intersection, components);

                const typename Neuman::LocalFunctionType neumanLocal = neuman.localFunction( entity);

                const typename IntersectionType::Geometry &intersectionGeometry = intersection.geometry();
                FaceQuadratureType quadInside( dfSpace.gridPart(), intersection, quadOrder, FaceQuadratureType::INSIDE );
                const size_t numQuadraturePoints = quadInside.nop();
                for( size_t pt = 0; pt < numQuadraturePoints; ++pt )
                {
                    const typename FaceQuadratureType::LocalCoordinateType &x = quadInside.localPoint( pt );
                    RangeType nval;
                    neumanLocal.evaluate(quadInside[pt], nval);
                    nval *= quadInside.weight( pt ) * intersectionGeometry.integrationElement( x );
                    for(int k = 0; k < RangeType::dimension; ++k)
                        if ( hasDirichletComponent && components[k] )
                            nval[k] = 0;
                    rhsLocal.axpy( quadInside[ pt ], nval );
                }
            }
        }
    }
    rhs.communicate();
}

// assembleDGRHS
// -----------

template< class Model, class Function, class Neuman, class Dirichlet, class DiscreteFunction >
void assembleDGRHS ( const Model &model, const Function &function, const Neuman &neuman, const Dirichlet &dirichlet, DiscreteFunction &rhs )
{
    assembleRHS( model, function, neuman, rhs );
    if ( ! model.hasDirichletBoundary() )
        return;

    typedef typename DiscreteFunction::DiscreteFunctionSpaceType DiscreteFunctionSpaceType;
    typedef typename DiscreteFunction::LocalFunctionType LocalFunctionType;

    typedef typename DiscreteFunctionSpaceType::IteratorType IteratorType;
    typedef typename IteratorType::Entity EntityType;
    typedef typename EntityType::Geometry GeometryType;
    typedef typename LocalFunctionType::RangeType RangeType;
    typedef typename LocalFunctionType::JacobianRangeType JacobianRangeType;
    typedef typename DiscreteFunctionSpaceType::DomainType DomainType;
    static const int dimDomain = LocalFunctionType::dimDomain;
    static const int dimRange = LocalFunctionType::dimRange;

    typedef typename DiscreteFunctionSpaceType::GridPartType GridPartType;
    typedef typename GridPartType::IntersectionIteratorType IntersectionIteratorType;
    typedef typename IntersectionIteratorType::Intersection IntersectionType;

    typedef Dune::Fem::ElementQuadrature< GridPartType, 1 > FaceQuadratureType;

    const DiscreteFunctionSpaceType &dfSpace = rhs.space();
    const int quadOrder = 2*dfSpace.order()+1;

    const IteratorType end = dfSpace.end();
    for( IteratorType it = dfSpace.begin(); it != end; ++it )
    {
        const EntityType &entity = *it;
        if ( !entity.hasBoundaryIntersections() )
            continue;

        const GeometryType &geometry = entity.geometry();
        double area = geometry.volume();

        LocalFunctionType rhsLocal = rhs.localFunction( entity );

        const IntersectionIteratorType iitend = dfSpace.gridPart().iend( entity );
        for( IntersectionIteratorType iit = dfSpace.gridPart().ibegin( entity ); iit != iitend; ++iit ) // looping over intersections
        {
            const IntersectionType &intersection = *iit;
            if ( ! intersection.boundary() ) // i.e. if intersection is on boundary: nothing to be done for Neumann zero b.c.
                continue;                      // since [u] = 0  and grad u.n = 0
            Dune::FieldVector<bool,dimRange> components;
            if ( ! model.isDirichletIntersection( intersection, components) )
                continue;

            const typename Dirichlet::LocalFunctionType dirichletLocal = dirichlet.localFunction( entity);

            typedef typename IntersectionType::Geometry  IntersectionGeometryType;
            const IntersectionGeometryType &intersectionGeometry = intersection.geometry();

            const double intersectionArea = intersectionGeometry.volume();
            const double beta = model.penalty() * intersectionArea / area;

            FaceQuadratureType quadInside( dfSpace.gridPart(), intersection, quadOrder, FaceQuadratureType::INSIDE );
            const size_t numQuadraturePoints = quadInside.nop();
            for( size_t pt = 0; pt < numQuadraturePoints; ++pt )
            {
                const typename FaceQuadratureType::LocalCoordinateType &x = quadInside.localPoint( pt );
                const DomainType normal = intersection.integrationOuterNormal( x );

                const double weight = quadInside.weight( pt );

                RangeType value;
                JacobianRangeType dvalue,advalue;

                RangeType vuOut;
                dirichletLocal.evaluate( quadInside[pt], vuOut );
                for (int r=0;r<dimRange;++r)
                    if (!components[r]) // do not use dirichlet constraints here
                        vuOut[r] = 0;

                value = vuOut;
                value *= beta * intersectionGeometry.integrationElement( x );

                //  [ u ] * { grad phi_en } = -normal(u+ - u-) * 0.5 grad phi_en
                // here we need a diadic product of u x n
                for (int r=0;r<dimRange;++r)
                    for (int d=0;d<dimDomain;++d)
                        dvalue[r][d] = -0.5 * normal[d] * vuOut[r];

                model.diffusiveFlux( entity, quadInside[ pt ], vuOut, dvalue, advalue );

                value *= weight;
                advalue *= weight;
                rhsLocal.axpy( quadInside[ pt ], value, advalue );
            }
        }
    }
    rhs.communicate();
}


#endif //DUNE_DIFFUSIONFEM_RHS_HH
