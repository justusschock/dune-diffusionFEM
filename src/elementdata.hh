
#ifndef DUNE_DIFFUSIONFEM_ELEMENTDATA_HH
#define DUNE_DIFFUSIONFEM_ELEMENTDATA_HH
#include <dune/grid/common/mcmgmapper.hh>

template<typename ct, int dim>
class initialvalues {

protected:
    virtual double f(const Dune::FieldVector<ct,dim> &x) const{
        return 6;
    }

public:
    initialvalues () { }

    double operator() (const Dune::FieldVector<ct,dim>& x) const
    {
        return f(x);
    }

};


template<class HGridType, class Function>
void elementdata (const HGridType& grid, const Function& f) {
    // the usual stuff
    //const int dim = G::dimension;
    const int dimworld = HGridType::dimensionworld;
    typedef typename HGridType::ctype ct;
    typedef typename HGridType::LeafGridView GridView;
    typedef typename GridView::template Codim<0>::Iterator ElementLeafIterator;
    typedef typename ElementLeafIterator::Entity::Geometry LeafGeometry;

    // get grid view on leaf part
    GridView gridView = grid.leafGridView();

    // make a mapper for codim 0 entities in the leaf grid
    Dune::LeafMultipleCodimMultipleGeomTypeMapper <HGridType, Dune::MCMGElementLayout>
            mapper(grid);

    // allocate a vector for the data
    std::vector<double> c(mapper.size());

    // iterate through all entities of codim 0 at the leaves
    for (ElementLeafIterator it = gridView.template begin<0>();
         it != gridView.template end<0>(); ++it) {
        // cell geometry
        const LeafGeometry geo = it->geometry();

        // get global coordinate of cell center
        Dune::FieldVector <ct, dimworld> global = geo.center();

        // evaluate functor and store value
        c[mapper.index(*it)] = f(global);
    }
}

#endif //DUNE_DIFFUSIONFEM_ELEMENTDATA_HH
