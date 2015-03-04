package fr.inria.tacoma.bft.combinations;

import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Stack;
import java.util.stream.Collectors;

/**
 * This iterator takes a list of collections. It then returns every
 * possible tuple when picking one element in every collection.
 *
 * For instance, if the iterator was built with [[1,2,3],[4,5]] as list
 * of collections, it will return [1,4], [1,5], [2,4] and so on.
 */
class CombinationIterator<T> implements Iterator<List<T>> {

    private final List<Collection<T>> elements;
    private final List<Iterator<T>> iterators;
    private final int maxCombinationNb;
    private int currentCombinationIndex = 0;
    Stack<T> nextCombination = new Stack<>();

    public CombinationIterator(List<Collection<T>> elements) {
        this.elements = elements;
        this.iterators = elements.stream().map(Iterable::iterator).collect(Collectors.toList());
        this.maxCombinationNb = elements.stream().mapToInt(Collection::size).reduce(1,(a,b)->a*b);
    }

    @Override
    public boolean hasNext() {
        return currentCombinationIndex < maxCombinationNb;
    }

    @Override
    public List<T> next() {
        int stackSize = 0;
        if(currentCombinationIndex > 0) { //This is not the first call
            nextCombination.pop();
            stackSize = nextCombination.size();
        }

        while (stackSize < elements.size()) {
            Iterator<T> stateSetIterator = iterators.get(stackSize);
            if (stateSetIterator.hasNext()) {
                nextCombination.push(stateSetIterator.next());
                stackSize++;
            }
            else if(stackSize == 0) {
                break;
            }
            else {
                iterators.set(stackSize, elements.get(stackSize).iterator());
                nextCombination.pop();
                stackSize--;
            }
        }
        currentCombinationIndex++;
        return nextCombination;
    }
}