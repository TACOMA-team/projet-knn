package fr.inria.tacoma.bft.core.frame;

import java.util.Set;

/**
 * Set of states which is associated to a value in a mass function.
 */
public interface StateSet extends Comparable<StateSet>{

    FrameOfDiscernment getFrame();

    /**
     * Performs an union with this and another set.
     * @param elem other set to perform the conjunction with.
     * @return a new object corresponding to the union.
     */
    StateSet conjunction(StateSet elem);

    /**
     * Performs an intersection with this and another set.
     * @param elem other set to perform the disjunction with.
     * @return a new object corresponding to the intersection.
     */
    StateSet disjunction(StateSet elem);

    /**
     * Perform a difference of this set and the other set.
     * @param elem other set
     * @return a new object which is this set without the elements contained in the other set.
     */
    StateSet difference(StateSet elem);


    /**
     * Returns the complement of the set.
     * @return all the possible states without this set
     */
    StateSet complement();

    /**
     * @return true if this is the empty set.
     */
    boolean isEmpty();

    /**
     * Checks that this state set contains every states from another state set.
     * @param that another state set
     * @return true if this includes or is equal to the given set
     */
    boolean includesOrEquals(StateSet that);

    /**
     * Transforms this state set to a set of states as strings. Useful
     * to list the states as a human readable format.
     * @return a set of strings representing every states in the set.
     */
    Set<String> toStringSet();

    /**
     * @return the cardinality of the set, i.e. the number of states
     */
    int card();
}
