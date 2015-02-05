package fr.inria.tacoma.bft.core.mass;

import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.frame.StateSet;

import java.util.Set;

public interface MassFunction {
    FrameOfDiscernment getFrameOfDiscernment();

    /**
     * Helper function. Retrieves the value for the given set. This method
     * allows to use strings to select the set but is not efficient. If
     * possible, try to use get with a StateSet.
     * @param elements elements in the set as strings
     * @return the value for the given set.
     */
    double get(String... elements);

    /**
     * Retrieves the value for a given set.
     * @param stateSet identifier for the set.
     * @return the value for the given set.
     */
    double get(StateSet stateSet);

    /**
     * Set a value for a focal element. If the focal element already existed,
     * the new value replace the former value.
     * @param stateSet identifier for the set.
     * @param value value for the focal element
     */
    void set(StateSet stateSet, double value);

    /**
     * Adds mass to a focal point. The focal point will be added if necessary.
     *
     * @param value          mass for the focal point
     * @param stateSet the ElementSet tu add the value to
     * @throws IllegalArgumentException if the value is negative or the elements are
     *                                  not within the frame of discernment.
     */
    void addToFocal(StateSet stateSet, double value);

    /**
     * @return every state set which is a focal element.
     */
    Set<StateSet> getFocalStateSets();

    /**
     * Iterates over every focal element (i.e. elements with a value != 0) and
     * apply the given function.
     * @param consumer function to apply to each focal.
     */
    void foreachFocalElement(FocalElementConsumer consumer);

    /**
     * Normalizes the mass function by assigning the remaining mass to
     * the total ignorance (i.e. the whole frame of discernment). The remaining
     * mass is 1 minus the total mass assigned so far. The total assigned mass
     * must be lower than 1 in order to use this method.
     *
     * @throws java.lang.ArithmeticException if the total assigned mass is greater than 1
     */
    void putRemainingOnIgnorance();

    /**
     * Normalizes the mass function. This method divides every focal point mass
     * by the total mass assigned so far, which allow to have a total mass of 1.0.
     *
     * @throws java.lang.ArithmeticException if the current total mass is 0.
     */
    void normalize();

    /**
     * Gets the total assigned mass in the function so far (i.e. the sum of all
     * values for the focal elements). On a normalized function, this value
     * should always be one.
     * @return the sum of all values for the focal elements
     */
    double getTotalAssignedMass();

    /**
     * Weakens the mass function. The weakening operation removes mass to every
     * focal points and transmit this mass to the full ignorance set. After
     * applying the weakening the new mass m2 should be (with Ω the full
     * ignorance set, α the coefficient, m the mass function before applying
     * the weakening): <br/>
     * if A ⊂ Ω m2(A) = (1 - α) * m(A) <br/>
     * m2(Ω) = (1 - α) * m(A) + α * totalAssignedMass<br/>
     * @param coefficient coefficient for the weakening, should be between 0 and 1.
     */
    void discount(double coefficient);
}
