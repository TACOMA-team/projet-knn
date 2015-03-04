package fr.inria.tacoma.bft.core.mass;

import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.frame.StateSet;

import java.util.Set;

/**
 * A mass function which is immutable. It throws UnsupportedOperationException
 * when trying to modify it. It is used to wrap a mass function when you have
 * to return it and want to be sure it is not modified afterward.
 */
public class ImmutableMassFuntion implements MassFunction {

    private MassFunction innerMass;

    /**
     * Creates a new immutable mass function from another mass function.
     * @param mass mass function which will be wrapped by this
     */
    public ImmutableMassFuntion(MassFunction mass) {
        this.innerMass = mass;
    }

    @Override
    public FrameOfDiscernment getFrameOfDiscernment() {
        return innerMass.getFrameOfDiscernment();
    }

    @Override
    public double get(String... elements) {
        return innerMass.get(elements);
    }

    @Override
    public double get(StateSet stateSet) {
        return innerMass.get(stateSet);
    }

    @Override
    public void set(StateSet stateSet, double value) {
        throw new UnsupportedOperationException("Cannot call set on immutable mass function");
    }

    @Override
    public void addToFocal(StateSet stateSet, double value) {
        throw new UnsupportedOperationException("Cannot call addToFocal on immutable mass function");
    }

    @Override
    public Set<StateSet> getFocalStateSets() {
        return innerMass.getFocalStateSets();
    }

    @Override
    public void foreachFocalElement(FocalElementConsumer consumer) {
        innerMass.foreachFocalElement(consumer);
    }

    @Override
    public void putRemainingOnIgnorance() {
        throw new UnsupportedOperationException(
                "Cannot call putRemainingOnIgnorance on immutable mass function");
    }

    @Override
    public void normalize() {
        throw new UnsupportedOperationException("Cannot call normalize on immutable mass function");
    }

    @Override
    public double getTotalAssignedMass() {
        return innerMass.getTotalAssignedMass();
    }

    @Override
    public void discount(double coefficient) {
        throw new UnsupportedOperationException("Cannot call discount on immutable mass function");
    }

    @Override
    public String toString() {
        return innerMass.toString();
    }

    @Override
    public boolean equals(Object o) {
        return innerMass.equals(o);
    }

    @Override
    public int hashCode() {
        return innerMass.hashCode();
    }
}
