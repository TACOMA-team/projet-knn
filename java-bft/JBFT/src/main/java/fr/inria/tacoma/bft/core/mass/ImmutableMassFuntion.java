package fr.inria.tacoma.bft.core.mass;

import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.frame.StateSet;

import java.util.Set;

public class ImmutableMassFuntion implements MassFunction {

    MassFunction innerMass;

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
