package fr.inria.tacoma.bft.core.mass;

import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.frame.StateSet;
import gnu.trove.map.hash.TObjectDoubleHashMap;

import java.util.Arrays;
import java.util.Set;

public class MassFunctionImpl implements MassFunction {

    public static final double PRECISION = 10e-12;

    private final FrameOfDiscernment frameOfDiscernment;
    private final TObjectDoubleHashMap<StateSet> focalPoints;
    private double totalAssignedMass = 0.0;

    public MassFunctionImpl(FrameOfDiscernment frameOfDiscernment) {
        this.frameOfDiscernment = frameOfDiscernment;
        this.focalPoints = new TObjectDoubleHashMap<>();
    }

    public MassFunctionImpl(MassFunction toCopy) {
        this.frameOfDiscernment =toCopy.getFrameOfDiscernment();
        this.focalPoints = new TObjectDoubleHashMap<>();
        toCopy.foreachFocalElement(this.focalPoints::put);
        this.totalAssignedMass = toCopy.getTotalAssignedMass();
    }

    @Override
    public FrameOfDiscernment getFrameOfDiscernment() {
        return this.frameOfDiscernment;
    }

    @Override
    public double get(String... elements) {
        if (!this.frameOfDiscernment.containsAll(elements)) {
            throw new IllegalArgumentException(Arrays.toString(elements) + " does not belong to the " +
                    "frame of discernment.");
        }
        return this.focalPoints.get(this.frameOfDiscernment.toStateSet(elements));
    }

    @Override
    public double get(StateSet stateSet) {
        return this.focalPoints.get(stateSet);
    }

    @Override
    public void set(StateSet stateSet, double value) {
        double oldValue;
        if(value == 0) {
            oldValue = this.focalPoints.remove(stateSet);
        }
        else {
            oldValue = this.focalPoints.put(stateSet, value);
        }
        this.totalAssignedMass+= value - oldValue;
    }

    @Override
    public Set<StateSet> getFocalStateSets() {
        return this.focalPoints.keySet();
    }

    @Override
    public void foreachFocalElement(FocalElementConsumer consumer) {
        this.focalPoints.forEachEntry((key, value) -> {
            consumer.apply(key, value);
            return true;
        });
    }

	/*
     * Mass function build methods:
	 */

    @Override
    public void addToFocal(StateSet stateSet, double value) {
        if (value < 0) {
            throw new IllegalArgumentException("Cannot add a focal point with a negative mass " +
                    "(tried to add mass " + value + ".");
        }
        if (value > 0) {
            this.focalPoints.adjustOrPutValue(stateSet, value, value);
            this.totalAssignedMass += value;
        }
    }

    @Override
    public void putRemainingOnIgnorance() {
        double sum = getTotalAssignedMass();
        if (sum > 1) {
            throw new ArithmeticException("The current function is not normalized. Sum of all " +
                    "assigned mass is " + sum + ".");
        }
        addToFocal(this.frameOfDiscernment.fullIgnoranceSet(), 1 - sum);
    }

    @Override
    public double getTotalAssignedMass() {
        return this.totalAssignedMass;
    }

    @Override
    public void discount(double coefficient) {
        final double maxedCoef = coefficient > 1.0 ? 1.0 : coefficient;
        this.focalPoints.transformValues(val -> (1 - maxedCoef) * val);
        this.addToFocal(this.getFrameOfDiscernment().fullIgnoranceSet(),
                maxedCoef * this.totalAssignedMass);
    }


    @Override
    public void normalize() {
        double sum = getTotalAssignedMass();
        if (sum == 0.0) {
            throw new ArithmeticException("Total mass equals 0. Cannot normalize");
        }
        this.focalPoints.transformValues(value -> value / sum);
        this.totalAssignedMass = 1.0;
    }

	/*
	 *
	 */

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }

        MassFunctionImpl that = (MassFunctionImpl) o;

        return equals(that);
    }

    private boolean equals(MassFunctionImpl that) {
        if (!this.frameOfDiscernment.equals(that.frameOfDiscernment)) {
            return false;
        }
        // We have to compare the set with a precision parameter to handle double errors
        return this.focalPoints.forEachEntry((key, value) -> {
            double error = (that.focalPoints.get(key) - value) / value;
            return Math.abs(error) < PRECISION;
        });
    }

    @Override
    public int hashCode() {
        int result = this.frameOfDiscernment != null ? this.frameOfDiscernment.hashCode() : 0;
        result = 31 * result + (this.focalPoints.hashCode());
        return result;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("{\"focals\": [");

        this.focalPoints.forEachEntry((key, value) -> {
            builder.append("{\"set\": ")
                    .append(key.toString())
                    .append(", \"value\": ")
                    .append(value)
                    .append("},");
            return true;
        });
        return builder.replace(builder.length() - 1, builder.length(), " ")
                .append("]}")
                .toString();
    }
}
