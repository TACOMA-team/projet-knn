package fr.inria.tacoma.bft.decision;

import fr.inria.tacoma.bft.core.frame.StateSet;
import fr.inria.tacoma.bft.core.mass.MassFunctionImpl;

/**
 * This class represent a decision taken by a decision algorithm.
 * It is composed of a state set, which gives the current state of our system
 * (for instance, the person is "sitting or standing"), and a confidence rate,
 * which gives the confidence we have in this StateSet.
 */
public class Decision {

    /**
     * Precision used to compare double in equals.
     */
    private static double PRECISION = MassFunctionImpl.PRECISION;

    private StateSet stateSet;
    private double confidence;



    protected Decision(StateSet stateSet, double confidence) {
        assert confidence > 0;
        this.stateSet = stateSet;
        this.confidence =confidence;
    }

    /**
     * @return the StateSet which is the current situation.
     */
    public StateSet getStateSet() {
        return this.stateSet;
    }

    /**
     * Gives the confidence for the current state set. This number is between
     * 0 (not sure at all) and 1 (completely sure about the situation).
     * @return the confidence rating.
     */
    public double getConfidence() {
        return this.confidence;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }

        Decision decision = (Decision) o;

        double error = (this.confidence - decision.confidence) / this.confidence;
        return Math.abs(error) < PRECISION && this.stateSet.equals(decision.stateSet);

    }

    @Override
    public int hashCode() {
        int result;
        long temp;
        result = this.stateSet.hashCode();
        temp = Double.doubleToLongBits(this.confidence);
        result = 31 * result + (int) (temp ^ (temp >>> 32));
        return result;
    }


    @Override
    public String toString() {
        return "{" +
                "\"stateSet\": \"" + stateSet + '"' +
                ", \"confidence\": " + confidence +
                '}';
    }
}
