package fr.inria.tacoma.bft.decision;

import fr.inria.tacoma.bft.core.frame.StateSet;
import fr.inria.tacoma.bft.core.mass.MassFunctionImpl;

public class Decision {

    private static double PRECISION = MassFunctionImpl.PRECISION;

    private StateSet stateSet;
    private double confidence;



    protected Decision(StateSet stateSet, double confidence) {
        assert confidence > 0;
        this.stateSet = stateSet;
        this.confidence =confidence;
    }

    public StateSet getStateSet() {
        return this.stateSet;
    }

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
