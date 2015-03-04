package fr.inria.tacoma.bft.core.frame.impl;

import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.frame.StateSet;

import java.math.BigInteger;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

class BigIntegerStateSet extends AbstractStateSet implements StateSet {

    private final BigInteger setId;

    protected BigIntegerStateSet(FrameOfDiscernment frameOfDiscernment, BigInteger setId) {
        super(frameOfDiscernment);
        this.setId = setId;
    }

    @Override
    public StateSet conjunction(StateSet elem) {
        return new BigIntegerStateSet(this.getFrame(),
                                      this.setId.and(((BigIntegerStateSet) elem).setId));
    }

    @Override
    public StateSet disjunction(StateSet elem) {
        return new BigIntegerStateSet(this.getFrame(),
                                      this.setId.or(((BigIntegerStateSet) elem).setId));
    }

    @Override
    public StateSet difference(StateSet elem) {
        return new BigIntegerStateSet(this.getFrame(),
                                      this.setId.andNot(((BigIntegerStateSet) elem).setId));
    }


    @Override
    public boolean isEmpty() {
        return this.setId.equals(BigInteger.ZERO);
    }

    @Override
    public boolean includesOrEquals(StateSet that) {
        BigInteger otherSetId = ((BigIntegerStateSet) that).setId;
        return this.setId.and(otherSetId).equals(otherSetId);
    }

    @Override
    public int card() {
        return this.setId.bitCount();
    }

    @Override
    public Set<String> toStringSet() {
        Set<String> stateSetAsStrings = new HashSet<>();
        List<String> states = this.getFrame().getStates();
        int stateIndex = 0;
        while(!this.setId.shiftRight(stateIndex).equals(BigInteger.ZERO)) {
            BigInteger mask = BigInteger.ONE.shiftLeft(stateIndex);
            if(!mask.and(this.setId).equals(BigInteger.ZERO)) {
                stateSetAsStrings.add(states.get(stateIndex));
            }
            stateIndex++;
        }
        return stateSetAsStrings;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }

        BigIntegerStateSet that = (BigIntegerStateSet) o;

        return this.setId.equals(that.setId);
    }

    @Override
    public int hashCode() {
        return this.setId.hashCode();
    }

    @Override
    public int compareTo(StateSet o) {
        BigIntegerStateSet that = (BigIntegerStateSet) o;
        if(this.equals(that)) {
            return 0;
        }
        int diff = this.card() - o.card();
        if(diff == 0) {
            return this.setId.compareTo(that.setId);
        }
        else {
            return diff;
        }
    }
}
