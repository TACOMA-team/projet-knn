package fr.inria.tacoma.bft.core.frame.impl;

import fr.inria.tacoma.bft.core.frame.StateSet;
import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

class LongStateSet extends AbstractStateSet implements StateSet {

    private final long setId;

    protected LongStateSet(FrameOfDiscernment frameOfDiscernment, long setId) {
        super(frameOfDiscernment);
        this.setId = setId;
    }

    @Override
    public StateSet conjunction(StateSet elem) {
        return new LongStateSet(this.getFrame(), this.setId & ((LongStateSet) elem).setId );
    }

    @Override
    public StateSet disjunction(StateSet elem) {
        return new LongStateSet(this.getFrame(), this.setId | ((LongStateSet) elem).setId );
    }

    @Override
    public StateSet difference(StateSet elem) {
        return new LongStateSet(this.getFrame(), this.setId & ~((LongStateSet) elem).setId );
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }

        LongStateSet that = (LongStateSet) o;

        return this.setId == that.setId;
    }

    @Override
    public Set<String> toStringSet() {
        List<String> states = this.getFrame().getStates();
        Set<String> stringSet = new HashSet<>();
        int stateIndex = 0;
        while ((this.setId >> stateIndex) != 0) {
            int mask = 1 << stateIndex;
            if ((mask & this.setId) != 0) { //if the state i is in the set
                stringSet.add(states.get(stateIndex));
            }
            stateIndex++;
        }
        return stringSet;
    }

    @Override
    public int hashCode() {
        return (int) (this.setId ^ (this.setId >>> 32));
    }

    @Override
    public boolean isEmpty() {
        return this.setId == 0;
    }

    @Override
    public boolean includesOrEquals(StateSet that) {
        long setId1 = ((LongStateSet) that).setId;
        return (setId1 & this.setId) == setId1;
    }

    @Override
    public int card() {
        return Long.bitCount(this.setId);
    }

    @Override
    public int compareTo(StateSet o) {
        LongStateSet that = (LongStateSet) o;
        if(this.equals(that)) {
            return 0;
        }
        int diff = this.card() - o.card();
        if(diff == 0) {
            return this.setId > that.setId ? 1 : -1;
        }
        else {
            return diff;
        }
    }
}