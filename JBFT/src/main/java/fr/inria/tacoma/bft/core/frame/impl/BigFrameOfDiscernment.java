package fr.inria.tacoma.bft.core.frame.impl;

import fr.inria.tacoma.bft.core.frame.StateSet;
import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;

import java.math.BigInteger;
import java.util.Collection;
import java.util.List;

/**
 * This concrete implementation of FrameOfDiscernment is used in case of
 * a frame of discernment with more than 64 elements.
 */
public class BigFrameOfDiscernment extends FrameOfDiscernment {

    public BigFrameOfDiscernment(String name, String... elements) {
        super(name, elements);
    }

    public BigFrameOfDiscernment(String name, List<String> elements) {
        super(name, elements);
    }



    public StateSet toStateSet(Collection<String> elements) {
        if (!this.getStates().containsAll(elements)) {
            throw new IllegalArgumentException(elements + " does not belong to " +
                    "the frame of discernment " + this + ".");
        }
        BigInteger elementSetId = elements.stream()
                .map(str -> BigInteger.ONE.shiftLeft(this.getStates().indexOf(str)))
                .reduce(BigInteger.ZERO, (a, b) -> a.add(b));
        return new BigIntegerStateSet(this, elementSetId);
    }

    public StateSet fullIgnoranceSet() {
        BigInteger setId = BigInteger.ONE;
        setId = setId.shiftLeft(this.card())
                .subtract(BigInteger.ONE);
        return new BigIntegerStateSet(this, setId);
    }

    public StateSet emptyStateSet() {
        return new BigIntegerStateSet(this, BigInteger.ZERO);
    }
}
