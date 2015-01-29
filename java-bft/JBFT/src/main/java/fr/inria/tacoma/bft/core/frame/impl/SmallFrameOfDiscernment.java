package fr.inria.tacoma.bft.core.frame.impl;

import fr.inria.tacoma.bft.core.frame.StateSet;
import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;

import java.util.Collection;
import java.util.List;

/**
 * This concrete implementation of FrameOfDiscernment is used in when the
 * frame of discernment contains less than 64 elements.
 */
public class SmallFrameOfDiscernment extends FrameOfDiscernment {

    private final  LongStateSet fullIgnoranceStateSet;

    public SmallFrameOfDiscernment(String name, String... elements) {
        super(name, elements);
        assert elements.length <= 64;
        fullIgnoranceStateSet = new LongStateSet(this,
                                                 Long.MAX_VALUE >> (63 - this.getStates().size()));
    }

    public SmallFrameOfDiscernment(String name, List<String> elements) {
        super(name, elements);
        fullIgnoranceStateSet = new LongStateSet(this,
                                                 Long.MAX_VALUE >> (63 - this.getStates().size()));
    }

    @Override
    public StateSet toStateSet(Collection<String> elements) {
        if (!this.getStates().containsAll(elements)) {
            throw new IllegalArgumentException(elements + " does not belong to " +
                    "the frame of discernment " + this + ".");
        }
        long elementSetId = elements.stream()
                .mapToInt(str -> 1 << this.getStates().indexOf(str))
                .reduce(0, (a, b) -> a + b);
        return new LongStateSet(this, elementSetId);
    }

    @Override
    public StateSet fullIgnoranceSet() {
        return fullIgnoranceStateSet;
    }

    @Override
    public StateSet emptyStateSet() {
        return new LongStateSet(this, 0);
    }
}
