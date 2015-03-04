package fr.inria.tacoma.bft.core.frame.impl;

import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.frame.StateSet;

import java.util.Set;


/**
 * Base class for state sets. This class implements the common features
 * which are not implementation specific.
 */
abstract class AbstractStateSet implements StateSet {

    private final FrameOfDiscernment frame;

    public AbstractStateSet(FrameOfDiscernment frame) {
        this.frame = frame;
    }

    @Override
    public FrameOfDiscernment getFrame() {
        return frame;
    }

    @Override
    public StateSet complement() {
        return this.getFrame().fullIgnoranceSet().difference(this);
    }

    @Override
    public String toString() {
        Set<String> stringSet = this.toStringSet();
        StringBuilder builder = new StringBuilder("[");
        for (String state : stringSet) {
            if(builder.length() > 1) { // if this is not the first state in the set
                builder.append(", ");
            }
            builder.append('"').append(state).append('"');

        }
        return builder.append("]").toString();
    }
}
